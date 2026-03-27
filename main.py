import sys
import os
import re
import json
from ollama_service import OllamaService
from tools import AVAILABLE_TOOLS

BASE_SYSTEM_PROMPT = """You are a Strict Linux System Agent with persistent memory. You MUST respond in valid JSON format ONLY.

RESPONSE FORMAT:
{{
  "thought": "Your internal reasoning for this step.",
  "tool": "tool_name or null",
  "args": ["arg1", "arg2"]
}}

STRICT RULES:
1. ONLY respond with the JSON object. No extra text.
2. If a request is a general question (Knowledge/Why/What) not requiring a tool, set "tool": null and provide a detailed explanation in "thought".
3. NEVER provide manual terminal commands or instructions for the user to run.
4. "args" MUST be a list.
5. PATIENCE: Never open a file or run a command unless explicitly asked to "open", "run", or "execute". If you find something, just report it first.

AVAILABLE TOOLS:
- check_updates(): Checks for system updates.
- download_youtube(url): Downloads a YouTube video.
- convert_video(input_file, output_format): Converts video files.
- get_system_status(): Returns CPU/RAM usage.
- gpu_status(): Returns GPU usage.
- list_directory(path=".", show_sizes=False, include_dir_size=False): List files/folders with optional size info.
- open_file(path): Opens a file or directory using the default system handler.
- run_safe_command(base_cmd, *args): Runs whitelisted command (ls, cat, df, uptime, nvidia-smi, yt-dlp, ffmpeg).

TIPS:
- Paths: "~" expands to your home directory.
- Memory: Recap what you did recently.
- Truth: Trust the tool output. If the tool lists it, IT IS THERE.

REASONING STEPS:
1. If the user asks for a file (e.g., "firebase video"):
   a. Call list_directory(parent_path) to see what's actually there.
   b. Look at the result. If you see "firebase intigration.mp4", THAT IS THE FILE.
   c. Use the EXACT name found in the tool output for your next action.
2. NEVER say a file is missing if it was in the tool result.
3. If no match is found, apologize and ask for the correct name.

MEDIA REASONING:
- If asked to "check" or "show formats": Use run_safe_command("yt-dlp", "-F", url).
- NEVER call download_youtube(url) unless the word "download" or "get" is used with intent to store.

CURRENT MEMORY:
{memory_context}
"""

class MemoryManager:
    def __init__(self, filename="memory.json"):
        self.filename = filename
        self.memory = self.load()

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    return json.load(f)
            except: pass
        return {"last_file": "None", "last_command": "None", "notes": []}

    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def update(self, tool_name, result, args):
        self.memory["last_command"] = f"{tool_name}({', '.join(map(str, args))})"
        
        # Heuristic extraction
        if tool_name == "download_youtube" and "Successfully" in result:
            match = re.search(r"as (.*)$", result)
            if match: self.memory["last_file"] = match.group(1).strip()
        elif tool_name == "convert_video" and "Converted" in result:
            match = re.search(r"Converted to (.*)$", result)
            if match: self.memory["last_file"] = match.group(1).strip()
            
        self.save()

    def get_context(self):
        return json.dumps(self.memory, indent=2)

def extract_json(text):
    """Extract and parse JSON from the LLM response."""
    try:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except Exception:
        return None

def main():
    service = OllamaService()
    mem = MemoryManager()
    
    print("Checking for available Ollama models...")
    models = service.list_models()
    
    if not models:
        print("No models found. Please make sure Ollama is running.")
        sys.exit(1)
        
    print("\nAvailable Models:")
    for i, model in enumerate(models):
        print(f"{i + 1}. {model}")
        
    choice = input("\nSelect a model (number) or press Enter for the first one: ").strip()
    selected_model = models[int(choice) - 1] if choice.isdigit() and 0 < int(choice) <= len(models) else models[0]
            
    print(f"\nUsing model: {selected_model}")
    print("Agent is ready! (type 'quit' to exit)")
    
    # Message history with sliding window
    history = []
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
            
        history.append({"role": "user", "content": user_input})
        
        # Dynamic prompt with current memory
        current_system_prompt = BASE_SYSTEM_PROMPT.format(memory_context=mem.get_context())
        messages = [{"role": "system", "content": current_system_prompt}] + history[-14:] # Keep last 14 messages
        
        last_tool_call = None
        
        for step in range(1, 6):
            print(f"\r[Step {step}] Thinking...", end="", flush=True)
            
            retries = 1
            data = None
            full_raw_response = ""
            
            while retries >= 0:
                full_raw_response = ""
                for chunk in service.chat(selected_model, messages, stream=True):
                    full_raw_response += chunk
                
                data = extract_json(full_raw_response)
                if data: break
                if retries > 0:
                    messages.append({"role": "user", "content": "Error: Invalid JSON. Respond ONLY in JSON."})
                retries -= 1
            
            if not data: break
                
            print(f"\r[Step {step}] Thought: {data.get('thought', '...')}")
            
            tool_name = data.get("tool")
            args = data.get("args", [])
            
            if not tool_name or tool_name.lower() == "null":
                # Final step: print the thought as the actual response
                thought = data.get('thought', 'Found the results.')
                print(f"\nResponse: {thought}")
                history.append({"role": "assistant", "content": full_raw_response})
                break
                
            if (tool_name, str(args)) == last_tool_call:
                print(f"\n[!] Error: Repeated identical tool call detected. Stopping.")
                break
            
            last_tool_call = (tool_name, str(args))
            
            if tool_name in AVAILABLE_TOOLS:
                print(f"[*] Executing {tool_name} with {args}...")
                try:
                    tool_result = AVAILABLE_TOOLS[tool_name](*args)
                    print(f"[*] Result: {tool_result}")
                    
                    # Update Memory
                    mem.update(tool_name, tool_result, args)
                    
                    # Store in message history for context
                    history.append({"role": "assistant", "content": full_raw_response})
                    history.append({"role": "user", "content": f"[SYSTEM]: Tool {tool_name} returned: {tool_result}"})
                    
                    # Update local messages for the next step in the loop
                    messages.append({"role": "assistant", "content": full_raw_response})
                    messages.append({"role": "user", "content": f"[SYSTEM]: Tool {tool_name} returned: {tool_result}"})
                except Exception as e:
                    print(f"\n[!] Tool Execution Error: {e}")
                    error_msg = f"I apologize, but I encountered an error while trying to execute the {tool_name} tool: {e}"
                    print(f"Response: {error_msg}")
                    break
            else:
                print(f"\n[!] Tool Error: '{tool_name}' not found.")
                apology = f"I apologize, but '{tool_name}' is not currently in my authorized toolset. I can only perform actions via my available tools: {', '.join(AVAILABLE_TOOLS.keys())}."
                print(f"Response: {apology}")
                break

        if step == 1 and not data:
            print("\nResponse: I apologize, but I encountered an internal error processing your request.")

if __name__ == "__main__":
    main()
