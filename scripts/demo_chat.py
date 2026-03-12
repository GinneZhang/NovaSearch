import json
import httpx
import sys

API_URL = "http://localhost:8000/ask"
SESSION_ID = None

def chat():
    global SESSION_ID
    print("==================================================")
    print("Welcome to NovaSearch Enterprise Copilot CLI")
    print("Type 'exit' or 'quit' to close the session.")
    print("==================================================\n")
    
    with httpx.Client(timeout=60.0) as client:
        while True:
            try:
                user_input = input("\033[94mYou:\033[0m ")
                if user_input.lower() in ['exit', 'quit']:
                    print("Ending session.")
                    break
                    
                if not user_input.strip():
                    continue

                payload = {"query": user_input}
                if SESSION_ID:
                    payload["session_id"] = SESSION_ID

                print("\033[95mCopilot:\033[0m ", end="", flush=True)
                
                has_answer = False
                
                # Streaming POST request
                with client.stream("POST", API_URL, json=payload) as response:
                    for line in response.iter_lines():
                        if not line:
                            continue
                        
                        try:
                            chunk = json.loads(line)
                            
                            if SESSION_ID is None and "session_id" in chunk:
                                SESSION_ID = chunk["session_id"]

                            chunk_type = chunk.get("type")
                            content = chunk.get("content", "")
                            
                            if chunk_type == "thought":
                                # Print thoughts in dim/italic gray
                                print(f"\033[90m\033[3m[Thought: {content}]\033[0m")
                            elif chunk_type == "answer":
                                # Print final answer
                                print(f"{content}")
                                has_answer = True
                                
                                # Print sources if available
                                sources = chunk.get("sources", [])
                                if sources:
                                    print("\n\033[93mSources Used:\033[0m")
                                    for idx, s in enumerate(sources):
                                        doc = s.get("doc_id", "Unknown") if isinstance(s, dict) else s
                                        print(f"  [{idx+1}] {doc}")
                            elif chunk_type == "error":
                                print(f"\n\033[1m\033[91m[Error: {content}]\033[0m")
                                
                        except json.JSONDecodeError:
                            print(f"\n[Raw Output]: {line}")
                            
                if not has_answer:
                    print("\033[95mCopilot: \033[0m[Generation failed or was blocked by consistency guardrails. No answer returned.]")
                    
                print("\n")
                
            except KeyboardInterrupt:
                print("\nEnding session.")
                break
            except Exception as e:
                print(f"\n\033[91m[Connection Error]: {str(e)}\033[0m\n")

if __name__ == "__main__":
    chat()
