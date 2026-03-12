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
                
                received_answer = False
                
                # Streaming POST request
                with client.stream("POST", API_URL, json=payload) as response:
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                chunk_type = data.get("type")
                                
                                if SESSION_ID is None and "session_id" in data:
                                    SESSION_ID = data["session_id"]
                                
                                if chunk_type == "thought":
                                    print(f"\033[90m\033[3m[Thought: {data.get('content', '')}]\033[0m")
                                elif chunk_type == "token":
                                    print(data.get("content", ""), end="", flush=True)
                                    received_answer = True
                                elif chunk_type == "clarification":
                                    print(f"\n\033[93m[Clarification Needed]: {data.get('content', '')}\033[0m")
                                    received_answer = True
                                elif chunk_type == "error":
                                    print(f"\n\033[1m\033[91m[Error: {data.get('content', '')}]\033[0m")
                                elif chunk_type == "answer_metadata":
                                    sources = data.get("sources", [])
                                    if sources:
                                        print("\n\n\033[93mSources Used:\033[0m")
                                        for idx, s in enumerate(sources):
                                            doc = s.get("doc_id", "Unknown") if isinstance(s, dict) else s
                                            print(f"  [{idx+1}] {doc}")
                            except json.JSONDecodeError:
                                continue

                print("\n") # New line after the stream finishes

                if not received_answer:
                    print("\033[95mCopilot: \033[0m[Generation failed or was blocked by consistency guardrails. No answer returned.]")
                    
                print("\n")
                
            except KeyboardInterrupt:
                print("\nEnding session.")
                break
            except Exception as e:
                print(f"\n\033[91m[Connection Error]: {str(e)}\033[0m\n")

if __name__ == "__main__":
    chat()
