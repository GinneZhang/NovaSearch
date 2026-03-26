import json
import httpx
import sys
import os
from dotenv import load_dotenv

# 自动加载项目根目录下的 .env 文件
load_dotenv()

API_URL = "http://localhost:8000/ask"
SESSION_ID = None

def chat():
    global SESSION_ID
    
    # 核心修改点：获取 API Key 并组装 Headers
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("\033[1m\033[91m[Fatal Error]: API_KEY environment variable is not set.\033[0m")
        print("Please ensure you have an .env file with API_KEY=\"your_secret_key\" in the root directory.")
        sys.exit(1)
        
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    print("==================================================")
    print("Welcome to AsterScope Enterprise Copilot CLI")
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
                
                # 核心修改点：在请求中携带 headers 进行鉴权
                with client.stream("POST", API_URL, json=payload, headers=headers) as response:
                    
                    # 拦截并打印由于认证失败导致的 HTTP 错误 (例如 401)
                    if response.status_code != 200:
                        error_msg = response.read().decode("utf-8")
                        print(f"\n\033[1m\033[91m[Server Error HTTP {response.status_code}]: {error_msg}\033[0m")
                        print("\n")
                        continue
                        
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

                if not received_answer and response.status_code == 200:
                    print("\033[95mCopilot: \033[0m[Generation failed or was blocked by consistency guardrails. No answer returned.]")
                    print("\n")
                
            except KeyboardInterrupt:
                print("\nEnding session.")
                break
            except Exception as e:
                print(f"\n\033[91m[Connection Error]: {str(e)}\033[0m\n")

if __name__ == "__main__":
    chat()