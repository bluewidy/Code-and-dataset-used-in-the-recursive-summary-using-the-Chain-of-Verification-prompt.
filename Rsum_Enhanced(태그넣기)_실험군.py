import requests
import json
import re
import os
import sys
from datetime import datetime

# <<< Tee 클래스 (유지) >>>
class Tee:
    def __init__(self, filename, mode='w', encoding='utf-8'):
        self.file = open(filename, mode, encoding=encoding)
        self.stdout = sys.stdout
        sys.stdout = self
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

# --- Ollama LLM 호출 (결정론적 설정 유지) ---
def call_ollama_llm(system_prompt: str, user_prompt: str, task_type: str, model: str = "gpt-oss:20b") -> str:
    print("=" * 50)
    print(f"🤖 Ollama '{model}' 모델에게 '{task_type}' 작업을 요청합니다...")
    print("-" * 50)
    
    try:
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": True,
            "options": {
                "temperature": 0.0,
                "seed": 0,
            }
        }
        
        print(f"⏳ 실시간 응답 생성 중... (Start)\n" + "-"*30)
        
        response = requests.post(url, json=payload, stream=True, timeout=(10, 600))
        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                try:
                    json_chunk = json.loads(decoded_line)
                    content_chunk = json_chunk.get("message", {}).get("content", "")
                    if json_chunk.get("done", False): break
                    print(content_chunk, end="", flush=True)
                    full_response += content_chunk
                except json.JSONDecodeError: continue
        
        print("\n" + "-"*30 + "\n✅ 응답 생성 완료 (End)")
        return full_response.strip()
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Ollama API 호출 오류: {e}")
        return f"LLM 호출 오류: {e}"

# --- Enhanced Rsum (No Speaker Separation, Tag+Quote Only) ---
class EnhancedRecursiveSummarizer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        
        self.response_system_prompt = "You are an advanced AI language model designed to engage in personality-based conversations."
        self.response_instruction = "You are an advanced AI designed for engaging in natural, personality-based conversations. You will be provided with a memory, containing the personal preferences and experiences of speakers (the assistant and the user), as well as a dialogue context. When responding, consider maintaining a conversational and fluent tone. Responses should be contextually relevant, consistent with given memory, aiming to keep the conversation flowing. Human queries are labeled 'User:', while your replies are marked 'Assistant:'. Your goal is to provide engaging and coherent responses based on the dialogue context provided."

    # -----------------------------------------------------------------
    # 💡 Step 1: 메모리 재귀 요약 + 태깅 + 인용구
    # -----------------------------------------------------------------
    def _generate_memory(self, prev_memory: str, current_context: str, session_num: int) -> str:
        print("\n" + "---" * 10)
        print(f"🌀 (Enhanced Rsum) 메모리 생성 - S{session_num} (Rsum + Tagging + Quote)")
        print("---" * 10)
        
        memory_context_for_prompt = prev_memory if prev_memory and prev_memory.strip() else "none"

        # 💡 기존 메모리 재귀 요약하라는 프롬프트에 태그와 인용구 규칙을 추가가
        instruction = """You are an advanced AI language model with the ability to store and update a memory to keep track of key personality information for both the user and the system. You will receive a previous memory and a dialogue context. Your goal is to update the memory by incorporating the new information while ensuring that the memory does not exceed 20 sentences. "
        
Analyze the [Dialogue Context] and extract facts.
Format each fact as: **[TAG] Fact content (Quote: "exact phrase from dialogue")**

**Allowed Tags:**
1. **[PLAN]** (Future / Intentions)
   - Use for future events, intentions, or wishes.
   - **Ex:** "I'm going to buy a new laptop next week." -> **[PLAN]** Plans to buy a laptop.
   - **Ex:** "I want to visit Japan someday." -> **[PLAN]** Wants to visit Japan.
   - **Anti-Ex:** "I am typing on my laptop." -> **[ACTION]** (Current activity).

2. **[ACTION]** (Current Session / Specific Past Event)
   - Use for specific actions done *during* or *immediately before* this conversation.
   - **Ex:** "I just finished my lunch." -> **[ACTION]** Finished lunch.
   - **Ex:** "I am reading a book right now." -> **[ACTION]** Is reading a book.
   - **Anti-Ex:** "I read books every night." -> **[FACT]** (Habit/Routine).

3. **[PREFERENCE]** (Explicit Likes/Dislikes)
   - Use ONLY if there is a clear **emotional verb** (love, hate, enjoy, prefer).
   - **Ex:** "I absolutely love spicy food." -> **[PREFERENCE]** Loves spicy food.
   - **Ex:** "I hate rainy days." -> **[PREFERENCE]** Hates rainy days.
   - **Anti-Ex:** "I eat spicy food often." -> **[FACT]** (Habit, emotion not stated).

4. **[FACT]** (Attributes / Habits / Biography)
   - Use for static truths, job, origin, habits, or general abilities.
   - **Ex:** "I work as a so군' 모듈입니다. ---")