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

# --- (수정) 유틸리티 함수: Ollama LLM 호출 (Temperature 0 적용) ---
def call_ollama_llm(system_prompt: str, user_prompt: str, task_type: str, model: str = "gpt-oss:20b") -> str:
    """
    Ollama를 통해 로컬 LLM API를 호출하는 함수입니다.
    (수정: options에 temperature=0.0 추가하여 무작위성 제거)
    """
    print("=" * 50)
    print(f"🤖 Ollama '{model}' 모델에게 '{task_type}' 작업을 요청합니다...")
    print("-" * 50)
    print(f"\n[SYSTEM PROMPT]\n{system_prompt}\n")
    print(f"\n[USER PROMPT]\n{user_prompt}\n")
    
    try:
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            # 💡 (핵심 수정) Temperature를 0으로 설정하여 결과의 일관성 보장
            "options": {
                "temperature": 0.0,
                "seed":0
            }
        }
        response = requests.post(url, json=payload, timeout=30000) 
        response.raise_for_status()
        
        response_data = response.json()
        content = response_data['message']['content'].strip()
        
        print(f"✅ Ollama 응답 수신:\n{content}")
        return content
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Ollama API 호출 오류: {e}")
        return f"LLM 호출 오류: {e}"

# --- 실험군 (Rsum + CoVe) 클래스 ---
class VerifiedRecursiveSummarizer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        
        self.memory_system_prompt = "You are an advanced AI language model with the ability to keep track of dialog information between speakers."
        self.memory_instruction = "You are an advanced AI language model with the ability to store and update a memory to keep track of key personality information for both the user and the system. You will receive a previous memory and a dialogue context. Your goal is to update the memory by incorporating the new personality information while ensuring that the memory does not exceed 20 sentences."
        
        self.response_system_prompt = "You are an advanced AI language model designed to engage in personality-based conversations."
        self.response_instruction = "You are an advanced AI designed for engaging in natural, personality-based conversations. You will be provided with a memory, containing the personal preferences and experiences of speakers (the assistant and the user), as well as a dialogue context. When responding, consider maintaining a conversational and fluent tone. Responses should be contextually relevant, consistent with given memory, aiming to keep the conversation flowing. Human queries are labeled 'User:', while your replies are marked 'Assistant:'. Your goal is to provide engaging and coherent responses based on the dialogue context provided."

    # Step 1: 요약 생성 (유지)
    def _generate_memory(self, prev_memory: str, current_context: str, session_num: int) -> str:
        print("\n" + "---" * 10)
        print(f"🌀 (Step 1) 요약 생성 (R-Sum) - S{session_num}")
        print("---" * 10)
        
        memory_context_for_prompt = prev_memory if prev_memory and prev_memory.strip() else "none"
        
        system_prompt = self.memory_system_prompt
        user_prompt = f"""**Instruction** {self.memory_instruction}

**Test** [Previous Memory] {memory_context_for_prompt} [Dialogue Context] {current_context} [Updated Memory]"""
        
        task_name = f"Memory Generation (S{session_num})"
        return call_ollama_llm(system_prompt, user_prompt, task_name, self.model_name)

    # Step 2: Diff 생성 (유지)
    def _get_memory_diff(self, old_memory: str, new_memory: str) -> str:
        print("\n" + "---" * 10)
        print("🔍 (Step 2) 'Diff' 생성: 이전 메모리와 새 요약본 비교 중...")
        print("---" * 10)
        
        system_prompt = "You are an AI analyzer. Your task is to compare a 'Previous Memory' with a 'New Memory' and extract *only* the facts that are NEW or MODIFIED in the 'New Memory'. Facts that are simply carried over (unchanged) should be ignored."
        user_prompt = f"""
[Previous Memory]
{old_memory}

[New Memory]
{new_memory}

[Instruction]
List all facts that are NEWLY ADDED or SIGNIFICANTLY MODIFIED in the '[New Memory]'.
- If no new or modified facts are found, output the exact string "NO CHANGES".
- Focus only on the delta (the changes).

[New or Modified Facts]
"""
        diff_facts = call_ollama_llm(system_prompt, user_prompt, "Memory Diff Generation", self.model_name)
        
        if "NO CHANGES" in diff_facts.upper() or len(diff_facts) < 5:
            print("ℹ️ (Step 2) 'Diff' 결과: 변경 사항 없음.")
            return ""
        return diff_facts

    # Step 3: 질문 생성 (유지)
    def _generate_verification_questions(self, facts_to_verify: str) -> list[str]:
        print("\n" + "---" * 10)
        print(f"❓ (Step 3) 'Diff' 기반 검증 질문 생성 중...")
        print("---" * 10)
        
        system_prompt = "You are an AI assistant. Given a list of facts, generate a set of simple, verifiable questions to check if these facts are true."
        user_prompt = f"""
[Facts to Verify]
{facts_to_verify}

[Instruction]
Generate a list of verification questions based *only* on the facts provided above.
- Each question should be on a new line.
- Do NOT use hyphens or numbers.

[Verification Questions]
"""
        questions_text = call_ollama_llm(system_prompt, user_prompt, "Question Generation (Diff)", self.model_name)
        questions = [q.strip() for q in questions_text.split('\n') if q.strip() and '?' in q]
        
        print(f"✅ {len(questions)}개의 검증 질문 생성 완료.")
        return questions

    # -----------------------------------------------------------------
    # 💡 (수정) Step 4: 자유 서술형 검증 (True/False 제약 삭제)
    # -----------------------------------------------------------------
    def _execute_verification_plan(self, questions: list[str], 
                                 current_session_context: str) -> dict:
        print("\n" + "---" * 10)
        print(f"🛡️ (Step 4) '최소 컨텍스트' 검증 실행 중... (자유 서술형 답변)")
        print("---" * 10)

        # (수정) 단답형 제약 삭제 및 설명 요청
        system_prompt = "You are an AI fact-checker. Verify the question based *strictly* on the provided [Context]. Provide exact quotes."
        
        verified_answers = {}
        for q in questions:
            user_prompt = f"""
[Context]
{current_session_context}

[Question]
{q}

[Instruction]
Answer based *strictly* on the context.
1. **Logic Check:** Do not conflate two different people's attributes.
2. **State Check:** Distinguish between past origin and current status.
3. **Nuance Check:** Distinguish between "ability" (can do) and "interest" (likes to watch/read).
4. **No Inference:** Do not assume unstated preferences based on facts.
5. **(CRITICAL) Provide a direct quote.**

[Few-Shot Examples (Domain-Agnostic)]

Example 1 (Origin vs Residence - Logic: Past != Current):
Context: User: "I grew up in Texas, but I've been living in Tokyo for 5 years."
Question: Does the user live in Texas?
Answer: No, the user is *from* Texas but currently *lives* in Tokyo.
Quote: "grew up in Texas... living in Tokyo"

Example 2 (Entity Binding - Logic: Speaker A's action != Speaker B's location):
Context: User: "I'm eating a burger." Assistant: "Delicious! I'm reading a book in the library."
Question: Is the user eating a burger in the library?
Answer: No. The User is eating a burger, but the location "library" applies to the Assistant.
Quote: "User: ...eating a burger", "Assistant: ...in the library"

Example 3 (Ability vs Interest - Logic: Cannot do != Dislike):
Context: Assistant: "I can't play the guitar, but I love listening to rock music."
Question: Is the assistant uninterested in guitars?
Answer: No. She lacks the ability to play (can't play), but she has an interest in the music (loves listening).
Quote: "can't play... love listening"

Example 4 (Fact vs Inference - Logic: Possession != Profession/Hobby):
Context: Assistant: "I own a vintage Ferrari."
Question: Is the assistant a professional racing driver?
Answer: Not mentioned. Owning a car does not imply being a professional driver.
Quote: "own a vintage Ferrari"

Example 5 (Plan vs Fact - Logic: Future != Present):
Context: User: "I plan to study French next year, currently I speak Spanish."
Question: Does the user speak French?
Answer: No, speaking French is a future plan. Currently, they speak Spanish.
Quote: "plan to study... next year"

Example 6 (Selection Nuance - Logic: A or B != A and B):
Context: User: "I'll buy either the red shirt or the blue one."
Question: Is the user buying both shirts?
Answer: No, the user is choosing between the two ("either... or"), not buying both.
Quote: "either the red shirt or the blue one"

Example 7 (External Knowledge - Logic: General != Specific):
Context: User: "I work at a tech company in Silicon Valley."
Question: Does the user work for Google?
Answer: Not mentioned. The context says "a tech company", it does not specify "Google".
Quote: "work at a tech company"

[Answer]
"""
            task_name = f"Fact Verification (Q: {q[:40]}...)"
            answer = call_ollama_llm(system_prompt, user_prompt, task_name, self.model_name)
            verified_answers[q] = answer.strip()
        
        print("✅ 모든 Diff 질문 검증 완료.")
        return verified_answers

    # -----------------------------------------------------------------
    # 💡 (수정 v3.3) Step 5: 초안 교정 (Draft Correction) & GC
    # -----------------------------------------------------------------
    def _reconstruct_final_memory(self, draft_memory: str, verified_answers: dict) -> str:
        # (수정) 입력 변수명을 old_memory -> draft_memory로 변경하여 의미 명확화
        
        qa_pairs_str = ""
        for q, a in verified_answers.items():
            qa_pairs_str += f"Q: {q}\nA: {a}\n\n"

        print("\n" + "---" * 10)
        print("📝 (Step 5) 메모리 교정 및 재구성 (Correction & GC)...")
        print("---" * 10)

        # (수정) 프롬프트: 병합(Merge)이 아니라 수정(Correct/Refine)에 초점
        system_prompt = "You are an AI memory editor. Your task is to correct and polish the [Draft Memory] based on the [Verification Results]."
        user_prompt = f"""
[Draft Memory]
{draft_memory}

[Verification Results (Fact-Check)]
{qa_pairs_str}

[Instruction]
Refine the [Draft Memory] to create the [Final Verified Memory].
1. **Correction:** If the [Verification Results] contradict any statement in the draft, **rewrite or remove** that statement in the draft. (Trust the Verification Results).
2. **Garbage Collection:** Remove any [PLAN] or [INTENTION] from the draft that is clearly outdated or completed based on the context.
3. **Constraint:** Ensure the final output is a concise list (under 20 sentences).

[Final Verified Memory]
"""
        final_memory = call_ollama_llm(system_prompt, user_prompt, "Final Memory Reconstruction", self.model_name)
        return final_memory

    # -----------------------------------------------------------------
    # 💡 메인 실행기 (수정)
    # -----------------------------------------------------------------
    def process_dialogue(self, past_sessions: list[str], current_context: str) -> str:
        print(f"🧑‍💻 {self.__class__.__name__}가 '{self.model_name}' 모델로 초기화되었습니다.")
        print("\n🧠 (v3.3: Draft Correction 모드) 프로세스를 시작합니다...")

        current_verified_summary = "" 
        all_sessions = past_sessions 
        
        if not all_sessions:
            print("⚠️ 경고: 'past_sessions'가 비어있습니다.")
        else:
            for i, session_context in enumerate(all_sessions):
                session_number = i + 1 
                print(f"\n\n{'='*25} Session {session_number} 처리 중 {'='*25}")
                
                # Step 1: 요약 (R-Sum) -> 초안 생성
                memory_draft = self._generate_memory(
                    prev_memory=current_verified_summary, 
                    current_context=session_context, 
                    session_num=session_number
                )
                
                # Step 2: Diff -> 초안에서 새로운 부분 감지
                memory_diff = self._get_memory_diff(
                    old_memory=current_verified_summary, 
                    new_memory=memory_draft
                )
                
                if not memory_diff.strip():
                    print("ℹ️ (메인 루프) 변경분 없음. 초안을 그대로 채택합니다.")
                    current_verified_summary = memory_draft 
                    continue 

                # Step 3: 질문 생성
                questions = self._generate_verification_questions(memory_diff)
                if not questions:
                    print("ℹ️ (메인 루프) 질문 생성 실패. 초안을 그대로 채택합니다.")
                    current_verified_summary = memory_draft
                    continue 

                # Step 4: 검증 (Logic-based 7-Shot)
                verified_answers = self._execute_verification_plan(
                    questions=questions, 
                    current_session_context=session_context
                )
                
                # Step 5: 재구성 (Draft Correction)
                # 💡 (핵심 수정) old_memory가 아니라 memory_draft를 넘깁니다.
                final_memory = self._reconstruct_final_memory(
                    draft_memory=memory_draft, 
                    verified_answers=verified_answers
                )
                
                current_verified_summary = final_memory
                print(f"\n--- ⭐️ Session {session_number} 최종 검증된 메모리 ⭐️ ---\n{current_verified_summary}\n----------------------------------")
            
            print(f"✅ {len(all_sessions)}개 과거 세션의 메모리 업데이트 완료.")

        print("💬 최종 응답 생성 중... (컨텍스트: S_N)")
        final_memory_context = current_verified_summary if current_verified_summary.strip() else "none"
        
        system_prompt = self.response_system_prompt
        user_prompt = f"""**Instruction** {self.response_instruction}

**Test** [Previous Memory] {final_memory_context} [Dialogue Context] {current_context} [Response] 
"""
        return call_ollama_llm(system_prompt, user_prompt, "Final Response Generation", self.model_name)

if __name__ == "__main__":
    print("--- ⚠️ 이 파일은 'Rsum_CoVe_실험군' 모듈입니다. ---")
    print("실험을 실행하려면 'run_realtalk_experiment.py'를 실행해주세요.")


