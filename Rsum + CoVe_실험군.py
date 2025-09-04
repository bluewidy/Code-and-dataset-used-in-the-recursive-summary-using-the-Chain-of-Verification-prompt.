import requests
import json
import re
import os
import sys
from datetime import datetime

# <<< 모든 print 출력을 파일과 콘솔로 동시에 보내는 클래스 >>>
class Tee:
    """
    모든 print 출력을 터미널(stdout)과 로그 파일로 동시에 보내는 역할을 하는 클래스.
    """
    def __init__(self, filename, mode='w', encoding='utf-8'):
        self.file = open(filename, mode, encoding=encoding)
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __del__(self):
        # 객체가 소멸될 때 원래 stdout으로 복원하고 파일을 닫습니다.
        sys.stdout = self.stdout
        self.file.close()

# --- 유틸리티 함수: Ollama LLM 호출 ---
def call_ollama_llm(prompt: str, task_type: str, model: str = "gpt-oss:20b") -> str:
    """
    Ollama를 통해 로컬 LLM API를 호출하는 함수입니다.
    """
    print("=" * 50)
    print(f"🤖 Ollama '{model}' 모델에게 '{task_type}' 작업을 요청합니다...")
    print("-" * 50)
    print(f"\n")
    try:
        url = "http://localhost:11434/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}
        response = requests.post(url, data=json.dumps(payload))
        response.raise_for_status()
        response_data = response.json()
        generated_text = response_data.get("response", "").strip()
        print(f"✅ 작업 완료!")
        return generated_text
    except requests.exceptions.RequestException as e:
        print(f"❌ Ollama API 호출 중 오류가 발생했습니다: {e}")
        print("Ollama 서버가 실행 중인지, 포트 번호(11434)가 올바른지 확인해주세요.")
        return f"오류: {e}"

# --- 모듈 2: CoVe 검증기 ---
class ChainOfVerifier:
    """
    CoVe 논문의 검증 메커니즘을 구현한 클래스.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    # CoVe의 각 단계별 프롬프트 생성 함수들...
    def _get_plan_verifications_prompt(self, context_to_verify: str) -> str:
        return f"""You are a fact-checker. Your task is to analyze the provided text and generate a list of simple, verifiable questions to check the factual claims within it. Each question should focus on a single, specific fact.\n\n[Text to Verify]\n{context_to_verify}\n\n[Verification Questions]\n"""
    
    def _get_execute_verification_prompt(self, question: str, source_context: str) -> str:
        return f"""You are an AI assistant. Please answer the following question based ONLY on the provided [Source Context]. Provide a concise and direct answer. If the answer is not in the context, say "I don't know".\n\n[Source Context]\n{source_context}\n\n[Question]\n{question}\n\n[Answer]\n"""

    def _get_final_verification_prompt(self, original_text: str, verifications: dict) -> str:
        verification_log = "\n".join([f"- Q: {q}\n- A: {a}" for q, a in verifications.items()])
        return f"""You are a revision assistant. You will be given an original text and a list of verification question-answer pairs. Your task is to revise the original text to ensure it is consistent with the verification results.\n - If a fact in the original text is contradicted by a verification answer, correct it.\n - If the original text is consistent with the verification, keep it as is.\n - Do not add new information that is not supported by the original text or the verifications. \n\n[Original Text]\n{original_text}\n\n[Verification Log]\n{verification_log}\n\n[Revised Text]\n"""

    def verify(self, text_to_verify: str, source_context: str) -> str:
        """
        주어진 텍스트에 대해 전체 CoVe 파이프라인을 실행합니다.
        """
        print("\n--- 🛡️ CoVe 검증 시작 ---")
        # 1. 검증 계획
        plan_prompt = self._get_plan_verifications_prompt(text_to_verify)
        questions_text = call_ollama_llm(plan_prompt, "Plan Verifications", self.model_name)
        questions = re.findall(r'^\d+\.\s*(.*)', questions_text, re.MULTILINE)
        
        if not questions:
            print("⚠️ 검증 질문을 생성하지 못했습니다. 원본 메모리를 그대로 사용합니다.")
            return text_to_verify
        print(f"생성된 검증 질문: {questions}\n")

        # 2. 검증 실행
        verifications = {}
        for q in questions:
            answer_prompt = self._get_execute_verification_prompt(q, source_context)
            answer = call_ollama_llm(answer_prompt, f"Execute Verification", self.model_name)
            verifications[q] = answer
        print(f"검증 Q&A 결과:\n{json.dumps(verifications, indent=2, ensure_ascii=False)}\n")

        # 3. 최종 검증된 텍스트 생성
        final_prompt = self._get_final_verification_prompt(text_to_verify, verifications)
        verified_text = call_ollama_llm(final_prompt, "Final Verification", self.model_name)
        print("--- 🛡️ CoVe 검증 종료 ---\n")
        return verified_text

# --- 모듈 1 + 2: 통합된 검증 기반 재귀 요약기 ---
class VerifiedRecursiveSummarizer:
    """
    R-Sum의 재귀적 요약에 CoVe 검증을 통합한 최종 파이프라인 클래스.
    """
    def __init__(self, model_name: str):
        self.memory = "none"
        self.model_name = model_name
        # CoVe 검증기를 내부적으로 인스턴스화
        self.verifier = ChainOfVerifier(model_name)
        print(f"🚀 검증 기능이 탑재된 재귀 요약기가 '{model_name}' 모델로 초기화되었습니다.\n")

    def _get_memory_iteration_prompt(self, session_context: str) -> str:
        return f"""
You are an advanced AI language model with the ability to store and update a memory to keep track of key personality information for both the user and the bot. You will receive a previous memory and dialogue context. Your goal is to update the memory by incorporating the new personality information.
To successfully update the memory, follow these steps:
1. Carefully analyze the existing memory and extract the key personality of the user and bot from it.
2. Consider the dialogue context provided to identify any new or changed personality that needs to be incorporated into the memory.
3. Combine the old and new personality information to create an updated representation of the user and bot's traits.
4. Structure the updated memory in a clear and concise manner, ensuring it does not exceed 20 sentences.
Remember, the memory should serve as a reference point to maintain continuity in the dialogue and help you respond accurately to the user based on their personality.
5. Update your memory in Korean.

[Previous Memory]
{self.memory}

[Session Context]
{session_context}

[Updated Memory]
"""

    def _get_response_generation_prompt(self, current_context: str, last_session_context: str) -> str:
        return f"""
You will be provided with a memory containing personality information for both yourself and the user. Your goal is to respond accurately to the user based on the personality traits and dialogue context.
Follow these steps to successfully complete the task:
1. Analyze the provided memory to extract the key personality traits for both yourself and the user.
2. Review the dialogue history to understand the context and flow of the conversation.
3. Utilize the extracted personality traits and dialogue context to formulate an appropriate response.
4. If no specific personality trait is applicable, respond naturally as a human would.
5. Pay attention to the relevance and importance of the personality information, focusing on capturing the most significant aspects while maintaining the overall coherence of the memory.

[Latest Memory]
{self.memory}

[Last Session Dialogue]
{last_session_context}

[Current Context]
{current_context}

### CRITICAL RULE ###
BEFORE GENERATING A RESPONSE, YOU MUST CHECK THE [Last Session Dialogue]. Answer appropriately based on the conversation from the last session.

[Response]
"""

    def process_dialogue(self, past_sessions: list[str], current_context: str) -> str:
        """
        전체 대화 세션을 처리하여 최종 응답을 생성합니다.
        """
        print("🧠 전체 프로세스를 시작합니다: 메모리 업데이트 및 검증")
        
        # --- 누적된 대화 내용을 저장할 변수 ---
        accumulated_context = ""
        
        for i, session in enumerate(past_sessions, 1):
            print(f"\n{'='*20} Session {i} 처리 중 {'='*20}")
            
            # --- 현재 세션 내용을 누적 ---
            accumulated_context += session + "\n"

            # 1. R-Sum: 메모리 초안 생성
            memory_prompt = self._get_memory_iteration_prompt(session)
            draft_memory = call_ollama_llm(memory_prompt, f"Memory Draft Generation (S{i})", self.model_name)
            print(f"📝 생성된 메모리 초안 (M_{i}_draft):\n{draft_memory}")
            
            # 2. CoVe: 생성된 메모리 초안을 즉시 검증
            # --- source_context로 '누적된' 대화를 전달 ---
            verified_memory = self.verifier.verify(
                text_to_verify=draft_memory,
                source_context=accumulated_context
            )
            
            self.memory = verified_memory
            print(f"✨ 최종 검증된 메모리 (M_{i}_verified):\n{self.memory}")
        
        print("\n\n🗣️ 최종 응답 생성을 시작합니다...")
        last_session = past_sessions[-1] if past_sessions else "" # 마지막 세션 내용 추출
        response_prompt = self._get_response_generation_prompt(current_context = current_context, last_session_context=last_session)
        final_response = call_ollama_llm(response_prompt, "Final Response Generation", self.model_name)
        
        return final_response

def load_and_prepare_data(file_path: list[str]) -> tuple[list[str], str]:
    """
    여러개의 JSON 데이터셋 파일을 로드하고 스크립트에 맞는 형식으로 변환합니다.
    """
    all_past_sessions = [] # 모든 세션을 저장할 리스트를 초기화합니다.
    final_current_context = "" # 최종 current_context를 저장할 변수를 초기화합니다.

    # 입력받은 파일 경로 리스트를 순회합니다.
    for i, file_path in enumerate(file_paths):
        print(f"📄 데이터 파일 로딩 중: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")

        # JSON 파일을 열고 데이터를 읽어옵니다.
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 'sessions' 키에 있는 각 세션(발화 리스트)을 하나의 긴 문자열로 합쳐서 전체 세션 리스트에 추가합니다.
        past_sessions = ["\n".join(session) for session in data["sessions"]]
        all_past_sessions.extend(past_sessions)
        if i == len(file_paths) - 1:
            # 'current_context' 도 마찬가지로 하나의 문자열로 합치고, 봇이 답변할 차례임을 나타내는 "Assistant: "를 추가합니다.
            final_current_context = "\n".join(data["current_context"]) + "\nAssistant: "
    if not final_current_context:
        # 파일 리스트가 비어있거나 마지막 파일에 current_context가 없는 경우를 대비한 예외 처리
        raise ValueError("current_context를 설정할 수 없습니다. 마지막 데이터 파일을 확인해주세요.")
        
    return all_past_sessions, final_current_context

# --- 최종 실행 ---
if __name__ == "__main__":
    # <<< 로그 파일 설정 및 표준 출력 리디렉션 >>>
    # 실행 시점의 타임스탬프를 기반으로 고유한 로그 파일 이름 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"Rsum_CoVe_실험군_log_{timestamp}.txt"
    
    # Tee 클래스를 사용하여 모든 print 출력을 파일로도 보냄
    sys.stdout = Tee(log_filename)
    
    OLLAMA_MODEL = "gpt-oss:20b"
    file_paths = ["interactive-35.json", "interactive-68.json", "interactive-172.json","interactive-274.json","interactive-283.json","interactive-322.json","interactive-466.json"]

    try:
        # 데이터셋 파일을 로드하고 처리합니다.
        past_sessions, current_context = load_and_prepare_data(file_paths)

        print(f"\n총 {len(past_sessions)}개의 세션으로 테스트를 시작합니다.")
        
        # 통합 클래스 인스턴스화 및 실행
        summarizer = VerifiedRecursiveSummarizer(model_name=OLLAMA_MODEL)
        final_response = summarizer.process_dialogue(
            past_sessions=past_sessions,
            current_context=current_context
        )
        
        print("\n" + "="*50)
        print("✨ 최종 생성된 봇의 응답 ✨")
        print("="*50)
        print(current_context, end="")
        print(final_response)

    except (FileNotFoundError, ValueError) as e:
        print(f"데이터 처리 중 오류 발생: {e}")
    except Exception as e:
        # 실행 중 어떤 오류라도 발생하면 여기서 처리합니다.
        print(f"오류 발생: {e}")
