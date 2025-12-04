# 필요한 라이브러리들을 가져옵니다.
import requests  # HTTP 요청을 보내기 위한 라이브러리 (Ollama API 호출에 사용)
import json      # JSON 데이터를 다루기 위한 라이브러리
import re        # 정규 표현식을 사용하여 텍스트에서 패턴을 찾기 위한 라이브러리
import os        # 운영 체제와 상호 작용하기 위한 라이브러리 (파일 경로 확인 등)

# --- 유틸리티 함수: Ollama LLM 호출 (저자 구조와 동일하게 수정) ---
def call_ollama_llm(system_prompt: str, user_prompt: str, task_type: str, model: str = "gpt-oss:20b") -> str:
    """
    Ollama를 통해 로컬 LLM API를 호출하는 함수입니다.
    (수정: system과 user 프롬프트를 분리하여 저자의 API 호출 구조를 100% 재현)
    """
    # 사용자에게 현재 진행 상황을 알려주기 위한 로그 출력
    print("=" * 50)
    print(f"🤖 Ollama '{model}' 모델에게 '{task_type}' 작업을 요청합니다...")
    print("-" * 50)
    print(f"\n[SYSTEM PROMPT]\n{system_prompt}\n")
    print(f"\n[USER PROMPT]\n{user_prompt}\n")
    
    try:
        # Ollama 서버의 기본 API 주소
        url = "http://localhost:11434/api/chat" # (수정: /api/generate -> /api/chat)
        
        # (수정: 저자의 구조대로 messages 리스트 사용)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # (수정: payload 구조 변경)
        payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0.0, "seed":0}}
        
        # requests 라이브러리를 사용해 Ollama 서버에 POST 요청을 보냅니다.
        response = requests.post(url, data=json.dumps(payload))
        # 만약 HTTP 에러(예: 404, 500)가 발생하면 예외를 발생시킵니다.
        response.raise_for_status()
        
        # 응답 받은 JSON 데이터를 파이썬 딕셔너리로 변환합니다.
        response_data = response.json()
        
        # (수정: /api/chat 응답 구조에 맞게 파싱)
        generated_text = response_data.get("message", {}).get("content", "").strip()
        
        print(f"✅ 작업 완료!")
        # 생성된 텍스트를 반환합니다.
        return generated_text
    except requests.exceptions.RequestException as e:
        # API 호출 중 네트워크 오류 등이 발생하면 예외를 처리합니다.
        print(f"❌ Ollama API 호출 중 오류가 발생했습니다: {e}")
        print("Ollama 서버가 실행 중인지, 포트 번호(11434)가 올바른지, /api/chat 엔드포인트를 지원하는지 확인해주세요.")
        return f"오류: {e}"

# --- 대조군: R-Sum Only 파이프라인 ---
class RecursiveSummarizer:
    """
    CoVe 검증이 없는 오리지널 R-Sum 재귀적 요약 클래스입니다.
    이것이 실험의 대조군 역할을 합니다.
    """
    def __init__(self, model_name: str):
        self.memory = "none"  # 초기 메모리는 비어있는 상태("none")로 시작합니다.
        self.model_name = model_name
        print(f"🧑‍💻 대조군(R-Sum Only) 요약기가 '{model_name}' 모델로 초기화되었습니다.\n")

    def _get_memory_iteration_prompt(self, session_context: str) -> tuple[str, str]:
        # (수정: 저자의 프롬프트 구조(system/user)에 맞게 반환)
        
        # 1. 시스템 프롬프트 (저자의 chatgpt/robot.py 참조)
        system_prompt = "You are an advanced AI language model with the ability to keep track of dialog information between speakers."
        
        # 2. 유저 프롬프트 (저자의 main_chatgpt.py 및 llama/data_loader.py 구조 참조)
        # (참고: 저자는 Carecall/MSC 모두 'user'와 'system'으로 강제 매핑하여 요약)
        instruction = "You are an advanced AI language model with the ability to store and update a memory to keep track of key personality information for both the user and the system. You will receive a previous memory and a dialogue context. Your goal is to update the memory by incorporating the new personality information while ensuring that the memory does not exceed 20 sentences."
        
        user_prompt = f"""**Instruction** {instruction}

**Test** [Previous Memory] {self.memory} [Dialogue Context] {session_context} [Updated Memory]"""
        
        return system_prompt, user_prompt

    def _get_response_generation_prompt(self, current_context: str, last_session_context: str) -> tuple[str, str]:
        # (수정: 저자의 프롬프트 구조(system/user)에 맞게 반환)
        
        # 1. 시스템 프롬프트 (저자의 chatgpt/robot.py 참조)
        system_prompt = "You are an advanced AI language model designed to engage in personality-based conversations."
        
        # 2. 유저 프롬프트 (저자의 dataset.py RSumDataset 참조)
        # (참고: 저자는 LLM을 'Assistant'로 명명하고 'User'에게 응답하도록 지시)
        instruction = "You are an advanced AI designed for engaging in natural, personality-based conversations. You will be provided with a memory, containing the personal preferences and experiences of speakers (the assistant and the user), as well as a dialogue context. When responding, consider maintaining a conversational and fluent tone. Responses should be contextually relevant, consistent with given memory, aiming to keep the conversation flowing. Human queries are labeled 'User:', while your replies are marked 'Assistant:'. Your goal is to provide engaging and coherent responses based on the dialogue context provided."
        
        # (참고: 저자의 프롬프트는 [Last Session Dialogue]를 명시적으로 포함하지 않았습니다. 
        # MSC 데이터로더는 'window'에 현재 세션의 대화만 포함합니다.)
        user_prompt = f"""**Instruction** {instruction}

**Test** [Previous Memory] {self.memory} [Dialogue Context] {current_context} [Response] 
"""
        # (참고: [Response] 뒤에 'Assistant:'를 붙이지 않아도, 
        # current_context의 마지막이 'Assistant: '로 끝나므로 모델이 이어서 생성하게 됩니다.)
        
        return system_prompt, user_prompt

    def process_dialogue(self, past_sessions: list[str], current_context: str) -> str:
        """
        전체 대화 세션을 순차적으로 처리하여 최종 응답을 생성합니다.
        (수정: call_ollama_llm 인자 구조 변경)
        """
        print("🧠 대조군 프로세스를 시작합니다: 메모리 업데이트 Only")
        
        # 과거 대화 세션들을 하나씩 순회합니다.
        for i, session in enumerate(past_sessions, 1):
            print(f"\n{'='*20} Session {i} 처리 중 {'='*20}")
            
            # 1. R-Sum: 현재 세션 대화와 이전 메모리를 바탕으로 메모리를 생성합니다.
            system_prompt, user_prompt = self._get_memory_iteration_prompt(session)
            new_memory = call_ollama_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                task_type=f"Memory Generation (S{i})", 
                model=self.model_name
            )
            
            # 2. CoVe 검증 없이 생성된 메모리를 바로 공식 메모리로 업데이트합니다.
            self.memory = new_memory
            print(f"📝 생성 및 확정된 메모리 (M_{i}):\n{self.memory}")
        
        # 모든 과거 세션 처리가 끝나면, 최종 메모리를 이용해 사용자에게 응답할 차례입니다.
        print("\n\n🗣️ 최종 응답 생성을 시작합니다...")
        last_session = past_sessions[-1] if past_sessions else "" # 마지막 내용 추출
        
        system_prompt, user_prompt = self._get_response_generation_prompt(
            current_context=current_context, 
            last_session_context=last_session
        )
        final_response = call_ollama_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            task_type="Final Response Generation", 
            model=self.model_name
        )
        
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
        # 파일 리스트가 비어있거나, 마지막 파일에 current_context가 없는 경우를 대비한 예외 처리
        raise ValueError("current_context를 설정할 수 없습니다. 마지막 데이터 파일을 확인해주세요.")
        
    return all_past_sessions, final_current_context

# --- 최종 실행 ---
# 이 스크립트 파일이 직접 실행될 때만 아래 코드가 동작합니다.
if __name__ == "__main__":
    OLLAMA_MODEL = "gpt-oss:20b"
    file_paths = ["interactive-35.json", "interactive-68.json", "interactive-172.json","interactive-274.json","interactive-283.json","interactive-322.json","interactive-466.json"]

    try:
        # 데이터셋 파일을 로드하고 처리합니다.
        past_sessions, current_context = load_and_prepare_data(file_paths)

        print(f"\n총 {len(past_sessions)}개의 세션으로 테스트를 시작합니다.")
        
        # 대조군 클래스인 RecursiveSummarizer를 생성합니다.
        summarizer = RecursiveSummarizer(model_name=OLLAMA_MODEL)
        # 전체 대화 처리 프로세스를 실행하여 최종 응답을 얻습니다.
        final_response = summarizer.process_dialogue(
            past_sessions=past_sessions,
            current_context=current_context
        )
        
        # 최종 결과를 예쁘게 출력합니다.
        print("\n" + "="*50)
        print("✨ 최종 생성된 봇의 응답 (대조군) ✨")
        print("="*50)
        print(current_context, end="")
        print(final_response)
        
    except (FileNotFoundError, ValueError) as e:
        print(f"데이터 처리 중 오류 발생: {e}")
    except Exception as e:
        # 실행 중 어떤 오류라도 발생하면 여기서 처리합니다.
        print(f"오류 발생: {e}")



