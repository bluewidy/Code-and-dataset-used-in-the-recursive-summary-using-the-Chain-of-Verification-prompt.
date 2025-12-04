# Code-and-dataset-used-in-the-recursive-summary-using-the-Chain-of-Verification-prompt.
"Chain of Verification 프롬프트를 활용한 재귀적 요약" 에서 실험하는데 사용한 데이터셋과 파이썬 스크립트를 첨부하였습니다.

파이썬 스크립트를 가동하는데 사용된 데이터셋 : 
dialogue_Chat_1_Emi_Elise_S1-S7.txt

재귀적 메모리 요약만을 수행하는 파이썬 스크립트 : Rsum_Only_대조군.py

재귀적 메모리 요약(이하 Rsum) + Chain of Verificaton(이하 CoVe)을 수행하는 파이썬 스크립트 : Rsum + CoVe_실험군.py

Rsum + CoVe + 태깅 을 수행하는 파이썬 스크립트 : Rsum_CoVe_태그넣기_실험군.py

Rsum + 태깅을 수행하는 파이썬 스크립트 : Rsum_Enhanced(태그넣기)_실험군.py


Rsum + CoVe을 수행한 결과는 experiment_REALTALK_Chat_1_Emi_Elise._20251125_Rsum_CoVe.txt 에 기록되어 있습니다.

Rsum + CoVe + 태깅을 수행한 결과는 experiment_REALTALK_Chat_1_Emi_Elise._20251127_Rsum_CoVe_Tag넣기_.txt 에 기록되어 있습니다.

Rsum + 태깅을 수행한 결과는 experiment_REALTALK_Chat_1_Emi_Elise._20251127_Rsum_Enhanced.txt 에 기록되어 있습니다.

이 실험을 진행하는데 사용한 dialogue_Chat_1_Emi_Elise_S1-S7.txt 는 REALTALK: A 21-Day Real-World Dataset for Long-Term Conversation 데이터셋
(https://github.com/danny911kr/REALTALK) 에서 제공하는 Chat_1_Emi_Elise.json 에서 "clean_text"와 "speaker" 를 추출하여 User, Assistant로 발화자를 설정하여
가공한 것입니다. (대화 내용은 전혀 건들지 않았습니다.)
