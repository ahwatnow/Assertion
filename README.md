# Assertion

Preliminaries
===
- python3.10 >=
- pip -q install git+https://github.com/huggingface/transformers
- pip install -q datasets loralib sentencepiece
- pip -q install bitsandbytes accelerate xformers einops
- pip -q install langchain

Refine example
===
1. python3 ./refine.py --key=`Your key`
2. query `give me a python code which converts binary string considering singed/unsigned domain`
3. observe result

Examples
===
1. 파이썬 코드 생성을 요청하는 자연어 문장 데이터셋
   - https://huggingface.co/datasets/mbpp?row=1
   - https://huggingface.co/datasets/openai_humaneval?row=0
     
2. test1.py (ChatGPT 3.5) 실행 방법

   python test1.py -k=[your open-ai api key]

3. 명령에 따른 코드 생성 예시

   ("Write a python function to find the first repeated character in a given string"라고 요청했을 때의 코드와 테스트 케이스 생성 결과)

![image](https://github.com/ByeongSunHong/Assertion/assets/49702343/cdb13367-9138-437b-a54b-2f7ef9540428)

![image](https://github.com/ByeongSunHong/Assertion/assets/49702343/81078717-57f6-4d47-8006-f8becbe2ddf3)

4. test2.py (Code Llama) 실행 방법
   
   python test2.py

   -k=[hugging face login token (지웅꺼로 Default, 유출 금지)]

   -m=[model name, default = "codellama/CodeLlama-7b-Instruct-hf"]
   
6. 명령에 따른 코드 생성 예시

   ("define a function for finding maximum value in a list.")
   [Code Generation]
   ![image](https://github.com/ByeongSunHong/Assertion/assets/75852687/fa87d3cb-78f1-4827-931b-a6c6c7fa6141)
   
   [Test Case Generation]
   ![image](https://github.com/ByeongSunHong/Assertion/assets/75852687/c0d67552-f1d0-44aa-879d-afbe76701963)
