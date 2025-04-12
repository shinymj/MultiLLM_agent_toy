# multiLLM_agent_toy
 Collaborations of multiLLMs under metaLLM  

### 202504112
질문 생성과 질문 평가

project/  
  ├── Prompts/  
  │   ├── meta_llm.txt           # 메타 LLM의 시스템 프롬프트  
  │   ├── inquisitive_llm.txt    # 질문 생성 LLM의 시스템 프롬프트   
  │   └── evaluation_llm.txt     # 평가 LLM의 시스템 프롬프트  
  ├── .env                       # API 키 및 모델 설정  
  ├── input.pdf                  # 테스트용 컨텍스트  
  ├── llm_clients.py             # 다양한 LLM API 클라이언트  
  ├── meta_llm_system.py         # 메인 시스템 코드  
  ├── test_system.py             # 테스트 스크립트  
  └── requirements.txt           # 필요 패키지 목록  

### 20250406 init  
`meta_llm_system.py`: initial version of system code