"""
다양한 LLM API 클라이언트를 구현하는 모듈입니다.
Langchain을 사용하여 OpenAI와 Anthropic API를 일관된 인터페이스로 호출합니다.
"""
import os
import asyncio
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv

# Langchain 임포트 - LLM 및 프롬프트 템플릿
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# .env 파일에서 환경 변수 로드
load_dotenv()

# API 키 가져오기
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 모델 이름 설정
META_LLM_MODEL = os.getenv("META_LLM_MODEL", "claude-3-7-sonnet-20250219")
INQUISITIVE_LLM_CLAUDE_MODEL = os.getenv("INQUISITIVE_LLM_CLAUDE_MODEL", "claude-3-haiku-20240307")
INQUISITIVE_LLM_OPENAI_MODEL = os.getenv("INQUISITIVE_LLM_OPENAI_MODEL", "gpt-4o-mini")
EVALUATION_LLM_MODEL = os.getenv("EVALUATION_LLM_MODEL", "claude-3-5-sonnet-20240620")

def initialize_llm_models():
    """
    Langchain LLM 모델을 초기화하는 함수
    
    Returns:
        dict: 초기화된 LLM 모델들의 딕셔너리
    """
    models = {}
    
    # 모델 초기화 전 API 키 확인
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    
    # Meta LLM 모델 초기화 (Claude 3.7 Sonnet)
    models["meta"] = ChatAnthropic(
        model=META_LLM_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=2000
    )
    
    # Inquisitive LLM 모델 초기화 (Claude 3 Haiku)
    models["inquisitive_claude"] = ChatAnthropic(
        model=INQUISITIVE_LLM_CLAUDE_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=1000
    )
    
    # Inquisitive LLM 모델 초기화 (GPT-4o mini)
    models["inquisitive_openai"] = ChatOpenAI(
        model=INQUISITIVE_LLM_OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY,
        max_tokens=1000
    )
    
    # Evaluation LLM 모델 초기화 (Claude 3.5 Sonnet)
    models["evaluation"] = ChatAnthropic(
        model=EVALUATION_LLM_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=2000
    )
    
    return models

# 모델 초기화
llm_models = initialize_llm_models()

def create_prompt_chain(model_key: str, system_prompt: str, output_parser=None):
    """
    Langchain 프롬프트 체인을 생성하는 함수
    
    Args:
        model_key (str): 사용할 모델의 키
        system_prompt (str): 시스템 프롬프트
        output_parser: 출력 파서 (기본값: StrOutputParser)
        
    Returns:
        chain: 프롬프트 체인
    """
    if output_parser is None:
        output_parser = StrOutputParser()
    
    # 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{user_input}")
    ])
    
    # 체인 생성
    chain = (
        {"user_input": RunnablePassthrough()} 
        | prompt 
        | llm_models[model_key] 
        | output_parser
    )
    
    return chain

def call_meta_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Meta LLM(Claude 3.7 Sonnet)을 호출하는 함수
    
    Args:
        system_prompt (str): 시스템 프롬프트
        user_prompt (str): 사용자 프롬프트
        
    Returns:
        str: LLM의 응답
    """
    try:
        # 프롬프트 체인 생성
        chain = create_prompt_chain("meta", system_prompt)
        
        # 체인 실행
        response = chain.invoke(user_prompt)
        return response
    except Exception as e:
        print(f"Meta LLM 호출 중 오류 발생: {e}")
        return None

def call_inquisitive_llm_claude(system_prompt: str, user_prompt: str) -> str:
    """
    Inquisitive LLM(Claude 3 Haiku)을 호출하는 함수
    
    Args:
        system_prompt (str): 시스템 프롬프트
        user_prompt (str): 사용자 프롬프트
        
    Returns:
        str: LLM의 응답
    """
    try:
        # 프롬프트 체인 생성
        chain = create_prompt_chain("inquisitive_claude", system_prompt)
        
        # 체인 실행
        response = chain.invoke(user_prompt)
        return response
    except Exception as e:
        print(f"Inquisitive Claude LLM 호출 중 오류 발생: {e}")
        return None

def call_inquisitive_llm_openai(system_prompt: str, user_prompt: str) -> str:
    """
    Inquisitive LLM(GPT-4o mini)을 호출하는 함수
    
    Args:
        system_prompt (str): 시스템 프롬프트
        user_prompt (str): 사용자 프롬프트
        
    Returns:
        str: LLM의 응답
    """
    try:
        # 프롬프트 체인 생성
        chain = create_prompt_chain("inquisitive_openai", system_prompt)
        
        # 체인 실행
        response = chain.invoke(user_prompt)
        return response
    except Exception as e:
        print(f"Inquisitive OpenAI LLM 호출 중 오류 발생: {e}")
        return None

def call_evaluation_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Evaluation LLM(Claude 3.5 Sonnet)을 호출하는 함수
    
    Args:
        system_prompt (str): 시스템 프롬프트
        user_prompt (str): 사용자 프롬프트
        
    Returns:
        str: LLM의 응답
    """
    try:
        # 프롬프트 체인 생성
        chain = create_prompt_chain("evaluation", system_prompt)
        
        # 체인 실행
        response = chain.invoke(user_prompt)
        return response
    except Exception as e:
        print(f"Evaluation LLM 호출 중 오류 발생: {e}")
        return None

async def call_inquisitive_llms_parallel(system_prompt: str, user_prompt: str) -> Tuple[str, str]:
    """
    두 개의 Inquisitive LLM을 병렬로 호출하는 비동기 함수
    
    Args:
        system_prompt (str): 시스템 프롬프트
        user_prompt (str): 사용자 프롬프트
        
    Returns:
        tuple: (OpenAI 응답, Claude 응답)
    """
    # 비동기 작업 생성
    async def call_openai_async():
        # 프롬프트 체인 생성
        chain = create_prompt_chain("inquisitive_openai", system_prompt)
        # 비동기로 체인 실행
        return await chain.ainvoke(user_prompt)
    
    async def call_claude_async():
        # 프롬프트 체인 생성
        chain = create_prompt_chain("inquisitive_claude", system_prompt)
        # 비동기로 체인 실행
        return await chain.ainvoke(user_prompt)
    
    # 병렬로 두 작업 실행
    openai_response, claude_response = await asyncio.gather(
        call_openai_async(),
        call_claude_async()
    )
    
    return openai_response, claude_response

# 모듈 테스트
if __name__ == "__main__":
    # 간단한 테스트 코드
    print("LLM 클라이언트 모듈 테스트")
    
    test_system_prompt = "당신은 도움이 되는 AI 어시스턴트입니다."
    test_user_prompt = "안녕하세요, 오늘 날씨는 어떤가요?"
    
    response = call_meta_llm(test_system_prompt, test_user_prompt)
    print(f"Meta LLM 응답: {response[:100]}...")  # 응답의 처음 100자만 출력