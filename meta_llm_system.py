"""
Multi LLM Agent 시스템의 메인 코드입니다.
전체 시스템의 흐름을 조정하고 각 LLM의 호출을 관리합니다.
"""
import os
import json
import time
import PyPDF2
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# LLM 클라이언트 모듈 임포트
from llm_clients import (
    call_meta_llm, 
    call_evaluation_llm, 
    call_inquisitive_llms_parallel,
    initialize_llm_models,
    llm_models
)

# Langchain 관련 모듈 임포트
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


def create_output_directory() -> Path:
    """
    결과 저장을 위한 _output 디렉토리를 생성하는 함수
    
    Returns:
        Path: 생성된 디렉토리 경로
    """
    output_dir = Path("_output")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def load_system_prompts() -> Dict[str, str]:
    """
    각 LLM의 시스템 프롬프트를 로드하는 함수
    
    Returns:
        dict: 각 LLM 유형별 시스템 프롬프트
    """
    prompts = {}
    
    # Meta LLM 프롬프트 로드
    with open("Prompts/meta_llm.txt", "r", encoding="utf-8") as f:
        prompts["meta"] = f.read()
    
    # Inquisitive LLM 프롬프트 로드
    with open("Prompts/inquisitive_llm.txt", "r", encoding="utf-8") as f:
        prompts["inquisitive"] = f.read()
    
    # Evaluation LLM 프롬프트 로드
    with open("Prompts/evaluation_llm.txt", "r", encoding="utf-8") as f:
        prompts["evaluation"] = f.read()
    
    return prompts


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    PDF 파일에서 텍스트를 추출하는 함수
    
    Args:
        pdf_path (str): PDF 파일 경로
        
    Returns:
        str: 추출된 텍스트
    """
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        print(f"PDF 파일 처리 중 오류 발생: {e}")
    
    return text


def replace_template_variables(template: str, variables: Dict[str, str]) -> str:
    """
    템플릿 문자열의 변수를 실제 값으로 대체하는 함수
    
    Args:
        template (str): 템플릿 문자열
        variables (dict): 변수 이름과 값의 딕셔너리
        
    Returns:
        str: 변수가 대체된 문자열
    """
    result = template
    
    # {변수명} 형식의 변수 대체
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", value)
    
    # ${변수명} 형식의 변수 대체
    for key, value in variables.items():
        result = result.replace(f"${{{key}}}", value)
    
    return result


class MetaLLMOutput(BaseModel):
    """Meta LLM의 출력을 구조화하는 Pydantic 모델"""
    input_summary: str = Field(description="입력 텍스트의 간략한 요약")
    user_request: str = Field(description="사용자의 요청")
    goal: str = Field(description="2-3단어로 된 목표")


class EvaluationCriteria(BaseModel):
    """평가 기준 항목을 구조화하는 Pydantic 모델"""
    criteria: str = Field(description="평가 기준 이름")
    score: str = Field(description="평가 점수 (1-5)")
    rationale: str = Field(description="평가 점수에 대한 근거")


class EvaluationResult(BaseModel):
    """평가 결과를 구조화하는 Pydantic 모델"""
    evaluation: List[EvaluationCriteria] = Field(description="평가 기준별 점수와 근거")


async def run_system() -> Dict[str, Any]:
    """
    전체 시스템 실행을 관리하는 비동기 함수
    
    Returns:
        dict: 최종 결과
    """
    # 시스템 프롬프트 로드
    prompts = load_system_prompts()
    
    # PDF에서 텍스트 추출
    input_text = extract_text_from_pdf("input.pdf")
    if not input_text:
        print("PDF에서 텍스트를 추출할 수 없습니다.")
        return None
    
    print(f"PDF에서 추출된 텍스트 길이: {len(input_text)} 자")
    
    # 1. Meta LLM이 입력 분석 (Langchain 사용)
    print("1. Meta LLM이 입력을 분석하는 중...")
    
    # 출력 파서 생성
    meta_parser = PydanticOutputParser(pydantic_object=MetaLLMOutput)
    
    # 프롬프트 템플릿 생성 (출력 형식 지시 포함)
    meta_prompt_template = PromptTemplate(
        template="""
        {system_prompt}
        
        다음 텍스트를 분석하고, 요약하세요:
        
        {input_text}
        
        {format_instructions}
        """,
        input_variables=["system_prompt", "input_text"],
        partial_variables={"format_instructions": meta_parser.get_format_instructions()}
    )
    
    # 체인 생성 및 실행
    meta_chain = (
        meta_prompt_template 
        | llm_models["meta"] 
        | meta_parser
    )
    
    try:
        meta_result = await meta_chain.ainvoke({
            "system_prompt": prompts["meta"],
            "input_text": input_text
        })
        
        # 구조화된 결과에서 값 추출
        input_summary = meta_result.input_summary
        user_request = meta_result.user_request
        goal = meta_result.goal
        
        print(f"입력 요약: {input_summary}")
        print(f"사용자 요청: {user_request}")
        print(f"목표: {goal}")
    except Exception as e:
        print(f"Meta LLM 분석 중 오류 발생: {e}")
        # 기본값 설정
        input_summary = "입력 텍스트 요약 실패"
        user_request = "사용자 요청 추출 실패"
        goal = "목표 추출 실패"
    
    # 2. Inquisitive LLM들이 질문 생성 (Langchain 사용)
    print("2. Inquisitive LLM들이 질문을 생성하는 중...")
    
    # 프롬프트 준비
    inquisitive_prompt = f"""
    입력 요약: {input_summary}
    사용자 요청: {user_request}
    목표: {goal}
    
    위 정보를 바탕으로 사용자의 요구를 달성하기 위해 구체화할 부분을 파악하고, 효과적인 후속 질문을 생성하세요.
    """
    
    try:
        # 병렬로 두 LLM 호출 - Langchain의 비동기 기능 활용
        followup_question_openai, followup_question_claude = await call_inquisitive_llms_parallel(
            system_prompt=prompts["inquisitive"],
            user_prompt=inquisitive_prompt
        )
        
        print(f"OpenAI 후속 질문: {followup_question_openai}")
        print(f"Claude 후속 질문: {followup_question_claude}")
    except Exception as e:
        print(f"Inquisitive LLM 질문 생성 중 오류 발생: {e}")
        # 기본값 설정
        followup_question_openai = "OpenAI 질문 생성 실패"
        followup_question_claude = "Claude 질문 생성 실패"
    
    # 3. Evaluation LLM이 질문 평가 (Langchain 사용)
    print("3. Evaluation LLM이 질문을 평가하는 중...")
    
    # 평가 변수 준비
    eval_variables = {
        "input_summary": input_summary,
        "user_request": user_request,
        "goal": goal
    }
    
    # JSON 파서 생성 (평가 결과를 구조화된 형식으로 받기 위함)
    eval_parser = JsonOutputParser(pydantic_object=EvaluationResult)
    
    # 평가 프롬프트 템플릿
    eval_prompt_template = PromptTemplate(
        template="{evaluation_prompt}\n\nFollow-up Question to evaluate: {followup_question}",
        input_variables=["evaluation_prompt", "followup_question"]
    )
    
    # 평가 프롬프트 준비 (변수 대체)
    evaluation_prompt = replace_template_variables(prompts["evaluation"], eval_variables)
    
    # OpenAI 질문 평가
    try:
        eval_chain_openai = (
            eval_prompt_template 
            | llm_models["evaluation"] 
            | eval_parser
        )
        
        eval_result_openai = await eval_chain_openai.ainvoke({
            "evaluation_prompt": evaluation_prompt,
            "followup_question": followup_question_openai
        })
        evaluation_openai = eval_result_openai.evaluation
    except Exception as e:
        print(f"OpenAI 질문 평가 중 오류 발생: {e}")
        evaluation_openai = []
    
    # Claude 질문 평가
    try:
        eval_chain_claude = (
            eval_prompt_template 
            | llm_models["evaluation"] 
            | eval_parser
        )
        
        eval_result_claude = await eval_chain_claude.ainvoke({
            "evaluation_prompt": evaluation_prompt,
            "followup_question": followup_question_claude
        })
        evaluation_claude = eval_result_claude.evaluation
    except Exception as e:
        print(f"Claude 질문 평가 중 오류 발생: {e}")
        evaluation_claude = []
    
    # 4. 최종 결과 구성
    final_result = {
        "context": {
            "input_summary": input_summary,
            "user_request": user_request,
            "goal": goal
        },
        "results": {
            "followup_question_openai": {
                "question": followup_question_openai,
                "evaluation": [eval_item.dict() for eval_item in evaluation_openai] if evaluation_openai else []
            },
            "followup_question_claude": {
                "question": followup_question_claude,
                "evaluation": [eval_item.dict() for eval_item in evaluation_claude] if evaluation_claude else []
            }
        }
    }
    
    # 5. 결과 저장
    output_dir = create_output_directory()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = output_dir / f"{timestamp}_output.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    print(f"결과가 {output_file}에 저장되었습니다.")
    
    return final_result


if __name__ == "__main__":
    # 비동기 함수 실행
    try:
        result = asyncio.run(run_system())
        if result:
            print("시스템 실행 완료!")
        else:
            print("시스템 실행 중 오류가 발생했습니다.")
    except Exception as e:
        print(f"프로그램 실행 중 예상치 못한 오류 발생: {e}")