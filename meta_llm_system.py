"""
Multi LLM Agent 시스템의 메인 코드입니다.
전체 시스템의 흐름을 조정하고 각 LLM의 호출을 관리합니다.
"""
import os
import json  # 전역 범위에 json 모듈 임포트
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
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser, StrOutputParser
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


def parse_evaluation_response(response_text: str) -> List[Dict[str, str]]:
    """
    평가 LLM의 응답에서 JSON 부분을 추출하고 파싱하는 함수
    
    Args:
        response_text (str): LLM 응답 텍스트
        
    Returns:
        list: 평가 항목 목록
    """
    try:
        # JSON 시작과 끝을 찾기
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            print(f"JSON을 찾을 수 없습니다: {response_text[:100]}...")
            return []
            
        json_str = response_text[start_idx:end_idx]
        data = json.loads(json_str)
        
        # evaluation 키가 있는지 확인
        if "evaluation" in data:
            return data["evaluation"]
        else:
            print(f"'evaluation' 키를 찾을 수 없습니다: {json_str[:100]}...")
            return []
            
    except Exception as e:
        print(f"평가 응답 파싱 중 오류 발생: {e}")
        print(f"응답 텍스트: {response_text[:200]}...")
        return []


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
    
    # 1.1 Meta LLM이 입력 분석하고 요약 생성 (Langchain 사용)
    print("1.1 Meta LLM이 입력을 분석하고 요약을 생성하는 중...")
    
    # 요약 생성을 위한 프롬프트 템플릿
    summary_prompt_template = PromptTemplate(
        template="""
        {system_prompt}
        
        다음 텍스트를 분석하고, 간결하게 요약하세요:
        
        {input_text}
        """,
        input_variables=["system_prompt", "input_text"]
    )
    
    # 요약 생성 체인
    summary_chain = (
        summary_prompt_template 
        | llm_models["meta"] 
        | StrOutputParser()
    )
    
    try:
        # 요약 생성 실행
        summary_response = await summary_chain.ainvoke({
            "system_prompt": prompts["meta"],
            "input_text": input_text
        })
        
        # 요약 추출 (JSON 형식이 아닌 경우를 처리)
        if "{" in summary_response and "input_summary" in summary_response:
            # JSON 응답에서 요약 추출 시도
            try:
                import re
                # JSON 부분 추출
                json_match = re.search(r'({.*})', summary_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    json_data = json.loads(json_str)
                    input_summary = json_data.get("input_summary", "")
                else:
                    input_summary = summary_response
            except:
                input_summary = summary_response
        else:
            # 일반 텍스트 응답
            input_summary = summary_response
        
        print(f"입력 요약: {input_summary}")
        
        # 1.2 사용자에게 텍스트로 무엇을 하고 싶은지 질문
        print("\n1.2 사용자에게 무엇을 하고 싶은지 질문 중...")
        
        # 사용자에게 질문 표시
        print("\n" + "="*50)
        print("입력 텍스트 요약:")
        print(input_summary)
        print("\n이 내용을 가지고 무엇을 하고 싶으신가요?")
        print("="*50 + "\n")
        
        # 사용자 응답 받기
        user_request = input("사용자 요청: ")
        
        # 1.3 사용자 응답을 바탕으로 목표 추출
        print("\n1.3 사용자 응답을 바탕으로 목표를 추출하는 중...")
        
        # 목표 추출을 위한 프롬프트 템플릿
        goal_prompt_template = PromptTemplate(
            template="""
            다음은 텍스트 요약과 그에 대한 사용자의 요청입니다:
            
            텍스트 요약: {input_summary}
            
            사용자 요청: {user_request}
            
            위 사용자의 요청에서 핵심 목표를 2-3단어로 간결하게 추출하세요. 
            추상적이지 않고 구체적인 단어를 사용하세요.
            
            예시 형식:
            "데이터 분석", "의사결정 지원", "문서 요약", "정보 검색" 등
            
            목표:
            """,
            input_variables=["input_summary", "user_request"]
        )
        
        # 목표 추출 체인
        goal_chain = (
            goal_prompt_template 
            | llm_models["meta"] 
            | StrOutputParser()
        )
        
        # 목표 추출 실행
        goal_response = await goal_chain.ainvoke({
            "input_summary": input_summary,
            "user_request": user_request
        })
        
        # 목표 정리 (불필요한 따옴표나 공백 제거)
        goal = goal_response.strip().strip('"\'')
        # "목표:" 같은 접두어 제거
        if "목표:" in goal:
            goal = goal.replace("목표:", "").strip()
        
        print(f"추출된 목표: {goal}")
        
    except Exception as e:
        print(f"Meta LLM 처리 중 오류 발생: {e}")
        # 기본값 설정
        input_summary = "입력 텍스트 요약 실패"
        user_request = "사용자 요청을 받지 못했습니다"
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
        
        # Claude 질문에서 불필요한 문장 제거
        if "여기 제가 생성한 후속 질문입니다:" in followup_question_claude:
            followup_question_claude = followup_question_claude.replace("여기 제가 생성한 후속 질문입니다:", "").strip()
        
        print(f"OpenAI 후속 질문: {followup_question_openai}")
        print(f"Claude 후속 질문: {followup_question_claude}")
    except Exception as e:
        print(f"Inquisitive LLM 질문 생성 중 오류 발생: {e}")
        # 기본값 설정
        followup_question_openai = "OpenAI 질문 생성 실패"
        followup_question_claude = "Claude 질문 생성 실패"
    
    # 3. Evaluation LLM이 질문 평가 (문자열 기반 접근)
    print("3. Evaluation LLM이 질문을 평가하는 중...")
    
    # 평가 변수 준비
    eval_variables = {
        "input_summary": input_summary,
        "user_request": user_request,
        "goal": goal
    }
    
    # 평가 프롬프트 템플릿
    eval_prompt_template = """
    {evaluation_prompt}
    
    Follow-up Question to evaluate: {followup_question}
    """
    
    # 평가 프롬프트 준비 (변수 대체)
    evaluation_prompt = replace_template_variables(prompts["evaluation"], eval_variables)
    
    # OpenAI 질문 평가 - 문자열 기반 접근으로 변경
    evaluation_openai = []
    try:
        # StrOutputParser를 사용하여 원시 문자열 응답 받기
        eval_chain_openai = (
            PromptTemplate.from_template(eval_prompt_template) 
            | llm_models["evaluation"] 
            | StrOutputParser()
        )
        
        # 체인 실행
        eval_result_openai_str = await eval_chain_openai.ainvoke({
            "evaluation_prompt": evaluation_prompt,
            "followup_question": followup_question_openai
        })
        
        # 응답에서 JSON 추출 및 파싱
        evaluation_openai = parse_evaluation_response(eval_result_openai_str)
        
    except Exception as e:
        print(f"OpenAI 질문 평가 중 오류 발생: {e}")
        evaluation_openai = []
    
    # Claude 질문 평가 - 문자열 기반 접근으로 변경
    evaluation_claude = []
    try:
        # StrOutputParser를 사용하여 원시 문자열 응답 받기
        eval_chain_claude = (
            PromptTemplate.from_template(eval_prompt_template) 
            | llm_models["evaluation"] 
            | StrOutputParser()
        )
        
        # 체인 실행
        eval_result_claude_str = await eval_chain_claude.ainvoke({
            "evaluation_prompt": evaluation_prompt,
            "followup_question": followup_question_claude
        })
        
        # 응답에서 JSON 추출 및 파싱
        evaluation_claude = parse_evaluation_response(eval_result_claude_str)
        
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
                "evaluation": evaluation_openai
            },
            "followup_question_claude": {
                "question": followup_question_claude,
                "evaluation": evaluation_claude
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