# multi_agent_system.py by Gemini

# 1. 필요한 라이브러리 임포트
import os
from typing import TypedDict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# 2. 환경 변수 로드 (.env 파일 필요)
load_dotenv()

# 3. LLM 로딩 함수 정의
def get_llm(provider="openai", model_name="gpt-4o"):
    """지정된 제공자와 모델 이름으로 LLM 객체를 반환합니다."""
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY 가 환경 변수에 설정되지 않았습니다.")

    if provider == "openai":
        return ChatOpenAI(api_key=api_key, model=model_name, temperature=0.5)
    elif provider == "anthropic":
        return ChatAnthropic(api_key=api_key, model_name=model_name, temperature=0.5)
    # elif provider == "google":
    #     return ChatGoogleGenerativeAI(google_api_key=api_key, model=model_name, temperature=0)
    else:
        raise ValueError(f"지원하지 않는 LLM 제공자입니다: {provider}")


# 4. LLM 인스턴스 생성
execution_llm = get_llm(provider="openai", model_name="gpt-4o-2024-08-06")
meta_llm = get_llm(provider="anthropic", model_name="claude-3-7-sonnet-20250219")  # 메타 에이전트용 LLM도 별도 설정 가능

# 5. 상태 클래스 정의
class AgentState(TypedDict):
    initial_task: str       # 메타 에이전트가 부여한 초기 작업
    task_goal: Optional[str] # 수행 에이전트가 파악한 작업 목표
    missing_info_questions: Optional[List[str]] # 수행 에이전트가 생성한 질문 목록
    meta_agent_answers: Optional[List[str]] # 메타 에이전트의 답변 목록
    final_response: Optional[str] # 수행 에이전트의 최종 결과물
    # 필요에 따라 중간 결과, 사용된 도구 등의 상태 추가 가능
    intermediate_steps: List = [] # LangChain 에이전트 실행 중간 단계 저장용 (선택 사항)

# --- 수행 에이전트 관련 로직 ---

# 6. (선택적) 구조화된 출력 클래스 정의
class TaskAnalysis(BaseModel):
    """수행 에이전트가 작업을 분석한 결과"""
    perceived_goal: str = Field(description="파악된 작업의 핵심 목표")
    is_info_sufficient: bool = Field(description="작업 수행에 정보가 충분한지 여부 (True/False)")
    questions_for_meta_agent: Optional[List[str]] = Field(description="정보가 부족할 경우 메타 에이전트에게 할 질문 목록")

# 7. 에이전트 노드 함수들 정의
def execution_agent_node(state: AgentState):
    """수행 에이전트: 작업을 분석하고, 정보가 부족하면 질문을 생성합니다."""
    print("--- 수행 에이전트: 작업 분석 시작 ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 주어진 작업을 분석하는 AI 에이전트입니다. 작업의 목표를 명확히 파악하고, 목표 달성에 필요한 정보가 모두 주어졌는지 판단하세요. 정보가 부족하다면, 명확한 답변을 얻을 수 있도록 구체적인 질문을 생성해야 합니다."),
        ("human", "다음 작업을 분석해 주세요:\n\n{task}\n\n작업 목표는 무엇이며, 수행에 필요한 정보가 충분한가요? 부족하다면 어떤 질문을 해야 할까요?")
    ])
    analyzer_chain = prompt | execution_llm.with_structured_output(TaskAnalysis)

    analysis_result = analyzer_chain.invoke({"task": state['initial_task']})

    print(f"분석 결과: 목표='{analysis_result.perceived_goal}', 정보 충분={analysis_result.is_info_sufficient}")

    state['task_goal'] = analysis_result.perceived_goal
    if not analysis_result.is_info_sufficient and analysis_result.questions_for_meta_agent:
        print(f"질문 생성: {analysis_result.questions_for_meta_agent}")
        state['missing_info_questions'] = analysis_result.questions_for_meta_agent
        state['meta_agent_answers'] = None # 질문이 생겼으므로 이전 답변 초기화
    else:
        state['missing_info_questions'] = None # 정보가 충분하므로 질문 없음

    state['intermediate_steps'].append(("execution_analysis", analysis_result))
    return state

# --- 메타 에이전트 관련 로직 ---

def meta_agent_node(state: AgentState):
    """메타 에이전트: 수행 에이전트의 질문에 답변합니다."""
    if not state.get('missing_info_questions'):
        # 질문이 없으면 아무것도 하지 않음
        return state

    print("--- 메타 에이전트: 질문 답변 시작 ---")
    answers = []
    question_context = "\n".join(state['missing_info_questions'])

    # 메타 에이전트는 초기 작업 정의와 전체 목표를 알고 있어야 함
    # 실제 구현에서는 이 정보를 state나 별도의 컨텍스트에서 가져와야 함
    # 여기서는 간단히 프롬프트에 포함
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"당신은 작업의 전체 목표와 방향을 아는 메타 에이전트입니다. 수행 에이전트가 원활히 작업을 진행할 수 있도록 다음 질문에 명확하고 간결하게 답변해주세요.\n\n원래 작업 지시: {state['initial_task']}\n작업의 숨겨진 목표나 제약사항: 여름에 갈 예정이다. 운전면허가 없다. 나는 동양인 여자다."),
        ("human", "다음 질문들에 답변해주세요:\n\n{questions}")
    ])
    responder_chain = prompt | meta_llm

    # 각 질문에 대해 답변 생성 (개선: 여러 질문을 한번에 처리하도록 LLM 구성 가능)
    for question in state['missing_info_questions']:
         # 실제로는 위 responder_chain.invoke({"questions": question_context}) 를 사용
         # 여기서는 간단히 예시 답변 생성
        print(f"질문 처리중: {question}")
        # 예시 답변 로직 (실제로는 LLM 호출)
        if "어떤 형식" in question:
             answer = "결과는 마크다운 리스트 형식으로 정리해주세요."
        elif "데이터 소스" in question:
             answer = "필요한 데이터는 내부 'knowledge_base.txt' 파일을 참조하세요."
        else:
             answer = "그 부분은 자율적으로 판단하여 진행해도 좋습니다."
        answers.append(answer)


    print(f"답변 생성: {answers}")
    state['meta_agent_answers'] = answers
    state['missing_info_questions'] = None # 질문에 답변했으므로 질문 목록 비움
    state['intermediate_steps'].append(("meta_response", answers))
    return state

# --- 최종 작업 실행 노드 (예시) ---
# 실제로는 LangChain의 AgentExecutor나 도구를 사용하는 로직이 들어감
 
def task_execution_node(state: AgentState):
    """수행 에이전트: 모든 정보를 바탕으로 최종 작업을 실행하고 결과를 생성합니다."""
    print("--- 수행 에이전트: 최종 작업 실행 시작 ---")

    # 작업 수행에 필요한 모든 정보 조합
    task_info = f"초기 작업: {state['initial_task']}\n파악된 목표: {state['task_goal']}"
    if state.get('meta_agent_answers'):
        answers_str = "\n".join([f"- {ans}" for ans in state['meta_agent_answers']])
        task_info += f"\n메타 에이전트 답변:\n{answers_str}"

    # 여기에 실제 작업 수행 로직 구현
    # 예: 특정 도구(Python REPL, 검색 등)를 사용하거나, LLM을 통해 최종 응답 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 주어진 정보와 지침에 따라 작업을 수행하고 최종 결과물을 생성하는 AI 에이전트입니다."),
        ("human", "다음 정보들을 바탕으로 최종 결과물을 생성해주세요:\n\n{task_info}\n\n결과:")
    ])
    final_chain = prompt | execution_llm

    final_result = final_chain.invoke({"task_info": task_info})

    print(f"최종 결과 생성: {final_result.content}")
    state['final_response'] = final_result.content
    state['intermediate_steps'].append(("final_execution", final_result.content))
    return state

# 8. 조건부 엣지 함수 정의
def should_ask_meta_agent(state: AgentState) -> str:
    """정보가 부족하면 메타 에이전트에게 묻고, 아니면 최종 실행으로 넘어갑니다."""
    if state.get('missing_info_questions'):
        print("조건 분기: 메타 에이전트에게 질문 필요")
        return "ask_meta_agent"
    else:
        print("조건 분기: 정보 충분, 최종 작업 실행")
        return "execute_final_task"

# 9. LangGraph 그래프 정의 및 컴파일
# 그래프 생성
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("analyze_task", execution_agent_node)
workflow.add_node("ask_meta_agent", meta_agent_node)
workflow.add_node("execute_final_task", task_execution_node)

# 시작점 설정
workflow.set_entry_point("analyze_task")

# ... (노드 추가, 엣지 연결) ...
# 조건부 엣지: analyze_task 노드 이후 상태에 따라 분기
workflow.add_conditional_edges(
    "analyze_task",
    should_ask_meta_agent,
    {
        "ask_meta_agent": "ask_meta_agent",
        "execute_final_task": "execute_final_task",
    }
)

# 메타 에이전트 답변 후에는 다시 작업 분석 또는 바로 최종 실행 가능
# 여기서는 간단히 답변 후 바로 최종 실행으로 연결
workflow.add_edge("ask_meta_agent", "execute_final_task")

# 최종 실행 후 종료
workflow.add_edge("execute_final_task", END)

# 그래프 컴파일
app = workflow.compile()

# 10. 그래프 실행 부분
if __name__ == "__main__": # 스크립트로 직접 실행될 때만 아래 코드 실행
    initial_task_input = "샌프란시스코 2박3일 여행 계획 세워줘."
    inputs = {"initial_task": initial_task_input,
              "intermediate_steps": [] 
    }

    print("--- 시스템 실행 시작 ---")
    # 스트리밍으로 중간 과정 보기
    # for event in app.stream(inputs, {"recursion_limit": 5}):
    #     for key, value in event.items():
    #         print(f"--- 이벤트: {key} ---")
    #         # 상세 내용 출력 (너무 길면 일부만 출력하도록 조정 가능)
    #         print(value)
    #     print("\n")

    # 최종 결과만 보기 (Invoke 사용)
    final_state = app.invoke(inputs, {"recursion_limit": 5})  
    print("--- 최종 상태 ---")
    print(final_state)
    print("\n--- 최종 결과물 ---")
    print(final_state.get('final_response'))

    print("--- 시스템 실행 종료 ---")