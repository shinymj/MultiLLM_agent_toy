# .env 파일 예시
# OPENAI_API_KEY=your-api-key-here
# META_MODEL=gpt-4-turbo
# AGENT_MODEL=gpt-3.5-turbo

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

# OpenAI 클라이언트 임포트
from openai import AsyncOpenAI

class OpenAIClient:
    def __init__(self, model_name: str):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model_name = model_name
    
    async def complete(self, prompt: str) -> str:
        """OpenAI API를 사용하여 프롬프트에 대한 응답을 생성합니다."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API 호출 중 오류 발생: {str(e)}")
            raise


class MetaLLMSystem:
    def __init__(self, config: Dict[str, Any]):
        self.meta_llm = config["meta_llm"]  # 메타 LLM 인스턴스
        self.agents = {}  # 전문 에이전트들
        self.evaluation_metrics = config.get("evaluation_metrics", [
            "accuracy", "relevance", "coherence", "safety"
        ])
        self.response_threshold = config.get("response_threshold", 0.7)
        self.max_retries = config.get("max_retries", 2)
        self.conversation_history = []

    # 새로운 에이전트 등록
    def register_agent(self, agent_id: str, llm_instance: Any, specialization: str):
        self.agents[agent_id] = {
            "llm": llm_instance,
            "specialization": specialization,
            "performance_metrics": {
                "success_rate": 1.0,
                "average_score": 0,
                "total_calls": 0
            }
        }
        print(f"에이전트 등록됨: {agent_id}, 전문 분야: {specialization}")

    # 사용자 요청 처리
    async def process_request(self, user_query: str) -> str:
        # 1. 메타 LLM이 쿼리 분석 및 작업 할당
        task_allocation = await self.analyze_and_delegate_task(user_query)
        
        # 2. 선택된 에이전트에게 작업 위임
        agent_responses = await self.delegate_to_agents(task_allocation, user_query)
        
        # 3. 메타 LLM이 응답들을 평가
        evaluated_responses = await self.evaluate_responses(agent_responses, user_query)
        
        # 4. 최종 응답 생성 또는 수정 요청
        final_response = await self.generate_final_response(evaluated_responses, user_query)
        
        # 5. 대화 기록 업데이트
        self.update_conversation_history(user_query, final_response)
        
        return final_response

    # 메타 LLM이 쿼리를 분석하고 적절한 에이전트에게 작업 할당
    async def analyze_and_delegate_task(self, user_query: str) -> Dict[str, Any]:
        prompt = f"""
        다음 사용자 쿼리를 분석하고 어떤 전문 에이전트에게 할당해야 하는지 결정하세요.
        
        사용자 쿼리: "{user_query}"
        
        사용 가능한 에이전트:
        {chr(10).join([f"- {id}: {agent['specialization']}" for id, agent in self.agents.items()])}
        
        각 에이전트에게 신뢰도 점수(0-1)와 함께 할당할 작업을 지정하세요. 
        필요한 경우 여러 에이전트에게 작업을 분배하세요.
        
        출력 형식을 정확하게 JSON으로 유지해야 합니다:
        {{
            "allocations": [
                {{"agentId": "에이전트ID", "task": "수행할 작업 설명", "confidence": 0.X}},
                ...
            ]
        }}
        """
        
        meta_response = await self.meta_llm.complete(prompt)
        
        try:
            # JSON 응답을 파싱
            return json.loads(meta_response)
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 간단한 할당으로 대체
            print("메타 LLM의 응답이 유효한 JSON이 아님. 기본 할당으로 대체.")
            
            # 모든 에이전트에게 동일한 작업 할당
            return {
                "allocations": [
                    {"agentId": agent_id, "task": user_query, "confidence": 0.8}
                    for agent_id in self.agents.keys()
                ]
            }

    # 할당된 작업을 에이전트에게 위임 (병렬 처리)
    async def delegate_to_agents(self, task_allocation: Dict[str, Any], original_query: str) -> List[Dict[str, Any]]:
        async def process_allocation(allocation):
            if allocation["confidence"] < self.response_threshold:
                return None
                
            agent_id = allocation["agentId"]
            if agent_id not in self.agents:
                return {
                    "agentId": agent_id,
                    "error": "에이전트를 찾을 수 없음",
                    "response": None
                }
            
            try:
                agent = self.agents[agent_id]
                response = await agent["llm"].complete(
                    f"{original_query}\n\n수행할 작업: {allocation['task']}"
                )
                
                return {
                    "agentId": agent_id,
                    "task": allocation["task"],
                    "confidence": allocation["confidence"],
                    "response": response
                }
            except Exception as e:
                return {
                    "agentId": agent_id,
                    "task": allocation["task"],
                    "confidence": allocation["confidence"],
                    "error": str(e),
                    "response": None
                }
        
        # 모든 할당을 병렬로 처리
        tasks = [process_allocation(allocation) for allocation in task_allocation["allocations"]]
        results = await asyncio.gather(*tasks)
        
        # None 값 필터링
        return [result for result in results if result is not None]

    # 메타 LLM이 에이전트 응답 평가
    async def evaluate_responses(self, agent_responses: List[Dict[str, Any]], original_query: str) -> Dict[str, Any]:
        if not agent_responses:
            # 유효한 응답이 없는 경우 기본 평가 반환
            return {
                "responses": [],
                "evaluation": {
                    "evaluations": [],
                    "recommendedAction": "regenerate"
                }
            }
        
        evaluation_prompt = f"""
        다음 사용자 쿼리와 여러 에이전트의 응답을 평가하세요.
        
        사용자 쿼리: "{original_query}"
        
        에이전트 응답:
        {chr(10).join([
            f"--- 에이전트 {resp['agentId']} (작업: {resp['task']}) ---\n"
            f"{'오류: ' + resp['error'] if 'error' in resp and resp['error'] else resp['response']}"
            for resp in agent_responses
        ])}
        
        각 응답에 대해 다음 기준으로 0-10 점수를 매기고 피드백을 제공하세요:
        {', '.join(self.evaluation_metrics)}
        
        출력 형식을 정확하게 JSON으로 유지해야 합니다:
        {{
            "evaluations": [
                {{
                    "agentId": "에이전트ID",
                    "scores": {{{', '.join([f'"{m}": X' for m in self.evaluation_metrics])}}},
                    "overallScore": X.X,
                    "feedback": "피드백 내용",
                    "usable": true
                }},
                ...
            ],
            "recommendedAction": "use_as_is|modify|regenerate|combine"
        }}
        """
        
        evaluation_response = await self.meta_llm.complete(evaluation_prompt)
        
        try:
            parsed_evaluation = json.loads(evaluation_response)
            
            # 에이전트 성능 지표 업데이트
            for eval_item in parsed_evaluation["evaluations"]:
                agent_id = eval_item["agentId"]
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    agent["performance_metrics"]["total_calls"] += 1
                    
                    prev_avg = agent["performance_metrics"]["average_score"]
                    prev_calls = agent["performance_metrics"]["total_calls"] - 1
                    new_score = eval_item["overallScore"]
                    
                    agent["performance_metrics"]["average_score"] = (prev_avg * prev_calls + new_score) / agent["performance_metrics"]["total_calls"]
                    
                    if not eval_item["usable"]:
                        prev_success = agent["performance_metrics"]["success_rate"]
                        agent["performance_metrics"]["success_rate"] = (prev_success * prev_calls) / agent["performance_metrics"]["total_calls"]
            
            return {
                "responses": agent_responses,
                "evaluation": parsed_evaluation
            }
            
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 간단한 평가로 대체
            print("메타 LLM의 평가 응답이 유효한 JSON이 아님. 기본 평가로 대체.")
            
            # 가장 신뢰도가 높은 응답 찾기
            best_response = max(agent_responses, key=lambda x: x["confidence"])
            
            return {
                "responses": agent_responses,
                "evaluation": {
                    "evaluations": [
                        {
                            "agentId": best_response["agentId"],
                            "scores": {m: 7 for m in self.evaluation_metrics},
                            "overallScore": 7.0,
                            "feedback": "기본 평가",
                            "usable": True
                        }
                    ],
                    "recommendedAction": "use_as_is"
                }
            }

    # 메타 LLM이 최종 응답 생성
    async def generate_final_response(self, evaluated_responses: Dict[str, Any], original_query: str) -> str:
        responses = evaluated_responses["responses"]
        evaluation = evaluated_responses["evaluation"]
        
        if not responses or not evaluation["evaluations"]:
            # 응답이 없는 경우 메타 LLM이 직접 응답 생성
            direct_prompt = f"""
            다음 쿼리에 직접 응답해주세요:
            
            쿼리: "{original_query}"
            """
            return await self.meta_llm.complete(direct_prompt)
        
        # 평가 결과에 따라 다른 처리
        recommended_action = evaluation["recommendedAction"]
        
        if recommended_action == "use_as_is":
            # 가장 높은 점수의 응답 반환
            best_eval = max(evaluation["evaluations"], key=lambda x: x["overallScore"])
            best_response = next((r for r in responses if r["agentId"] == best_eval["agentId"]), None)
            
            if best_response:
                return best_response["response"]
            else:
                return "응답을 찾을 수 없습니다."
        
        elif recommended_action == "modify":
            # 메타 LLM이 응답 수정
            best_eval = max(evaluation["evaluations"], key=lambda x: x["overallScore"])
            best_response = next((r for r in responses if r["agentId"] == best_eval["agentId"]), None)
            
            if not best_response:
                return "수정할 응답을 찾을 수 없습니다."
            
            modify_prompt = f"""
            다음 응답을 개선하세요:
            
            원본 쿼리: "{original_query}"
            
            원본 응답:
            {best_response["response"]}
            
            피드백:
            {best_eval["feedback"]}
            
            개선된 응답을 작성하세요.
            """
            
            return await self.meta_llm.complete(modify_prompt)
        
        elif recommended_action == "regenerate":
            # 메타 LLM이 새로운 응답 생성
            regenerate_prompt = f"""
            다음 쿼리에 대한 새로운 응답을 생성하세요:
            
            쿼리: "{original_query}"
            
            기존 응답들에는 다음과 같은 문제가 있었습니다:
            {chr(10).join([f"- {e['agentId']}: {e['feedback']}" for e in evaluation["evaluations"]])}
            
            이러한 문제를 해결한 새로운 응답을 작성하세요.
            """
            
            return await self.meta_llm.complete(regenerate_prompt)
        
        elif recommended_action == "combine":
            # 여러 응답을 결합
            usable_responses = [
                r for r in responses 
                if any(e["agentId"] == r["agentId"] and e.get("usable", False) for e in evaluation["evaluations"])
            ]
            
            if not usable_responses:
                return "결합할 응답이 없습니다."
            
            combine_prompt = f"""
            다음 응답들을 통합하여 최적의 답변을 생성하세요:
            
            원본 쿼리: "{original_query}"
            
            {chr(10).join([
                f"응답 {idx + 1} (에이전트 {r['agentId']}):\n{r['response']}"
                for idx, r in enumerate(usable_responses)
            ])}
            
            이 응답들의 장점을 조합한 통합 답변을 작성하세요.
            """
            
            return await self.meta_llm.complete(combine_prompt)
        
        else:
            # 기본적으로 가장 좋은 응답 사용
            if evaluation["evaluations"]:
                best_eval = max(evaluation["evaluations"], key=lambda x: x["overallScore"])
                best_response = next((r for r in responses if r["agentId"] == best_eval["agentId"]), None)
                
                if best_response:
                    return best_response["response"]
            
            # 적절한 응답이 없는 경우
            return "적절한 응답을 찾을 수 없습니다."

    # 대화 기록 업데이트
    def update_conversation_history(self, query: str, response: str):
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "query": query,
            "response": response
        })
        
        # 필요한 경우 기록 정리 (예: 길이 제한)
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]

    # 에이전트 성능 통계 조회
    def get_agent_performance_stats(self):
        return [
            {
                "agentId": agent_id,
                "specialization": agent_data["specialization"],
                "metrics": agent_data["performance_metrics"]
            }
            for agent_id, agent_data in self.agents.items()
        ]


# 예제 실행 코드
async def main():
    # 환경 변수에서 모델 이름 가져오기 (또는 기본값 사용)
    meta_model = os.getenv("META_MODEL", "gpt-4")
    agent_model = os.getenv("AGENT_MODEL", "gpt-3.5-turbo")
    
    # OpenAI 클라이언트 생성
    meta_llm = OpenAIClient(meta_model)
    
    # 시스템 초기화
    system = MetaLLMSystem({
        "meta_llm": meta_llm,
        "evaluation_metrics": ["정확성", "관련성", "일관성", "안전성", "유용성"]
    })
    
    # 전문 에이전트 등록 (다양한 모델 또는 동일한 모델 사용 가능)
    system.register_agent("code-agent", OpenAIClient(agent_model), "코드 생성 및 디버깅")
    system.register_agent("research-agent", OpenAIClient(agent_model), "정보 검색 및 요약")
    system.register_agent("creative-agent", OpenAIClient(agent_model), "창의적인 콘텐츠 생성")
    system.register_agent("math-agent", OpenAIClient(agent_model), "수학 문제 해결")
    
    # 사용자 요청 처리
    user_query = input("질문을 입력하세요: ")
    print("\n처리 중...\n")
    
    response = await system.process_request(user_query)
    
    print("\n--- 최종 응답 ---")
    print(response)
    
    # 에이전트 성능 통계 출력
    print("\n--- 에이전트 성능 통계 ---")
    print(json.dumps(system.get_agent_performance_stats(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())