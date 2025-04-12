"""
Multi LLM Agent 시스템 테스트 파일
"""
import asyncio
import json
from pathlib import Path
from meta_llm_system import run_system

async def test_run():
    """
    시스템을 실행하고 결과를 확인하는 함수
    """
    # input.pdf 파일이 존재하는지 확인
    input_file = Path("input.pdf")
    if not input_file.exists():
        print(f"오류: {input_file} 파일이 존재하지 않습니다.")
        print("루트 폴더에 input.pdf 파일을 넣은 후 다시 실행하세요.")
        return
    
    # 시스템 실행
    print("Multi LLM Agent 시스템 테스트를 시작합니다...")
    result = await run_system()
    
    if result:
        # 결과 출력
        print("\n===== 결과 요약 =====")
        print(f"입력 요약: {result['context']['input_summary'][:100]}...")
        print(f"사용자 요청: {result['context']['user_request']}")
        print(f"목표: {result['context']['goal']}")
        
        print("\nOpenAI 후속 질문:")
        print(result['results']['followup_question_openai']['question'])
        
        print("\nClaude 후속 질문:")
        print(result['results']['followup_question_claude']['question'])
        
        # 평가 결과 요약
        print("\n===== 평가 결과 요약 =====")
        
        print("OpenAI 질문 평가:")
        openai_scores = [int(item['score']) for item in result['results']['followup_question_openai']['evaluation']]
        if openai_scores:
            avg_score = sum(openai_scores) / len(openai_scores)
            print(f"평균 점수: {avg_score:.2f}/5.0")
        else:
            print("평가 결과가 없습니다.")
        
        print("\nClaude 질문 평가:")
        claude_scores = [int(item['score']) for item in result['results']['followup_question_claude']['evaluation']]
        if claude_scores:
            avg_score = sum(claude_scores) / len(claude_scores)
            print(f"평균 점수: {avg_score:.2f}/5.0")
        else:
            print("평가 결과가 없습니다.")
    else:
        print("시스템 실행 중 오류가 발생했습니다.")

if __name__ == "__main__":
    # 비동기 함수 실행
    asyncio.run(test_run())