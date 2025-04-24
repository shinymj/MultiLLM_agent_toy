# https://python.langchain.com/docs/integrations/llms/ollama/
# install package
# %pip install -U langchain-ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import asyncio
import json
from datetime import datetime

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="gemma3:12b")
# model = OllamaLLM(model="deepseek-r1:8b")

chain = prompt | model

async def main():
    result = await chain.ainvoke({"question": "Why is question important between human and AI?"})
    print(result)
    
    # 결과 저장
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": "Why is question important between human and AI?",
        "answer": result
    }
    
    # 파일명에 타임스탬프 포함
    filename = f"_output/ollama_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과가 {filename} 파일에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main())