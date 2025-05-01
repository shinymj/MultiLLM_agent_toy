#!/bin/bash

# Ollama 모델을 사용한 BigGen Bench 테스트 실행 스크립트

# 색상 설정
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 디렉토리 생성
RESULTS_DIR="results"
VISUALIZATIONS_DIR="visualizations"

mkdir -p "$RESULTS_DIR" "$VISUALIZATIONS_DIR"

# 섹션 헤더 표시 함수
section() {
    echo -e "\n${BLUE}======== $1 ========${NC}\n"
}

# 명령어 실행 상태 확인 함수
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1 완료${NC}"
    else
        echo -e "${RED}❌ $1 실패${NC}"
        exit 1
    fi
}

# 파라미터 기본값 설정
USE_OLLAMA=true
USE_OPENAI=false
USE_ANTHROPIC=false
START_IDX=0
END_IDX=""

# 명령줄 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-ollama)
            USE_OLLAMA=false
            shift
            ;;
        --openai)
            USE_OPENAI=true
            shift
            ;;
        --anthropic)
            USE_ANTHROPIC=true
            shift
            ;;
        --start)
            START_IDX="$2"
            shift 2
            ;;
        --end)
            END_IDX="$2"
            shift 2
            ;;
        --all-api)
            USE_OPENAI=true
            USE_ANTHROPIC=true
            shift
            ;;
        *)
            echo -e "${RED}알 수 없는 옵션: $1${NC}"
            echo "사용법: $0 [--no-ollama] [--openai] [--anthropic] [--all-api] [--start START_IDX] [--end END_IDX]"
            exit 1
            ;;
    esac
done

# 적어도 하나의 모델을 선택했는지 확인
if ! $USE_OLLAMA && ! $USE_OPENAI && ! $USE_ANTHROPIC; then
    echo -e "${YELLOW}어떤 모델도 선택되지 않았습니다. 기본값으로 Ollama 모델을 사용합니다.${NC}"
    USE_OLLAMA=true
fi

# 1단계: Ollama 실행 확인
section "Ollama 서비스 확인"

if $USE_OLLAMA; then
    echo "Ollama 서비스가 실행 중인지 확인 중..."
    curl -s localhost:11434/api/version > /dev/null
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Ollama 서비스가 실행 중입니다.${NC}"
        
        # 필요한 모델이 Ollama에 있는지 확인
        echo "사용 가능한 Ollama 모델 확인 중..."
        OLLAMA_MODELS=$(curl -s localhost:11434/api/tags | grep name | cut -d'"' -f4)
        
        REQUIRED_MODELS=("llama3.1" "deepseek-r1" "gemma3")
        MISSING_MODELS=()
        
        for model in "${REQUIRED_MODELS[@]}"; do
            if ! echo "$OLLAMA_MODELS" | grep -q "$model"; then
                MISSING_MODELS+=("$model")
            fi
        done
        
        if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
            echo -e "${YELLOW}일부 필요한 모델이 Ollama에 설치되지 않았습니다: ${MISSING_MODELS[*]}${NC}"
            echo "다음 명령어로 모델을 설치할 수 있습니다:"
            
            for model in "${MISSING_MODELS[@]}"; do
                echo "  ollama pull $model"
            done
            
            echo -e "${YELLOW}계속 진행하시겠습니까? (Y/n)${NC}"
            read -r response
            if [[ "$response" =~ ^([nN][oO]|[nN])$ ]]; then
                echo "중단합니다."
                exit 0
            fi
        else
            echo -e "${GREEN}모든 필요한 모델이 Ollama에 설치되어 있습니다.${NC}"
        fi
    else
        echo -e "${RED}Ollama 서비스가 실행되고 있지 않습니다.${NC}"
        echo "Ollama를 실행하려면 다음 명령어를 입력하세요:"
        echo "  ollama serve"
        exit 1
    fi
fi

# 2단계: 의존성 설치
section "의존성 설치"

echo "Python 의존성 패키지 설치 중..."
pip install -q langchain langchain-core langchain-community langchain-openai langchain-anthropic python-dotenv numpy matplotlib seaborn pandas
check_status "의존성 설치"

# 3단계: 벤치마크 데이터 확인
section "벤치마크 데이터 확인"

if [ ! -f "biggen_bench_instruction_idx0.json" ]; then
    echo -e "${YELLOW}벤치마크 데이터 파일을 찾을 수 없습니다. 업로드된 파일에 심볼릭 링크를 생성합니다...${NC}"
    # 업로드된 파일 찾기
    UPLOADED_FILE=$(find . -maxdepth 1 -name "biggen_bench_instruction_idx0.json" | head -n 1)
    
    if [ -z "$UPLOADED_FILE" ]; then
        echo -e "${RED}벤치마크 데이터 파일을 찾을 수 없습니다.${NC}"
        exit 1
    fi
    
    ln -sf "$UPLOADED_FILE" "biggen_bench_instruction_idx0.json"
    check_status "벤치마크 데이터 링크 생성"
else
    echo -e "${GREEN}벤치마크 데이터 파일을 찾았습니다.${NC}"
fi

# 4단계: 벤치마크 실행
section "벤치마크 실행"

BENCHMARK_CMD="python biggen_bench_test.py --output-dir $RESULTS_DIR"

# 모델 설정
if $USE_OLLAMA; then
    BENCHMARK_CMD="$BENCHMARK_CMD --use-ollama"
fi

if $USE_OPENAI; then
    BENCHMARK_CMD="$BENCHMARK_CMD --use-openai"
fi

if $USE_ANTHROPIC; then
    BENCHMARK_CMD="$BENCHMARK_CMD --use-anthropic"
fi

# 인덱스 범위 설정
if [ ! -z "$START_IDX" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --start-idx $START_IDX"
fi

if [ ! -z "$END_IDX" ]; then
    BENCHMARK_CMD="$BENCHMARK_CMD --end-idx $END_IDX"
fi

echo "실행 명령어: $BENCHMARK_CMD"
eval $BENCHMARK_CMD
check_status "벤치마크 실행"

# 5단계: 결과 시각화
section "결과 시각화"

echo "시각화 생성 중..."
python visualize_results.py --results-dir "$RESULTS_DIR" --output-dir "$VISUALIZATIONS_DIR"
check_status "시각화 생성"

# 최종 메시지
section "벤치마크 파이프라인 완료"

echo -e "${GREEN}벤치마크 파이프라인이 성공적으로 완료되었습니다!${NC}"
echo -e "결과는 다음 위치에서 확인할 수 있습니다:"
echo -e "- 벤치마크 결과: ${YELLOW}$RESULTS_DIR${NC}"
echo -e "- 시각화: ${YELLOW}$VISUALIZATIONS_DIR${NC}"
echo -e "- 리포트: ${YELLOW}$VISUALIZATIONS_DIR/benchmark_report.html${NC}"

echo -e "\nHTML 리포트를 보려면 다음 파일을 브라우저에서 열어주세요:"
echo -e "${BLUE}$(realpath "$VISUALIZATIONS_DIR/benchmark_report.html")${NC}"
