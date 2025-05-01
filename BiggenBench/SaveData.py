from datasets import load_dataset

# 로그인 후 데이터셋 로드
ds = load_dataset("prometheus-eval/BiGGen-Bench")

# 로컬에 저장
ds.save_to_disk("./biggen_bench_dataset")