import os
from pathlib import Path

# 기본 디렉토리 경로
base_dirs = [
    'data/processed/features',
    'data/processed/labels',
    'data/processed/consensus_labels',
    'models/adaptive',
    'results/dl_optimization',
    'logs'
]

# 디렉토리 생성
for dir_path in base_dirs:
    full_path = Path(dir_path)
    full_path.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {full_path}")

print("\nDirectory setup completed!")
