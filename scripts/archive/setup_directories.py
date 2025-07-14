from pathlib import Path

def setup_project_directories() -> None:
    """프로젝트에 필요한 기본 디렉토리를 생성합니다."""
    
    print("🚀 프로젝트 디렉토리 설정을 시작합니다...")
    # 기본 디렉토리 경로
    base_dirs: list[str] = [
        'data/processed/features',
        'data/processed/labels',
        'data/processed/consensus_labels',
        'models/adaptive',
        'results/dl_optimization',
        'logs'
    ]

    # 디렉토리 생성
    for dir_path in base_dirs:
        try:
            full_path = Path(dir_path)
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ 생성/확인 완료: {full_path}")
        except OSError as e:
            print(f"❌ '{dir_path}' 생성 실패: {e}")

    print("\n👍 디렉토리 설정이 완료되었습니다.")

if __name__ == "__main__":
    setup_project_directories()
