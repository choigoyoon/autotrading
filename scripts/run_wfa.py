import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가하여 모듈 임포트
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tools.perform_walk_forward_analysis import perform_walk_forward_analysis

if __name__ == "__main__":
    data_path = 'results/enhanced_feature_dataset.parquet'
    output_dir = 'results/walk_forward_validation'
    
    print("🚀 Walk-Forward Analysis 실행 스크립트 시작...")
    try:
        perform_walk_forward_analysis(
            data_path=data_path,
            output_dir=output_dir
        )
        print("🎉 Walk-Forward Analysis 성공적으로 완료!")
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc() 