import pandas as pd
from pathlib import Path
import warnings

# 리팩토링된 핵심 시스템을 src.strategies에서 import
try:
    from src.strategies.take_profit_system import IntegratedTakeProfitSystem
except ImportError as e:
    print(f"핵심 모듈 import 실패: {e}")
    print("리팩토링된 IntegratedTakeProfitSystem이 src/strategies/ 경로에 있는지 확인하세요.")
    IntegratedTakeProfitSystem = None

warnings.filterwarnings('ignore', category=FutureWarning)

if __name__ == '__main__':
    if IntegratedTakeProfitSystem is None:
        print("프로그램을 실행할 수 없습니다. 모듈 임포트 문제를 해결해주세요.")
    else:
        try:
            project_root = Path(__file__).resolve().parents[1]
            tp_system = IntegratedTakeProfitSystem(base_path=project_root)
            
            tp_system.prepare_analyzers()
            
            # 여기서부터는 graded_df에 안전하게 접근해야 합니다.
            # get_graded_labels()는 BounceStatisticsAnalyzer의 메서드여야 합니다.
            # 이 클래스는 tp_system.bounce_analyzer 내에 있습니다.
            graded_df = tp_system.bounce_analyzer.get_graded_labels()
            
            if graded_df.empty:
                 raise ValueError("등급이 매겨진 라벨 데이터가 비어있습니다.")

            # top_100_indices는 이제 BounceStatisticsAnalyzer에서 항상 초기화됩니다.
            top_100_indices = tp_system.bounce_analyzer.top_100_indices # type: ignore

            sample_label_series = graded_df[
                (graded_df['grade'] == 'A_Grade') & (~graded_df.index.isin(top_100_indices))
            ]
            
            if sample_label_series.empty:
                print("\n분석할 A등급 샘플 라벨을 찾을 수 없습니다. 존재하는 라벨 중 하나로 테스트합니다.")
                if not graded_df.empty:
                    sample_label_series = graded_df
                else:
                    raise ValueError("테스트할 샘플 라벨이 존재하지 않습니다.")

            sample_timestamp = sample_label_series.iloc[0]['timestamp']
            
            strategy = tp_system.get_full_strategy(label_timestamp=sample_timestamp)
            
            print("\n--- 최종 전략 결과 ---")
            for key, value in strategy.items():
                print(f"{key:<25}: {value:.4f}" if isinstance(value, float) else f"{key:<25}: {value}")

        except (FileNotFoundError, RuntimeError, ValueError) as e:
            print(f"\n[오류] 실행 중 문제가 발생했습니다: {e}")
        except Exception as e:
            import traceback
            print(f"\n[예상치 못한 오류] {e}")
            traceback.print_exc()
