import pandas as pd
from pathlib import Path
import warnings

# 이전에 생성한 분석 모듈들을 import 합니다.
try:
    from src.analysis.bounce_statistics_analyzer import BounceStatisticsAnalyzer
    from src.analysis.multi_timeframe_validator import MultiTimeframeValidator
except ImportError as e:
    print(f"필수 모듈 import 실패: {e}")
    print("bounce_statistics_analyzer.py와 multi_timeframe_validator.py가 src/analysis/ 경로에 있는지 확인하세요.")
    # 임시 클래스로 대체하여 기본 구조는 유지
    class BounceStatisticsAnalyzer: 
        def __init__(self, *args, **kwargs): pass
        def calculate_grade_statistics(self): pass
        def get_graded_labels(self): return pd.DataFrame()
    class MultiTimeframeValidator: 
        def __init__(self, *args, **kwargs): pass
        def analyze_all_timeframes(self, *args, **kwargs): pass
        def score_environment(self): return 0

warnings.filterwarnings('ignore', category=FutureWarning)

class IntegratedTakeProfitSystem:
    """
    라벨 등급과 시장 환경을 종합하여 동적 익절/손절 전략을 수립합니다.
    """
    def __init__(self, base_path: Path):
        """
        시스템 초기화 및 분석 모듈 준비
        """
        self.base_path = base_path
        self.bounce_analyzer = BounceStatisticsAnalyzer(base_path)
        self.mtf_validator = MultiTimeframeValidator(base_path)
        self.is_ready = False
        print("IntegratedTakeProfitSystem이 초기화되었습니다.")

    def prepare_analyzers(self):
        """분석에 필요한 모든 데이터를 준비하고 사전 계산을 수행합니다."""
        print("사전 분석 모듈 준비 중...")
        self.bounce_analyzer.calculate_grade_statistics()
        self.mtf_validator.analyze_all_timeframes()
        self.is_ready = True
        print("모든 분석 모듈 준비 완료.")

    def _get_label_info(self, label_timestamp: pd.Timestamp) -> pd.Series:
        """주어진 타임스탬프에 해당하는 라벨의 상세 정보를 반환합니다."""
        if not self.is_ready:
            raise RuntimeError("분석 모듈이 준비되지 않았습니다. prepare_analyzers()를 먼저 호출하세요.")

        graded_df = self.bounce_analyzer.get_graded_labels() # Corrected method call
        label_info_row = graded_df[graded_df['timestamp'] == label_timestamp]

        if label_info_row.empty:
            raise ValueError(f"해당 타임스탬프({label_timestamp})의 라벨 정보를 찾을 수 없습니다.")

        return label_info_row.iloc[0]

    def calculate_base_take_profit(self, label_grade: str) -> float:
        """라벨 등급에 따라 기본 익절 목표(%)를 반환합니다."""
        grade_tp_map = {
            'A_Grade': 0.25, 'B_Grade': 0.15, 'Top100_Profit': 0.45, 'C_Grade': 0.05
        }
        base_tp = grade_tp_map.get(label_grade, 0.05)
        print(f"등급 '{label_grade}'에 대한 기본 익절 목표: {base_tp:.2%}")
        return base_tp

    def adjust_by_environment(self, base_tp: float, label_timestamp: pd.Timestamp) -> float:
        """시장 환경 점수에 따라 익절 목표를 조정합니다."""
        self.mtf_validator.analyze_all_timeframes(at_timestamp=label_timestamp)
        score = self.mtf_validator.score_environment()

        if score >= 75: multiplier, env_desc = 1.5, "매우 유리"
        elif 25 <= score < 75: multiplier, env_desc = 1.2, "유리"
        elif -25 < score < 25: multiplier, env_desc = 1.0, "보통"
        elif -75 < score <= -25: multiplier, env_desc = 0.7, "불리"
        else: multiplier, env_desc = 0.5, "매우 불리"

        adjusted_tp = base_tp * multiplier
        print(f"환경 점수: {score:.2f} ({env_desc}) -> 조정 배수: {multiplier}x")
        return adjusted_tp

    def set_risk_management(self, label_grade: str, label_type: int) -> dict:
        """손절선과 최대 보유 기간을 설정합니다. 통계 부재 시 안전한 폴백 로직을 포함합니다."""
        stats_key = 'Buy_Label (L)' if label_type == 1 else 'Sell_Label (H)'
        avg_duration = None
        
        if hasattr(self.bounce_analyzer, 'stats') and self.bounce_analyzer.stats:
            try:
                # 1. 특정 등급 통계 시도
                avg_duration = self.bounce_analyzer.stats[label_grade][stats_key]['avg_duration_tomax_min']
            except KeyError:
                try:
                    # 2. 전체(Overall) 통계로 폴백
                    avg_duration = self.bounce_analyzer.stats['Overall'][stats_key]['avg_duration_tomax_min']
                except KeyError:
                    pass
        
        if avg_duration is None or pd.isna(avg_duration):
            avg_duration = 60.0
            print(f"[경고] 유효한 통계 부재. 기본 보유 기간({avg_duration}분)을 사용합니다.")

        return {
            "stop_loss_pct": -0.05, 
            "max_holding_period_min": avg_duration * 1.5
        }

    def get_full_strategy(self, label_timestamp: pd.Timestamp) -> dict:
        """특정 라벨에 대한 최종 익절/손절 전략을 반환합니다."""
        print(f"\n===== {label_timestamp} 라벨 전략 분석 시작 =====")
        
        label_info = self._get_label_info(label_timestamp)
        label_type = int(label_info['label_type'])

        label_grade = 'N/A'
        if hasattr(self.bounce_analyzer, 'top_100_indices'):
            label_index = label_info.name
            if label_index in self.bounce_analyzer.top_100_indices:
                label_grade = 'Top100_Profit'
        
        if label_grade == 'N/A':
             label_grade = label_info['grade']

        base_tp = self.calculate_base_take_profit(label_grade)
        adjusted_tp = self.adjust_by_environment(base_tp, label_timestamp)
        risk_params = self.set_risk_management(label_grade, label_type)
        
        final_strategy = {"label_timestamp": label_timestamp, "label_grade": label_grade, "adjusted_take_profit_pct": adjusted_tp, **risk_params}
        print(f"최종 익절 목표: {adjusted_tp:.2%}, 최종 손절 목표: {risk_params['stop_loss_pct']:.2%}")
        return final_strategy 