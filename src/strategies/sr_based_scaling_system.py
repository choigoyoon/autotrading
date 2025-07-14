import logging
import pandas as pd
from typing import Dict, Optional
import numpy as np

# --- 프로젝트 모듈 임포트 ---
# 상위 디렉토리의 경로를 sys.path에 추가하여 모듈 임포트
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.risk_management.leverage_safety_system import LeverageSafetySystem
from src.features.comprehensive_sr_detector import ComprehensiveSRDetector

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SRBasedScalingSystem:
    """
    지지/저항(S/R) 기반의 3단계 분할 매수/매도 전략을 실행하는 클래스.
    LeverageSafetySystem과 ComprehensiveSRDetector를 통합하여 10배 레버리지 환경에서 작동합니다.
    """
    def __init__(self, leverage_safety_system: LeverageSafetySystem, sr_detector: ComprehensiveSRDetector, max_account_risk_per_trade: float = 0.01):
        self.safety_system = leverage_safety_system
        self.sr_detector = sr_detector
        self.max_account_risk = max_account_risk_per_trade
        
        self.entry_allocations = {'stage_1': 0.4, 'stage_2': 0.35, 'stage_3': 0.25}
        self.exit_allocations = {'stage_1': 0.3, 'stage_2': 0.4, 'stage_3': 0.3}

        logging.info("SRBasedScalingSystem 초기화 완료. 안전 및 탐지 시스템과 연동되었습니다.")

    def _create_buy_scaling_plan(self, signal: Dict, current_price: float, ohlcv_df: pd.DataFrame) -> Optional[Dict]:
        """
        주어진 신호를 기반으로 3단계 분할 매수 전략을 계획합니다.
        
        :param signal: 다이버전스 등 외부에서 발생한 매수 신호
        :param current_price: 현재 가격
        :param ohlcv_df: 전체 OHLCV 데이터
        :return: 분할 매수 계획 딕셔셔너리, 또는 생성 불가 시 None
        """
        # 1. 지지/저항 레벨 탐지 및 품질 기준 강화
        sr_levels = self.sr_detector.detect_levels(current_price)
        
        min_sr_strength = 0.75  # Gold 등급 이상만 사용
        valid_supports = [
            level for level in sr_levels.get('support_levels', []) 
            if level.get('strength', 0) >= min_sr_strength and 
               level.get('touches', 0) >= 5 and
               level.get('volume_confirmation', False)
        ]

        if len(valid_supports) < 2:
            logging.warning(f"품질 기준(강도 {min_sr_strength}, 터치 5회, 거래량 확인)을 충족하는 지지 레벨이 2개 미만이라 거래를 포기합니다.")
            return None

        # 2. ATR 계산 (동적 간격 설정을 위해)
        atr = self.calculate_atr(ohlcv_df)
        if atr == 0:
            atr = current_price * 0.01 # ATR 계산 불가 시 기본값

        # 3. 각 단계별 진입 가격 및 크기 계획 (가장 강한 2~3개의 지지선 사용)
        entry_plan = {}
        total_planned_risk = 0

        # Stage 1: 즉시 진입
        stage1_support = valid_supports[0]
        stop_loss_price_s1 = stage1_support['price'] * 0.995 
        base_size_s1 = self.safety_system.calculate_safe_position_size(current_price, stop_loss_price_s1)
        final_size_s1 = base_size_s1 * self.entry_allocations['stage_1']
        entry_plan['stage_1'] = self._create_stage_plan('buy', 1, current_price, final_size_s1, stop_loss_price_s1, stage1_support)
        total_planned_risk += final_size_s1 * (current_price - stop_loss_price_s1) / current_price * self.safety_system.leverage
        
        # Stage 2: 2번째 지지선 근처 및 ATR 기반 추가 매수
        stage2_support = valid_supports[1]
        entry_price_s2 = min(stage2_support['price'] * 1.005, current_price - atr * 1.5) # 지지선 또는 ATR 기반 중 더 낮은 가격
        stop_loss_price_s2 = stage2_support['price'] * 0.995
        base_size_s2 = self.safety_system.calculate_safe_position_size(entry_price_s2, stop_loss_price_s2)
        final_size_s2 = base_size_s2 * self.entry_allocations['stage_2']
        entry_plan['stage_2'] = self._create_stage_plan('buy', 2, entry_price_s2, final_size_s2, stop_loss_price_s2, stage2_support)
        total_planned_risk += final_size_s2 * (entry_price_s2 - stop_loss_price_s2) / entry_price_s2 * self.safety_system.leverage

        # Stage 3: (선택적) 3번째 지지선 근처 및 ATR 기반 추가 매수
        if len(valid_supports) > 2:
            stage3_support = valid_supports[2]
            entry_price_s3 = min(stage3_support['price'] * 1.005, current_price - atr * 3.0) # 지지선 또는 ATR 기반 중 더 낮은 가격
            stop_loss_price_s3 = stage3_support['price'] * 0.995
            base_size_s3 = self.safety_system.calculate_safe_position_size(entry_price_s3, stop_loss_price_s3)
            final_size_s3 = base_size_s3 * self.entry_allocations['stage_3']
            entry_plan['stage_3'] = self._create_stage_plan('buy', 3, entry_price_s3, final_size_s3, stop_loss_price_s3, stage3_support)
            total_planned_risk += final_size_s3 * (entry_price_s3 - stop_loss_price_s3) / entry_price_s3 * self.safety_system.leverage

        # 4. 최종 리스크 평가
        risk_assessment = {
            'total_account_risk': total_planned_risk,
            'max_potential_loss_pct': total_planned_risk / self.safety_system.leverage,
            'safety_validation': total_planned_risk <= self.max_account_risk * 1.5 # 분할매수이므로 리스크 살짝 여유
        }

        return {
            'strategy_id': f"scaling_buy_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
            'signal_basis': signal,
            'entry_stages': entry_plan,
            'risk_assessment': risk_assessment
        }
    
    def plan_entry_strategy(self, signal: Dict, current_price: float, ohlcv_df: pd.DataFrame) -> Optional[Dict]:
        """Wrapper for the new planning method to maintain interface compatibility."""
        return self._create_buy_scaling_plan(signal, current_price, ohlcv_df)

    def calculate_atr(self, df, period=14):
        """데이터프레임의 최근 ATR을 계산합니다."""
        if len(df) < period:
            return 0
        
        high_low = df['high'][-period:] - df['low'][-period:]
        high_close = abs(df['high'][-period:] - df['close'].shift(1)[-period:])
        low_close = abs(df['low'][-period:] - df['close'].shift(1)[-period:])
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.mean()

    def _create_stage_plan(self, side, stage_num, price, size, sl, sr_basis):
        return {
            'price': price,
            'size': size,
            'stop_loss': sl,
            'sr_basis': f"strength={sr_basis['strength']:.2f}, methods={sr_basis['methods']}",
            'status': 'planned'
        }

    def plan_exit_strategy(self, entry_price: float, current_price: float) -> Dict:
        """
        주어진 진입 가격을 기반으로 3단계 분할 청산 전략을 계획합니다.
        """
        sr_levels = self.sr_detector.detect_levels(current_price)
        resistance_levels = [lvl for lvl in sr_levels['resistance_levels'] if lvl['strength'] >= 0.6]

        if not resistance_levels:
            logging.warning("유효한 저항 레벨이 없어 기본 목표가로 청산 전략을 수립합니다.")
            return {
                'stage_1': {'price': entry_price * 1.03, 'percentage': self.exit_allocations['stage_1']},
                'stage_2': {'price': entry_price * 1.06, 'percentage': self.exit_allocations['stage_2']},
                'stage_3': {'price': 'trailing_stop', 'percentage': self.exit_allocations['stage_3']}
            }
            
        exit_plan = {}
        # Stage 1 & 2: 저항 레벨 기반
        if len(resistance_levels) > 0:
            exit_plan['stage_1'] = {'price': resistance_levels[0]['price'], 'percentage': self.exit_allocations['stage_1'], 'sr_basis': resistance_levels[0]}
        if len(resistance_levels) > 1:
            exit_plan['stage_2'] = {'price': resistance_levels[1]['price'], 'percentage': self.exit_allocations['stage_2'], 'sr_basis': resistance_levels[1]}
        
        # Stage 3: 트레일링 스탑
        exit_plan['stage_3'] = {'price': 'trailing_stop', 'percentage': self.exit_allocations['stage_3'], 'trigger': 'macd_reversal_or_time_limit'}

        return exit_plan


if __name__ == '__main__':
    # --- 유닛 테스트 및 통합 시나리오 ---
    print("SRBasedScalingSystem 통합 테스트 시작...")

    # 1. 의존성 시스템 초기화
    safety_system = LeverageSafetySystem(leverage=10)
    
    # 2. 샘플 데이터 및 SR 탐지기 준비
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=500, freq='4H'))
    price_data = 40000 + np.random.randn(500).cumsum() * 20
    price_data[250:260] = price_data[250:260] - 800 # 강력한 지지선 생성
    sample_df = pd.DataFrame({
        'open': price_data - 10, 'high': price_data + 100, 'low': price_data - 100,
        'close': price_data, 'volume': np.random.randint(100, 1000, 500)
    }, index=dates)

    sr_detector = ComprehensiveSRDetector(sample_df, timeframe='4h')

    # 3. 분할 매매 시스템 초기화
    scaling_system = SRBasedScalingSystem(safety_system, sr_detector)
    
    # 4. 시나리오 실행: 매수 신호 발생
    current_price = sample_df['close'].iloc[-1]
    buy_signal = {'type': 'bullish_divergence', 'timestamp': pd.Timestamp.now()}
    
    print(f"\n시나리오: 매수 신호 발생 (현재가: {current_price:.2f})")
    entry_strategy = scaling_system.plan_entry_strategy(buy_signal, current_price, sample_df)

    if entry_strategy:
        print("\n--- 생성된 분할 매수 전략 ---")
        print(f"전략 ID: {entry_strategy.get('strategy_id')}")
        print("리스크 평가:", entry_strategy.get('risk_assessment'))
        for stage, plan in entry_strategy.get('entry_stages', {}).items():
            print(f"  - {stage}: 진입가={plan['price']:.2f}, 크기={plan['size']:.3%}, 손절가={plan['stop_loss']:.2f}, 근거={plan['sr_basis']}")
    else:
        print("전략을 생성하지 못했습니다.")
        
    # 5. 시나리오 실행: 분할 청산 계획
    avg_entry_price = 39800 # 진입 성공했다고 가정
    print(f"\n시나리오: 분할 청산 계획 (평균 진입가: {avg_entry_price:.2f})")
    exit_strategy = scaling_system.plan_exit_strategy(avg_entry_price, current_price)
    
    if exit_strategy:
        print("\n--- 생성된 분할 청산 전략 ---")
        for stage, plan in exit_strategy.items():
            print(f"  - {stage}: 목표가={plan.get('price')}, 청산비율={plan.get('percentage')}")

    print("\n통합 테스트 완료.") 