import logging
import pandas as pd


class LeverageSafetySystem:
    """
    10배 레버리지 환경에서의 포괄적인 리스크 관리를 담당하는 클래스.
    모든 거래 결정 이전에 이 시스템의 검증을 통과해야 합니다.
    """
    def __init__(self, leverage=10, max_account_risk_per_trade=0.01, max_total_exposure=0.15, max_daily_loss_pct=0.03):
        self.leverage = leverage
        self.max_account_risk_per_trade = max_account_risk_per_trade
        self.max_total_exposure = max_total_exposure
        self.max_daily_loss_pct = max_daily_loss_pct
        
        logging.info(f"LeverageSafetySystem 초기화: 레버리지={self.leverage}x, 거래당 리스크={self.max_account_risk_per_trade:.2%}")

    def calculate_safe_position_size(self, entry_price, stop_loss_price, volatility=None, sr_distance=None):
        """
        레버리지를 고려한 안전한 포지션 크기를 계산합니다.
        계좌의 특정 비율(max_account_risk_per_trade)만 리스크에 노출되도록 포지션 크기를 조절합니다.
        
        :param entry_price: 진입 가격
        :param stop_loss_price: 손절 가격
        :param volatility: 변동성 (ATR 등, 옵션)
        :param sr_distance: 지지/저항까지의 거리 (옵션)
        :return: 계좌 자산 대비 포지션 크기 (예: 0.05는 5% 의미)
        """
        if entry_price <= stop_loss_price:
            logging.error("손절 가격은 진입 가격보다 낮아야 합니다.")
            return 0.0

        price_decline_pct = (entry_price - stop_loss_price) / entry_price
        
        if price_decline_pct == 0:
            logging.warning("진입 가격과 손절 가격이 동일하여 포지션 크기를 계산할 수 없습니다.")
            return 0.0
            
        # 기본 포지션 크기 계산: (계좌 리스크) / (가격 하락률 * 레버리지)
        position_size = self.max_account_risk_per_trade / (price_decline_pct * self.leverage)
        
        # TODO: 변동성 및 지지/저항 거리에 따른 동적 조정 로직 추가
        if volatility:
            # 변동성이 높으면 포지션 축소
            pass
        if sr_distance:
            # 지지/저항이 가까우면 포지션 확대 가능
            pass
            
        # 최대 포지션 크기를 10%로 제한 (상향 조정)
        position_size = min(position_size, 0.10)
        
        return position_size

    def calculate_atr_based_stop_loss(self, df, current_idx, entry_price, lookback=14):
        """ATR 기반 동적 손절 계산"""
        if current_idx < lookback + 1:
            return entry_price * 0.95  # Not enough data, use default 5% stop

        # Correctly slice data for ATR calculation
        start_idx = current_idx - lookback
        end_idx = current_idx
        
        highs = df['high'].iloc[start_idx:end_idx]
        lows = df['low'].iloc[start_idx:end_idx]
        prev_closes = df['close'].iloc[start_idx - 1:end_idx - 1]

        high_low = highs.values - lows.values
        high_close = abs(highs.values - prev_closes.values)
        low_close = abs(lows.values - prev_closes.values)

        tr_df = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close})
        true_range = tr_df.max(axis=1)
        atr = true_range.mean()

        # Use 2x ATR for stop distance, reflecting market volatility
        stop_distance_pct = (atr * 2) / entry_price
        
        # Cap stop loss at 8% max
        final_stop_distance_pct = min(stop_distance_pct, 0.08)

        return entry_price * (1 - final_stop_distance_pct)

    def define_stop_loss_levels(self, entry_price, structural_stop=None, max_holding_period_candles=None):
        """
        다층 손절 시스템을 정의합니다.
        
        :param entry_price: 진입 가격
        :param structural_stop: 지지/저항 기반의 구조적 손절 가격
        :param max_holding_period_candles: 최대 보유 기간 (캔들 수)
        :return: 손절 전략 딕셔너리
        """
        # 레버리지를 고려한 최대 손실 허용 가격 계산 (계좌 리스크 1% 기준)
        leverage_adjusted_stop = entry_price * (1 - (self.max_account_risk_per_trade / self.leverage))
        
        final_stop_loss = leverage_adjusted_stop
        if structural_stop:
            # 구조적 손절과 레버리지 기반 손절 중 더 타이트한(가격이 높은) 것을 선택하여 리스크 최소화
            final_stop_loss = max(leverage_adjusted_stop, structural_stop)

        return {
            "structural_stop": structural_stop,
            "leverage_adjusted_stop": leverage_adjusted_stop,
            "final_stop_loss": final_stop_loss,
            "time_based_stop_after_candles": max_holding_period_candles
        }

    def validate_new_position(self, proposed_position_size, current_total_exposure, daily_loss, num_concurrent_positions):
        """
        새로운 포지션 진입 전 포트폴리오 수준의 리스크를 검증합니다.
        
        :param proposed_position_size: 제안된 포지션 크기 (자산 대비 비율)
        :param current_total_exposure: 현재 총 노출 (자산 대비 비율)
        :param daily_loss: 당일 발생한 손실률
        :param num_concurrent_positions: 현재 보유중인 포지션 수
        :return: (승인 여부, 사유) 튜플
        """
        # 1. 일일 손실 한도 검증
        if daily_loss >= self.max_daily_loss_pct:
            return False, f"일일 손실 한도({self.max_daily_loss_pct:.2%}) 초과"
            
        # 2. 총 노출 한도 검증
        if (current_total_exposure + proposed_position_size) > self.max_total_exposure:
            return False, f"총 노출 한도({self.max_total_exposure:.2%}) 초과"

        # 3. 동시 포지션 수 제한 (임시)
        # TODO: 포지션 제한 로직을 더 정교하게 수정
        if num_concurrent_positions >= 5:
             return False, "최대 동시 포지션 수 초과"

        return True, "모든 리스크 검증 통과"

    def emergency_liquidation_trigger(self, current_drawdown):
        """특정 조건에서 시스템을 중단시키는 비상 청산 트리거."""
        # 이 기능은 매우 신중하게 적용해야 함
        if current_drawdown >= self.max_daily_loss_pct * 1.5: # 예: 일일 손실 한도의 1.5배 이상 하락 시
            logging.critical(f"비상 청산 트리거 발동! 현재 MDD: {current_drawdown:.2%}")
            return True
        return False

if __name__ == '__main__':
    # --- 유닛 테스트 및 사용 예시 ---
    logging.basicConfig(level=logging.INFO)
    
    safety_system = LeverageSafetySystem(leverage=10, max_account_risk_per_trade=0.01)

    # 1. 포지션 사이즈 계산 예시
    entry = 50000
    stop = 49500 # 1% 가격 하락
    size = safety_system.calculate_safe_position_size(entry, stop)
    print(f"가격 하락률 1%일 때, 안전 포지션 크기: {size:.2%} (계좌 자산의)")
    # 예상 결과: 0.01 / (0.01 * 10) = 0.1 = 10%

    entry = 50000
    stop = 49000 # 2% 가격 하락
    size = safety_system.calculate_safe_position_size(entry, stop)
    print(f"가격 하락률 2%일 때, 안전 포지션 크기: {size:.2%} (계좌 자산의)")
    # 예상 결과: 0.01 / (0.02 * 10) = 0.05 = 5%

    # 2. 손절 레벨 정의 예시
    stop_levels = safety_system.define_stop_loss_levels(entry_price=50000, structural_stop=49600)
    print(f"\n손절 레벨 정의: {stop_levels}")
    # 레버리지 조정 손절: 50000 * (1 - 0.01/10) = 49950
    # 구조적 손절: 49600
    # 최종 손절은 둘 중 더 높은 가격인 49950이 되어야 함

    # 3. 포지션 진입 검증 예시
    is_approved, reason = safety_system.validate_new_position(
        proposed_position_size=0.06, # 6%
        current_total_exposure=0.1, # 10%
        daily_loss=0.02, # 2%
        num_concurrent_positions=2
    )
    print(f"\n포지션 진입 검증 (실패 예상): {is_approved}, {reason}")
    # 실패 사유: 총 노출 한도 (15%) 초과 (10% + 6% = 16%)
    
    is_approved, reason = safety_system.validate_new_position(
        proposed_position_size=0.05, # 5%
        current_total_exposure=0.08, # 8%
        daily_loss=0.01, # 1%
        num_concurrent_positions=2
    )
    print(f"포지션 진입 검증 (성공 예상): {is_approved}, {reason}")
    # 성공 사유: 모든 한도 내 