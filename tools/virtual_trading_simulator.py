import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import warnings
import os
import argparse
from datetime import timedelta

# --- 프로젝트 설정 ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from tools.optimize_hybrid_system import get_base_signals, OptimizationConfig
from tools.portfolio_optimizer import get_rule_based_exposure

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 가상 거래 시스템 설정 ---
class TraderConfig(OptimizationConfig):
    FEE = 0.001  # 거래 수수료 (0.1%)
    SLIPPAGE = 0.0005  # 슬리피지 (0.05%)

    STOP_LOSS_PCT = -0.02  # -2% 손절
    TAKE_PROFIT_PCT = 0.05  # +5% 익절
    PARTIAL_PROFIT_FRAC = 0.5  # 부분 익절 시 청산 비율 (50%)
    MAX_HOLDING_HOURS = 24  # 최대 포지션 보유 시간

class VirtualTrader:
    """가상 거래를 수행하고 계좌 상태를 관리하는 클래스."""
    def __init__(self, initial_balance: float, config: TraderConfig):
        self.config = config
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None  # 현재는 단일 포지션만 지원
        self.trade_log = []
        print(f"✅ 가상 트레이더 초기화 완료. 초기 잔고: ${initial_balance:,.2f}")

    def execute_trade(self, signal: float, price: float, timestamp: pd.Timestamp):
        """신호에 따라 거래를 실행합니다."""
        # 기존 포지션과 신호가 같거나, 신호가 0이면 거래 없음
        if (self.position and np.sign(self.position['direction']) == np.sign(signal)) or signal == 0:
            return

        # 기존 포지션 청산
        if self.position:
            self.close_position(price, timestamp, reason="New Signal")

        # 새 포지션 진입
        direction = np.sign(signal)
        exposure = abs(signal)
        size = self.balance * exposure
        
        entry_price = price * (1 + direction * self.config.SLIPPAGE)
        fee = size * self.config.FEE
        self.balance -= fee
        
        self.position = {
            'entry_time': timestamp,
            'entry_price': entry_price,
            'size': size,
            'direction': direction, # 1: Long, -1: Short
            'initial_stop_loss': entry_price * (1 + self.config.STOP_LOSS_PCT * direction),
            'take_profit': entry_price * (1 + self.config.TAKE_PROFIT_PCT * direction)
        }
        self._log_event("OPEN", self.position, price, timestamp, fee=fee)

    def close_position(self, price: float, timestamp: pd.Timestamp, reason: str, partial_frac: float = 1.0):
        """포지션을 청산하고 손익을 기록합니다."""
        if not self.position: return

        close_price = price * (1 - self.position['direction'] * self.config.SLIPPAGE)
        
        size_to_close = self.position['size'] * partial_frac
        pnl = (close_price / self.position['entry_price'] - 1) * self.position['direction'] * size_to_close
        
        fee = size_to_close * self.config.FEE
        net_pnl = pnl - fee
        
        self.balance += net_pnl
        
        self._log_event("CLOSE", self.position, close_price, timestamp, pnl=net_pnl, reason=reason)
        
        if partial_frac == 1.0:
            self.position = None
        else: # 부분 청산
            self.position['size'] *= (1 - partial_frac)

    def manage_positions(self, current_price: float, timestamp: pd.Timestamp):
        """매 틱마다 포지션을 관리 (손절, 익절, 최대 보유 시간)"""
        if not self.position: return
        
        # 최대 보유 시간 체크
        holding_duration = timestamp - self.position['entry_time']
        if holding_duration >= timedelta(hours=self.config.MAX_HOLDING_HOURS):
            self.close_position(current_price, timestamp, reason="Max Hold Time")
            return

        # 손절/익절 체크
        pnl_ratio = (current_price / self.position['entry_price'] - 1) * self.position['direction']
        
        if pnl_ratio <= self.config.STOP_LOSS_PCT:
            self.close_position(current_price, timestamp, reason="Stop Loss")
        elif pnl_ratio >= self.config.TAKE_PROFIT_PCT:
            self.close_position(current_price, timestamp, reason="Take Profit (Partial)", partial_frac=self.config.PARTIAL_PROFIT_FRAC)
            # 부분 익절 후에는 익절 기준을 다시 설정하지 않음 (추가 익절 방지)
            self.position['take_profit'] = None


    def _log_event(self, event_type, pos, price, ts, **kwargs):
        log_entry = {
            'timestamp': ts,
            'event': event_type,
            'price': price,
            'direction': pos['direction'],
            'size': pos['size'],
            'balance': self.balance,
        }
        log_entry.update(kwargs)
        self.trade_log.append(log_entry)

    def generate_report(self):
        """현재 계좌 상태 리포트를 생성합니다."""
        total_pnl = self.balance - self.initial_balance
        total_return_pct = (total_pnl / self.initial_balance)
        
        closed_trades = [t for t in self.trade_log if t['event'] == 'CLOSE']
        wins = [t for t in closed_trades if t['pnl'] > 0]
        
        win_rate = len(wins) / len(closed_trades) if closed_trades else 0
        
        return {
            "Balance": self.balance,
            "Total PnL": total_pnl,
            "Total Return (%)": total_return_pct * 100,
            "Trades": len(closed_trades),
            "Win Rate (%)": win_rate * 100
        }

def run_simulation(trader: VirtualTrader, signals_df: pd.DataFrame):
    """전체 데이터에 대해 가상매매 시뮬레이션을 실행합니다."""
    print("\n--- 🚀 가상매매 시뮬레이션 시작 ---")
    
    for timestamp, row in tqdm(signals_df.iterrows(), total=len(signals_df), desc="가상매매 진행 중"):
        current_price = row['price']
        
        # 1. 포지션 관리 (손절/익절 등)
        trader.manage_positions(current_price, timestamp)
        
        # 2. 거래 실행
        signal = row['exposure']
        trader.execute_trade(signal, current_price, timestamp)
        
    # 시뮬레이션 종료 후 남은 포지션 청산
    if trader.position:
        last_price = signals_df['price'].iloc[-1]
        last_timestamp = signals_df.index[-1]
        trader.close_position(last_price, last_timestamp, reason="End of Simulation")
        
    print("\n--- 🏁 가상매매 시뮬레이션 종료 ---")


def main(initial_balance: float):
    config = TraderConfig()
    
    # 1. AI 및 MACD 기본 신호 생성
    base_signals_df = get_base_signals(config)
    
    # 2. 규칙 기반 포지션 사이즈 계산
    exposure = get_rule_based_exposure(base_signals_df)
    base_signals_df['exposure'] = exposure
    
    # 3. 가상 트레이더 초기화 및 시뮬레이션 실행
    trader = VirtualTrader(initial_balance, config)
    run_simulation(trader, base_signals_df)

    # 4. 최종 결과 리포트
    final_report = trader.generate_report()
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*25)
    print("  가상매매 최종 결과")
    print("="*25)
    print(f"💰 최종 잔고: ${final_report['Balance']:,.2f}")
    print(f"📈 총 수익률: {final_report['Total Return (%)']:.2f}%")
    print(f"📊 총 거래: {final_report['Trades']}회")
    print(f"🎯 승률: {final_report['Win Rate (%)']:.2f}%")
    print("="*25)
    
    # 거래 로그 저장
    log_df = pd.DataFrame(trader.trade_log)
    log_path = config.OUTPUT_DIR / "virtual_trade_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\n✅ 거래 로그 저장 완료: {log_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="가상 트레이딩 시뮬레이터")
    parser.add_argument(
        "--balance",
        type=float,
        default=10000,
        help="초기 계좌 잔고"
    )
    args = parser.parse_args()
    
    main(initial_balance=args.balance) 