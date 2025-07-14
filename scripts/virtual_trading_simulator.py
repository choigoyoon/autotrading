import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import warnings
import os
import argparse
from datetime import timedelta, datetime
from typing import Dict, Optional, List, TypedDict, Any

# --- 프로젝트 설정 ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 외부 의존성 제거
# from tools.optimize_hybrid_system import get_base_signals, OptimizationConfig
# from tools.portfolio_optimizer import get_rule_based_exposure

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 타입 및 데이터 클래스 정의 ---

class TraderConfig:
    FEE: float = 0.001
    SLIPPAGE: float = 0.0005
    STOP_LOSS_PCT: float = -0.02
    TAKE_PROFIT_PCT: float = 0.05
    PARTIAL_PROFIT_FRAC: float = 0.5
    MAX_HOLDING_HOURS: int = 24

class Position(TypedDict):
    entry_time: datetime
    entry_price: float
    size: float
    direction: int  # 1: Long, -1: Short
    initial_stop_loss: float
    take_profit: Optional[float]

class TradeLog(TypedDict, total=False):
    timestamp: datetime
    event: str
    price: float
    direction: int
    size: float
    balance: float
    pnl: Optional[float]
    reason: Optional[str]
    fee: Optional[float]

class Report(TypedDict):
    Balance: float
    Total_PnL: float
    Total_Return_pct: float
    Trades: int
    Win_Rate_pct: float


class VirtualTrader:
    """가상 거래를 수행하고 계좌 상태를 관리하는 클래스."""
    def __init__(self, initial_balance: float, config: TraderConfig):
        self.config = config
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position: Optional[Position] = None
        self.trade_log: List[TradeLog] = []
        print(f"✅ 가상 트레이더 초기화 완료. 초기 잔고: ${initial_balance:,.2f}")

    def execute_trade(self, signal: float, price: float, timestamp: datetime) -> None:
        """신호에 따라 거래를 실행합니다."""
        if (self.position and np.sign(self.position['direction']) == np.sign(signal)) or signal == 0:
            return

        if self.position:
            self.close_position(price, timestamp, reason="New Signal")

        direction = int(np.sign(signal))
        exposure = abs(signal)
        size_to_invest = self.balance * exposure
        
        entry_price = price * (1 + direction * self.config.SLIPPAGE)
        fee = size_to_invest * self.config.FEE
        self.balance -= fee
        
        self.position = {
            'entry_time': timestamp,
            'entry_price': entry_price,
            'size': size_to_invest,
            'direction': direction,
            'initial_stop_loss': entry_price * (1 + self.config.STOP_LOSS_PCT * direction),
            'take_profit': entry_price * (1 + self.config.TAKE_PROFIT_PCT * direction)
        }
        self._log_event("OPEN", self.position, price, timestamp, fee=fee)

    def close_position(self, price: float, timestamp: datetime, reason: str, partial_frac: float = 1.0) -> None:
        """포지션을 청산하고 손익을 기록합니다."""
        if not self.position: return

        pos = self.position
        close_price = price * (1 - pos['direction'] * self.config.SLIPPAGE)
        
        size_to_close = pos['size'] * partial_frac
        pnl = (close_price / pos['entry_price'] - 1) * pos['direction'] * size_to_close
        
        fee = size_to_close * self.config.FEE
        net_pnl = pnl - fee
        
        self.balance += net_pnl
        
        self._log_event("CLOSE", pos, close_price, timestamp, pnl=net_pnl, reason=reason, fee=fee)
        
        if partial_frac == 1.0:
            self.position = None
        else:
            pos['size'] -= size_to_close

    def manage_positions(self, current_price: float, timestamp: datetime) -> None:
        """매 틱마다 포지션을 관리 (손절, 익절, 최대 보유 시간)"""
        if not self.position: return
        
        pos = self.position
        
        holding_duration = timestamp - pos['entry_time']
        if holding_duration >= timedelta(hours=self.config.MAX_HOLDING_HOURS):
            self.close_position(current_price, timestamp, reason="Max Hold Time")
            return

        pnl_ratio = (current_price / pos['entry_price'] - 1) * pos['direction']
        
        if pnl_ratio <= self.config.STOP_LOSS_PCT:
            self.close_position(current_price, timestamp, reason="Stop Loss")
            return
            
        take_profit_price = pos.get('take_profit')
        if take_profit_price is not None:
            is_long_tp = pos['direction'] == 1 and current_price >= take_profit_price
            is_short_tp = pos['direction'] == -1 and current_price <= take_profit_price
            
            if is_long_tp or is_short_tp:
                self.close_position(current_price, timestamp, reason="Take Profit (Partial)", partial_frac=self.config.PARTIAL_PROFIT_FRAC)
                if self.position:
                    self.position['take_profit'] = None

    def _log_event(self, event_type: str, pos: Position, price: float, ts: datetime, **kwargs: Any) -> None:
        log_entry: TradeLog = {
            'timestamp': ts,
            'event': event_type,
            'price': price,
            'direction': pos['direction'],
            'size': pos['size'],
            'balance': self.balance,
            # kwargs에서 올 수 있는 선택적 키들을 .get()으로 안전하게 처리
            'pnl': kwargs.get('pnl'),
            'reason': kwargs.get('reason'),
            'fee': kwargs.get('fee')
        }
        self.trade_log.append(log_entry)

    def generate_report(self) -> Report:
        """현재 계좌 상태 리포트를 생성합니다."""
        total_pnl = self.balance - self.initial_balance
        total_return_pct = (total_pnl / self.initial_balance) if self.initial_balance != 0 else 0.0
        
        closed_trades = [t for t in self.trade_log if t.get('event') == 'CLOSE']
        wins = [t for t in closed_trades if (t.get('pnl') or 0.0) > 0]
        
        win_rate = len(wins) / len(closed_trades) if closed_trades else 0.0
        
        return {
            "Balance": self.balance,
            "Total_PnL": total_pnl,
            "Total_Return_pct": total_return_pct * 100,
            "Trades": len(closed_trades),
            "Win_Rate_pct": win_rate * 100
        }

def run_simulation(trader: VirtualTrader, signals_df: pd.DataFrame) -> None:
    """전체 데이터에 대해 가상매매 시뮬레이션을 실행합니다."""
    print("\n--- 🚀 가상매매 시뮬레이션 시작 ---")
    
    # iterrows가 튜플을 반환하므로 타입을 명시적으로 지정
    for row_tuple in tqdm(signals_df.iterrows(), total=len(signals_df), desc="가상매매 진행 중"):
        timestamp, row = row_tuple
        
        if not isinstance(timestamp, datetime):
            print(f"경고: 타임스탬프 타입이 올바르지 않아 건너뜁니다: {type(timestamp)}")
            continue
            
        current_price = float(row['price'])
        
        trader.manage_positions(current_price, timestamp)
        
        signal = float(row['exposure'])
        trader.execute_trade(signal, current_price, timestamp)
        
    if trader.position:
        last_price = float(signals_df['price'].iloc[-1])
        last_timestamp = signals_df.index[-1]
        if isinstance(last_timestamp, datetime):
            trader.close_position(last_price, last_timestamp, reason="End of Simulation")
        
    print("\n--- 🏁 가상매매 시뮬레이션 종료 ---")


def main(initial_balance: float) -> None:
    config = TraderConfig()
    
    print("--- 🔧 테스트 데이터 생성 중 ---")
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=5000, freq='h'))
    price_data = 10000 + (np.random.randn(5000).cumsum() * 10)
    exposure_data = np.random.choice([1, 0.5, 0, -0.5, -1], 5000, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    
    signals_df = pd.DataFrame({
        'price': price_data,
        'exposure': exposure_data
    }, index=dates)
    print(f"✅ 테스트 데이터 생성 완료: {len(signals_df)} 시간 데이터")

    trader = VirtualTrader(initial_balance, config)
    run_simulation(trader, signals_df)

    final_report = trader.generate_report()
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("="*25)
    print("  가상매매 최종 결과")
    print("="*25)
    print(f"💰 최종 잔고: ${final_report['Balance']:,.2f}")
    print(f"📈 총 수익률: {final_report['Total_Return_pct']:.2f}%")
    print(f"📊 총 거래: {final_report['Trades']}회")
    print(f"🎯 승률: {final_report['Win_Rate_pct']:.2f}%")
    print("="*25)
    
    log_df = pd.DataFrame(trader.trade_log)
    output_dir = project_root / "results" / "virtual_trades"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "virtual_trade_log.csv"
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