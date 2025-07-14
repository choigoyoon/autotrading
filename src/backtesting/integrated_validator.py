import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from typing import Optional, Dict, Any, cast, TypedDict, List
from datetime import datetime

# 리팩토링된 모듈을 src 내부에서 직접 import
from src.strategies.take_profit_system import IntegratedTakeProfitSystem
from src.analysis.multi_timeframe_validator import MultiTimeframeValidator

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class TradeResult(TypedDict):
    entry_timestamp: datetime
    exit_timestamp: datetime
    holding_period_min: float
    label_type: str
    label_grade: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    exit_reason: str
    is_win: int


class IntegratedBacktestValidator:
    """
    라벨, 동적 전략, 시장 환경을 모두 통합하여 종합적인 백테스트를 수행하고
    다양한 관점에서 성과를 검증하는 클래스.
    """
    def __init__(self, base_path: Path):
        """
        초기화 메서드. 경로 설정, 결과 저장 폴더 생성, 분석 모듈 인스턴스화.
        """
        if IntegratedTakeProfitSystem is None or MultiTimeframeValidator is None:
            raise ImportError("필수 분석 모듈이 로드되지 않았습니다. 프로그램을 종료합니다.")
            
        self.base_path = base_path
        self.results_path = self.base_path / "results" / "integrated_backtest"
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # 분석 시스템 초기화
        self.tp_system = IntegratedTakeProfitSystem(base_path)
        self.mtf_validator = MultiTimeframeValidator(base_path)
        
        self.all_trades: Optional[pd.DataFrame] = None
        self.ohlcv_1m: Optional[pd.DataFrame] = None
        print("IntegratedBacktestValidator가 초기화되었습니다.")

    def _load_data(self):
        """백테스트에 필요한 1분봉 데이터와 라벨 데이터를 로드합니다."""
        print("데이터 로딩 중...")
        # 1분봉 데이터 로드
        ohlcv_path = self.base_path / "data" / "processed" / "btc_usdt_kst" / "resampled_ohlcv" / "1min.parquet"
        if not ohlcv_path.exists():
            raise FileNotFoundError(f"1분봉 데이터를 찾을 수 없습니다: {ohlcv_path}")
        self.ohlcv_1m = pd.read_parquet(ohlcv_path)
        
        # 라벨 데이터는 TP 시스템을 통해 접근 (내부적으로 graded_df 로드)
        self.tp_system.prepare_analyzers()
        print("데이터 로딩 완료.")

    def _run_single_trade_simulation(self, label_info: pd.Series) -> Optional[TradeResult]:
        """단일 라벨에 대한 거래 시뮬레이션을 실행합니다."""
        if self.ohlcv_1m is None:
            return None # 데이터 미로드 시 시뮬레이션 불가

        entry_timestamp = cast(pd.Timestamp, pd.to_datetime(label_info['timestamp']))
        
        # 1. 전략 가져오기
        try:
            strategy = self.tp_system.get_full_strategy(entry_timestamp)
        except (ValueError, RuntimeError):
            # print(f"전략 생성 실패({entry_timestamp}): {e}")
            return None # 전략 생성 실패 시 거래 스킵

        # 2. 진입 가격 설정
        if label_info['label_type'] == 1:
            entry_price = float(self.ohlcv_1m.at[entry_timestamp, 'low'])
        else:
            entry_price = float(self.ohlcv_1m.at[entry_timestamp, 'high'])
        
        # 3. 거래 기간 데이터 슬라이싱
        max_holding_minutes = int(strategy['max_holding_period_min'])
        trade_duration = pd.to_timedelta(max_holding_minutes, unit='m')
        
        if pd.isna(trade_duration):
            return None # 유효하지 않은 보유 기간

        end_timestamp = entry_timestamp + trade_duration
        trade_period_df = self.ohlcv_1m.loc[entry_timestamp:end_timestamp]

        if trade_period_df.empty:
            return None

        # 4. 출구 조건 탐색
        exit_price, exit_timestamp, exit_reason = 0.0, None, "Max Hold"
        
        tp_price = entry_price * (1 + strategy['adjusted_take_profit_pct']) if label_info['label_type'] == 1 else entry_price * (1 - strategy['adjusted_take_profit_pct'])
        stop_loss_pct = strategy.get('stop_loss_pct', -0.02)  # 기본값 -2%
        sl_price = entry_price * (1 + stop_loss_pct) if label_info['label_type'] == 1 else entry_price * (1 - stop_loss_pct)
        
        for idx, candle in trade_period_df.iloc[1:].iterrows(): # 진입 캔들 제외
            current_time = cast(pd.Timestamp, idx)
            if label_info['label_type'] == 1: # 매수
                if candle['high'] >= tp_price:
                    exit_price, exit_timestamp, exit_reason = tp_price, current_time, "Take Profit"
                    break
                if candle['low'] <= sl_price:
                    exit_price, exit_timestamp, exit_reason = sl_price, current_time, "Stop Loss"
                    break
            else: # 매도
                if candle['low'] <= tp_price:
                    exit_price, exit_timestamp, exit_reason = tp_price, current_time, "Take Profit"
                    break
                if candle['high'] >= sl_price:
                    exit_price, exit_timestamp, exit_reason = sl_price, current_time, "Stop Loss"
                    break

        if exit_timestamp is None: # 최대 보유 기간 도달
            exit_timestamp = cast(pd.Timestamp, trade_period_df.index[-1])
            exit_price = float(trade_period_df['close'].iloc[-1])
        
        exit_timestamp = cast(pd.Timestamp, exit_timestamp)

        # 5. 거래 결과 기록
        pnl_pct = (exit_price - entry_price) / entry_price if label_info['label_type'] == 1 else (entry_price - exit_price) / entry_price
        
        return {
            'entry_timestamp': entry_timestamp.to_pydatetime(),
            'exit_timestamp': exit_timestamp.to_pydatetime(),
            'holding_period_min': (exit_timestamp - entry_timestamp).total_seconds() / 60,
            'label_type': 'buy' if label_info['label_type'] == 1 else 'sell',
            'label_grade': strategy.get('label_grade', 'N/A'),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'is_win': 1 if pnl_pct > 0 else 0,
        }

    def run_comprehensive_backtest(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        모든 라벨에 대해 백테스트를 실행하고 거래 로그를 생성합니다.
        """
        self._load_data()
        
        # bounce_analyzer에서 직접 등급 라벨을 가져옵니다.
        labels_df = cast(pd.DataFrame, self.tp_system.bounce_analyzer.get_graded_labels())
        if not isinstance(labels_df, pd.DataFrame) or labels_df.empty:
            raise ValueError("라벨 등급 데이터(graded_df)를 생성할 수 없습니다.")
            
        labels_df = labels_df.copy()
        if start_date:
            labels_df = labels_df[labels_df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            labels_df = labels_df[labels_df['timestamp'] <= pd.to_datetime(end_date)]

        trade_results: List[TradeResult] = []
        print(f"{len(labels_df)}개의 라벨에 대해 백테스트를 시작합니다...")
        
        # iterrows를 사용하기 전, 다시 한번 DataFrame인지 확인
        if isinstance(labels_df, pd.DataFrame):
            for _, label_info in tqdm(labels_df.iterrows(), total=len(labels_df)):
                trade_result = self._run_single_trade_simulation(label_info)
                if trade_result:
                    trade_results.append(trade_result)
        
        if not trade_results:
            print("경고: 백테스트 결과 생성된 거래가 없습니다.")
            self.all_trades = pd.DataFrame()
        else:
            self.all_trades = pd.DataFrame(trade_results)
        
        if self.all_trades.empty:
            print("생성된 거래가 없어 리포트를 종료합니다.")
            return

        self.all_trades.to_csv(self.results_path / "full_trade_log.csv", index=False)
        print(f"백테스트 완료. 총 {len(self.all_trades)}건의 거래가 기록되었습니다.")
        
    def _calculate_performance_metrics(self, trades_df: pd.DataFrame, title: str) -> pd.Series:
        """거래 로그를 바탕으로 주요 성과 지표를 계산합니다."""
        if trades_df.empty:
            return pd.Series(dtype='float64', name=title)
            
        total_trades = len(trades_df)
        win_rate = trades_df['is_win'].mean() * 100 if total_trades > 0 else 0
        avg_return = trades_df['pnl_pct'].mean() * 100 if total_trades > 0 else 0
        
        # 최대 낙폭 (MDD) 계산
        trades_df = trades_df.copy()
        trades_df['cumulative_pnl'] = (1 + trades_df['pnl_pct']).cumprod()
        peak = trades_df['cumulative_pnl'].expanding().max()
        drawdown = (trades_df['cumulative_pnl'] - peak) / peak
        max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0
        
        # 샤프 비율 (일 단위, 무위험 수익률 0 가정)
        if pd.api.types.is_datetime64_any_dtype(trades_df.index):
            daily_returns = trades_df['pnl_pct'].resample('D').sum()
        else:
            daily_returns = trades_df.set_index('exit_timestamp')['pnl_pct'].resample('D').sum()

        sharpe_ratio = (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(365) if not daily_returns.empty and daily_returns.std() > 0 else 0.0
        
        gains = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
        losses = abs(trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
        profit_factor = gains / losses if losses > 0 else np.inf

        return pd.Series({
            '총 거래 수': total_trades,
            '승률 (%)': win_rate,
            '평균 수익률 (%)': avg_return,
            '최대 낙폭 (%)': max_drawdown,
            '샤프 비율': sharpe_ratio,
            '수익 팩터': profit_factor,
            '평균 보유 시간 (분)': trades_df['holding_period_min'].mean()
        }, name=title)

    def validate_by_period(self):
        """기간별(연도, 분기, 시장 상황) 성과를 검증합니다."""
        if self.all_trades is None or self.all_trades.empty:
            raise RuntimeError("먼저 run_comprehensive_backtest()를 실행하세요.")

        if self.ohlcv_1m is None:
            raise RuntimeError("OHLCV 데이터가 로드되지 않았습니다. run_comprehensive_backtest()를 먼저 실행하세요.")

        print("기간별 성과 분석 중...")
        trades = self.all_trades.copy()
        trades['entry_timestamp'] = pd.to_datetime(trades['entry_timestamp'])
        trades['year'] = trades['entry_timestamp'].dt.year
        trades['quarter'] = trades['entry_timestamp'].dt.to_period('Q').astype(str)
        
        # 시장 상황 정의 (예시: 연간 수익률 기반)
        yearly_returns = self.ohlcv_1m['close'].resample('Y').last().pct_change().to_dict()
        trades['market_regime'] = trades['year'].apply(
            lambda y: '상승장' if yearly_returns.get(pd.to_datetime(f"{y}-12-31"), 0) > 0.15 
            else ('하락장' if yearly_returns.get(pd.to_datetime(f"{y}-12-31"), 0) < -0.15 else '횡보장')
        )
        
        report = []
        # 전체 기간
        report.append(self._calculate_performance_metrics(trades, "전체 기간"))
        # 연도별
        for year, group in trades.groupby('year'):
            report.append(self._calculate_performance_metrics(group, f"{year}년"))
        # 분기별
        for quarter, group in trades.groupby('quarter'):
            report.append(self._calculate_performance_metrics(group, str(quarter)))
        # 시장 상황별
        for regime, group in trades.groupby('market_regime'):
            report.append(self._calculate_performance_metrics(group, str(regime)))
            
        period_report_df = pd.DataFrame(report)
        period_report_df.to_csv(self.results_path / "period_performance_report.csv")
        print("기간별 성과 리포트가 저장되었습니다.")
        return period_report_df

    def validate_by_grade_and_environment(self):
        """라벨 등급 및 시장 환경 점수별 성과를 검증합니다."""
        if self.all_trades is None or self.all_trades.empty:
            raise RuntimeError("먼저 run_comprehensive_backtest()를 실행하세요.")
        
        print("등급 및 환경별 성과 분석 중...")
        trades = self.all_trades.copy()
        
        report = []
        # 등급별
        for grade, group in trades.groupby('label_grade'):
             report.append(self._calculate_performance_metrics(group, f"등급: {grade}"))
        # 거래 유형별
        for ltype, group in trades.groupby('label_type'):
            report.append(self._calculate_performance_metrics(group, f"유형: {str(ltype).upper()}"))
        # 출구 사유별
        for reason, group in trades.groupby('exit_reason'):
            report.append(self._calculate_performance_metrics(group, f"출구: {reason}"))
            
        grade_report_df = pd.DataFrame(report)
        grade_report_df.to_csv(self.results_path / "grade_env_performance_report.csv")
        print("등급 및 환경별 성과 리포트가 저장되었습니다.")
        return grade_report_df

    def _generate_html_report(self, reports: dict):
        """분석 결과를 종합하여 HTML 대시보드를 생성합니다."""
        print("HTML 리포트 생성 중...")
        
        html = "<html><head><title>통합 백테스트 리포트</title>"
        html += "<style>body{font-family: sans-serif;} table{border-collapse: collapse; margin: 25px 0;} th, td{border: 1px solid #ddd; padding: 8px;} th{background-color: #f2f2f2;} h1, h2{text-align: center;}</style>"
        html += "</head><body>"
        html += "<h1>통합 백테스트 리포트</h1>"

        # 이미지 추가
        html += "<h2>수익 곡선</h2>"
        html += '<img src="equity_curve.png" alt="Equity Curve" style="width:100%;">'
        
        for name, df in reports.items():
            html += f"<h2>{name}</h2>"
            html += df.to_html()
            
        html += "</body></html>"
        
        with open(self.results_path / "dashboard.html", "w", encoding="utf-8") as f:
            _ = f.write(html)
        print("HTML 대시보드가 생성되었습니다.")
        
    def _generate_charts(self):
        """수익 곡선 등 주요 차트를 생성합니다."""
        if self.all_trades is None:
            raise RuntimeError("먼저 run_comprehensive_backtest()를 실행하세요.")
            
        print("차트 생성 중...")
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 수익 곡선 (Equity Curve)
        fig, ax = plt.subplots(figsize=(15, 7))
        if 'cumulative_pnl' in self.all_trades.columns:
            _ = self.all_trades['cumulative_pnl'].plot(ax=ax)
        _ = ax.set_title('Equity Curve', fontsize=16)
        _ = ax.set_ylabel('Cumulative Return')
        _ = ax.set_xlabel('Trade Number')
        fig.tight_layout()
        fig.savefig(self.results_path / "equity_curve.png")
        plt.close(fig)
        
        print("차트 생성이 완료되었습니다.")

if __name__ == '__main__':
    try:
        project_root = Path(__file__).resolve().parents[2]
        validator = IntegratedBacktestValidator(base_path=project_root)
        
        # 전체 기간에 대해 백테스트 실행
        validator.run_comprehensive_backtest(start_date="2020-01-01")
        
        # 성과 검증
        period_report = validator.validate_by_period()
        grade_report = validator.validate_by_grade_and_environment()
        
        # 리포트 생성
        validator._generate_charts()
        validator._generate_html_report({
            "기간별 성과": period_report,
            "등급/환경별 성과": grade_report
        })
        
        print(f"\n모든 분석 및 리포트 생성이 완료되었습니다. 결과는 '{validator.results_path}' 폴더를 확인하세요.")

    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as e:
        print(f"\n[오류] 실행 중 문제가 발생했습니다: {e}")
    except Exception as e:
        import traceback
        print(f"\n[예상치 못한 오류] {e}")
        traceback.print_exc() 