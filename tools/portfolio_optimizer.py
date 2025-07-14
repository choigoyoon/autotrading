import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# --- 프로젝트 설정 ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from tools.optimize_hybrid_system import get_base_signals, OptimizationConfig as BaseConfig
from tools.backtest_ai_model import run_backtest as run_simple_backtest

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 포트폴리오 최적화 설정 ---
class PortfolioConfig(BaseConfig):
    OUTPUT_DIR = project_root / "results" / "portfolio_optimization"
    # 켈리 공식 최대 베팅 비율 (파산 방지를 위한 안전장치)
    KELLY_MAX_FRACTION = 0.5 

# --- 새로운 백테스팅 엔진 ---
def backtest_with_dynamic_sizing(prices: pd.Series, target_exposure: pd.Series, config: PortfolioConfig):
    """
    동적 포지션 사이징을 반영하는 벡터화된 백테스팅 엔진.
    target_exposure: 포트폴리오의 목표 노출 수준 (-1.0 ~ 1.0)
    """
    # 백테스팅을 위해 데이터 정렬
    data = pd.DataFrame({'price': prices, 'exposure': target_exposure}).dropna()
    prices = data['price']
    target_exposure = data['exposure']

    # 실제 포지션은 신호 발생 다음 캔들부터 적용
    actual_exposure = target_exposure.shift(1).fillna(0)
    
    # 로그 수익률 계산
    log_returns = np.log(prices / prices.shift(1)).fillna(0)
    
    # 전략의 로그 수익률
    strategy_log_returns = actual_exposure * log_returns
    
    # 거래 비용 계산 (노출도 변화량에 비례)
    exposure_changes = actual_exposure.diff().fillna(0).abs()
    costs = exposure_changes * config.FEE
    
    # 비용을 차감한 순수익률
    net_strategy_log_returns = strategy_log_returns - costs
    
    # 누적 수익률 및 자산 곡선
    cumulative_log_returns = net_strategy_log_returns.cumsum()
    equity_curve = config.INITIAL_CAPITAL * np.exp(cumulative_log_returns)

    # 성과 지표 계산
    total_return = (equity_curve.iloc[-1] / config.INITIAL_CAPITAL) - 1
    
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    mdd = drawdown.min()
    
    annualization_factor = np.sqrt(365 * 24 * 60) # 1분봉 기준
    sharpe_ratio = net_strategy_log_returns.mean() / net_strategy_log_returns.std() * annualization_factor if net_strategy_log_returns.std() != 0 else 0
    
    # 거래 횟수는 노출이 0이 아니었던 기간으로 계산
    total_trades = (exposure_changes > 0).sum()
    
    return {
        "Total Return (%)": total_return * 100,
        "MDD (%)": mdd * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Trades": total_trades,
        "Final Equity": equity_curve.iloc[-1],
    }, equity_curve

# --- 포지션 사이징 전략 함수 ---

def get_rule_based_exposure(signals_df: pd.DataFrame) -> pd.Series:
    """사용자가 정의한 규칙에 따라 포지션 크기를 결정합니다."""
    def sizing_rule(row):
        ai_signal = row['ai_signal']
        macd_signal = row['macd_signal']
        confidence = row['ai_confidence']
        
        # 합의 신호
        if ai_signal == macd_signal:
            if confidence >= 0.95: return ai_signal * 0.30
            if confidence >= 0.85: return ai_signal * 0.20
            if confidence >= 0.75: return ai_signal * 0.10
        # AI 단독 신호
        elif confidence >= 0.80:
            return ai_signal * 0.05
        # 신호 충돌 또는 낮은 신뢰도
        return 0.0
        
    return signals_df.apply(sizing_rule, axis=1)

def get_kelly_exposure(signals_df: pd.DataFrame, config: PortfolioConfig) -> pd.Series:
    """켈리 공식을 사용하여 포지션 크기를 결정합니다."""
    print("\n--- 켈리 공식 계산을 위한 기반 데이터 분석 ---")
    # 'Consensus (Strict)' 전략을 기반으로 승률/손익비 계산
    consensus_signals = signals_df.apply(lambda r: r['ai_signal'] if r['ai_signal'] == r['macd_signal'] else 0, axis=1)
    
    # 간단한 백테스트로 거래별 수익률 확보
    _, equity_curve = run_simple_backtest(consensus_signals, signals_df['price'], config)
    trades = consensus_signals.diff().fillna(0).abs()
    trade_returns = equity_curve.pct_change()[trades != 0]

    if trade_returns.empty or trade_returns.std() == 0:
        print("경고: 켈리 공식 계산을 위한 거래 데이터가 부족합니다. 켈리 전략을 건너뜁니다.")
        return pd.Series(0, index=signals_df.index)

    win_rate = (trade_returns > 0).mean()
    
    avg_win = trade_returns[trade_returns > 0].mean()
    avg_loss = abs(trade_returns[trade_returns < 0].mean())
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

    if win_rate == 1.0: # 100% 승률이면 최대치 사용
        kelly_fraction = config.KELLY_MAX_FRACTION
    else:
        kelly_fraction = win_rate - ((1 - win_rate) / payoff_ratio)

    print(f"  - 승률 (p): {win_rate:.2%}")
    print(f"  - 손익비 (b): {payoff_ratio:.2f}")
    print(f"  - 계산된 켈리 비율: {kelly_fraction:.2f}")

    # 파산 방지를 위해 최대값 제한
    final_kelly_fraction = max(0, min(kelly_fraction, config.KELLY_MAX_FRACTION))
    print(f"  - 적용될 켈리 비율 (최대 {config.KELLY_MAX_FRACTION:.0%}): {final_kelly_fraction:.2f}")

    # 켈리 비율을 모든 합의 신호에 적용
    return consensus_signals.apply(lambda x: x * final_kelly_fraction)

# --- 메인 실행 로직 ---

def main():
    config = PortfolioConfig()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. AI 및 MACD 기본 신호 생성
    base_signals_df = get_base_signals(config)
    if base_signals_df.empty:
        return

    # 2. 포지션 사이징 전략별 목표 노출 계산
    exposures = {
        "Rule-Based Sizing": get_rule_based_exposure(base_signals_df),
        "Kelly Criterion Sizing": get_kelly_exposure(base_signals_df, config),
        "Fixed Sizing (Consensus)": base_signals_df.apply(lambda r: r['ai_signal'] if r['ai_signal'] == r['macd_signal'] else 0, axis=1)
    }

    # 3. 전략별 포트폴리오 백테스팅
    all_results = {}
    equity_curves = {}
    print("\n--- 동적 포지셔닝 포트폴리오 백테스팅 실행 ---")
    for name, exposure in tqdm(exposures.items(), desc="포트폴리오 백테스팅 중"):
        results, equity = backtest_with_dynamic_sizing(base_signals_df['price'], exposure, config)
        all_results[name] = results
        equity_curves[name] = equity

    # 4. 결과 정리 및 출력
    print("\n--- 포트폴리오 최적화 결과 ---")
    report = pd.DataFrame(all_results).T.sort_values(by="Sharpe Ratio", ascending=False)
    report_path = config.OUTPUT_DIR / "portfolio_performance_report.csv"
    report.to_csv(report_path)
    
    print(f"✅ 포트폴리오 성과 리포트 저장 완료: {report_path}")
    print(report)

    # 5. 시각화
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))
    
    for name, equity in equity_curves.items():
        perf = all_results[name]
        label = f"{name} (Return: {perf['Total Return (%)']:.2f}%, Sharpe: {perf['Sharpe Ratio']:.2f})"
        ax.plot(equity.index, equity.values, label=label, lw=2)

    ax.set_title('Portfolio Performance by Position Sizing Strategy', fontsize=18, weight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Equity')
    ax.legend()
    ax.grid(True)
    
    plot_path = config.OUTPUT_DIR / 'portfolio_equity_curves.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ 포트폴리오 수익 곡선 그래프 저장 완료: {plot_path}")
    print("\n🎉 포트폴리오 최적화 분석 완료.")

if __name__ == '__main__':
    main() 