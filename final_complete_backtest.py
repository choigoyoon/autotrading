import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# Load data
signals_df = pd.read_csv('valid_signals.csv')
candles_df = pd.read_csv('analysis_15m.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])

TP = 5.0
SL = 2.0
MAX_HOLD_HOURS = 72

print("=" * 80)
print("최종 전략 전체 분석 리포트")
print("=" * 80)

print("\n📋 데이터 기간:")
print(f"  시작: {candles_df['datetime'].min()}")
print(f"  종료: {candles_df['datetime'].max()}")
print(f"  기간: {(candles_df['datetime'].max() - candles_df['datetime'].min()).days / 365.25:.1f}년")
print(f"  총 캔들: {len(candles_df):,}개 (15분봉)")

print("\n📊 시그널 통계:")
print(f"  총 시그널: {len(signals_df)}건")
print(f"  연평균 시그널: {len(signals_df) / 5.7:.1f}건")
print(f"  월평균 시그널: {len(signals_df) / 5.7 / 12:.1f}건")

def backtest_final_strategy(signal):
    """
    최종 전략:
    1. H-H 하락 추세선 돌파 + HL 확인
    2. 진입
    3. TP 5% 도달 → 익절
    4. 추세선 아래로 '마감' → 손절
    5. SL -2% → 손절
    6. 72시간 경과 → 시간 청산
    """
    entry_time = pd.to_datetime(signal['breakout_time'])
    entry_price = signal['breakout_price']
    trendline_price = signal['trendline_price']
    h1_time = pd.to_datetime(signal['h1_time'])
    h2_time = pd.to_datetime(signal['h2_time'])
    h1_price = signal['h1_price']
    h2_price = signal['h2_price']
    
    # Get future candles (72시간 = 288 캔들)
    future = candles_df[candles_df['datetime'] > entry_time].head(288)
    
    if len(future) == 0:
        return None
    
    # 추세선 기울기 계산
    h1_idx = candles_df[candles_df['datetime'] == h1_time].index[0]
    h2_idx = candles_df[candles_df['datetime'] == h2_time].index[0]
    slope = (h2_price - h1_price) / (h2_idx - h1_idx)
    
    exit_reason = None
    exit_time = None
    exit_price = None
    pnl = 0
    touched_trendline = False
    max_gain = 0
    
    for i, (idx, candle) in enumerate(future.iterrows()):
        # 현재 추세선 가격 계산
        current_trendline = trendline_price + slope * i
        
        # Track max gain
        current_gain = (candle['high'] - entry_price) / entry_price * 100
        max_gain = max(max_gain, current_gain)
        
        # Check TP first (우선순위 1)
        if candle['high'] >= entry_price * (1 + TP/100):
            exit_reason = 'TP'
            exit_time = candle['datetime']
            exit_price = entry_price * (1 + TP/100)
            pnl = TP
            break
        
        # 추세선 터치 체크
        if candle['low'] <= current_trendline:
            touched_trendline = True
        
        # 추세선 아래로 마감 체크 (우선순위 2)
        if candle['close'] < current_trendline:
            exit_reason = 'TRENDLINE_BREAK'
            exit_time = candle['datetime']
            exit_price = candle['close']
            pnl = ((exit_price - entry_price) / entry_price) * 100
            break
        
        # SL 체크 (우선순위 3)
        if candle['low'] <= entry_price * (1 - SL/100):
            exit_reason = 'SL'
            exit_time = candle['datetime']
            exit_price = entry_price * (1 - SL/100)
            pnl = -SL
            break
    
    # Time limit (72시간)
    if exit_reason is None:
        last_candle = future.iloc[-1]
        exit_reason = 'TIME'
        exit_time = last_candle['datetime']
        exit_price = last_candle['close']
        pnl = ((exit_price - entry_price) / entry_price) * 100
    
    # 홀딩 시간 계산
    hold_hours = (exit_time - entry_time).total_seconds() / 3600
    
    return {
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': exit_time,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'pnl': pnl,
        'touched_trendline': touched_trendline,
        'max_gain': max_gain,
        'hold_hours': hold_hours,
        'gap_pct': signal['gap_pct']
    }

# 백테스트 실행
print("\n⏳ 백테스트 실행 중...")
trades = []
for idx, signal in signals_df.iterrows():
    result = backtest_final_strategy(signal)
    if result:
        trades.append({**signal.to_dict(), **result})

trades_df = pd.DataFrame(trades)

# 기본 통계
print("\n" + "=" * 80)
print("📊 전체 성과")
print("=" * 80)

total_trades = len(trades_df)
win_trades = len(trades_df[trades_df['pnl'] > 0])
loss_trades = len(trades_df[trades_df['pnl'] < 0])
breakeven_trades = len(trades_df[trades_df['pnl'] == 0])

print(f"\n총 거래: {total_trades}건")
print(f"  승: {win_trades}건 ({win_trades/total_trades*100:.1f}%)")
print(f"  패: {loss_trades}건 ({loss_trades/total_trades*100:.1f}%)")
print(f"  무: {breakeven_trades}건 ({breakeven_trades/total_trades*100:.1f}%)")

print(f"\n수익률:")
print(f"  평균 수익: {trades_df['pnl'].mean():.2f}%")
print(f"  중앙값 수익: {trades_df['pnl'].median():.2f}%")
print(f"  최대 수익: {trades_df['pnl'].max():.2f}%")
print(f"  최대 손실: {trades_df['pnl'].min():.2f}%")
print(f"  승리 평균: {trades_df[trades_df['pnl'] > 0]['pnl'].mean():.2f}%")
print(f"  손실 평균: {trades_df[trades_df['pnl'] < 0]['pnl'].mean():.2f}%")

# 청산 이유별 분석
print("\n" + "=" * 80)
print("🎯 청산 이유별 분석")
print("=" * 80)

for reason in ['TP', 'TRENDLINE_BREAK', 'SL', 'TIME']:
    subset = trades_df[trades_df['exit_reason'] == reason]
    if len(subset) == 0:
        continue
    
    print(f"\n{reason}:")
    print(f"  건수: {len(subset)}건 ({len(subset)/total_trades*100:.1f}%)")
    print(f"  평균 수익: {subset['pnl'].mean():.2f}%")
    print(f"  평균 홀딩: {subset['hold_hours'].mean():.1f}시간")
    print(f"  평균 최대상승: {subset['max_gain'].mean():.2f}%")
    
    if reason == 'TP':
        print(f"  → TP 도달률: {len(subset)/total_trades*100:.1f}%")
    elif reason == 'TRENDLINE_BREAK':
        touched = subset['touched_trendline'].sum()
        print(f"  → 추세선 터치 후 손절: {touched}건 ({touched/len(subset)*100:.1f}%)")

# 추세선 터치 분석
print("\n" + "=" * 80)
print("📈 추세선 터치 분석")
print("=" * 80)

touched = trades_df[trades_df['touched_trendline'] == True]
not_touched = trades_df[trades_df['touched_trendline'] == False]

print(f"\n추세선 터치함: {len(touched)}건 ({len(touched)/total_trades*100:.1f}%)")
print(f"  평균 수익: {touched['pnl'].mean():.2f}%")
print(f"  TP 도달: {len(touched[touched['exit_reason'] == 'TP'])}건")

print(f"\n추세선 안 닿음: {len(not_touched)}건 ({len(not_touched)/total_trades*100:.1f}%)")
print(f"  평균 수익: {not_touched['pnl'].mean():.2f}%")
print(f"  TP 도달: {len(not_touched[not_touched['exit_reason'] == 'TP'])}건")

# 터치했지만 회복한 케이스
touched_recovered = touched[touched['exit_reason'].isin(['TP', 'TIME'])]
print(f"\n✅ 터치했지만 회복하여 수익: {len(touched_recovered)}건")
print(f"  평균 수익: {touched_recovered['pnl'].mean():.2f}%")
print(f"  TP 도달: {len(touched_recovered[touched_recovered['exit_reason'] == 'TP'])}건")

# 시간별 분석
print("\n" + "=" * 80)
print("⏱️ 홀딩 시간 분석")
print("=" * 80)

print(f"\n평균 홀딩: {trades_df['hold_hours'].mean():.1f}시간")
print(f"중앙값 홀딩: {trades_df['hold_hours'].median():.1f}시간")
print(f"최소 홀딩: {trades_df['hold_hours'].min():.1f}시간")
print(f"최대 홀딩: {trades_df['hold_hours'].max():.1f}시간")

# 연도별 분석
print("\n" + "=" * 80)
print("📅 연도별 성과")
print("=" * 80)

trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year

for year in sorted(trades_df['year'].unique()):
    year_trades = trades_df[trades_df['year'] == year]
    year_pnl = year_trades['pnl'].sum()
    
    print(f"\n{year}년:")
    print(f"  거래: {len(year_trades)}건")
    print(f"  승률: {(year_trades['pnl'] > 0).mean()*100:.1f}%")
    print(f"  평균 수익: {year_trades['pnl'].mean():.2f}%")
    print(f"  누적 수익: {year_pnl:.2f}%")
    print(f"  TP 도달: {(year_trades['exit_reason'] == 'TP').sum()}건 ({(year_trades['exit_reason'] == 'TP').mean()*100:.1f}%)")

# 복리 계산
print("\n" + "=" * 80)
print("💰 복리 수익 시뮬레이션")
print("=" * 80)

capital = 100
capital_history = [capital]
dates = []

for idx, trade in trades_df.iterrows():
    capital = capital * (1 + trade['pnl'] / 100)
    capital_history.append(capital)
    dates.append(trade['exit_time'])

final_capital = capital_history[-1]
total_return = (final_capital - 100) / 100 * 100
years = 5.7
cagr = (final_capital / 100) ** (1 / years) - 1

print(f"\n초기 자본: 100")
print(f"최종 자본: {final_capital:.2f}")
print(f"총 수익률: {total_return:.2f}%")
print(f"연평균 수익률 (CAGR): {cagr*100:.2f}%")

# MDD 계산
peak = 100
mdd = 0
mdd_peak = 100
mdd_trough = 100
mdd_start = None
mdd_end = None

for i, cap in enumerate(capital_history):
    if cap > peak:
        peak = cap
    
    drawdown = (cap - peak) / peak * 100
    if drawdown < mdd:
        mdd = drawdown
        mdd_peak = peak
        mdd_trough = cap
        if i > 0:
            mdd_start = dates[i-1]
            mdd_end = dates[i-1]

print(f"\nMDD (최대 낙폭): {mdd:.2f}%")
print(f"  고점: {mdd_peak:.2f}")
print(f"  저점: {mdd_trough:.2f}")
if mdd_start:
    print(f"  기간: {mdd_start} ~ {mdd_end}")

# Risk metrics
print("\n" + "=" * 80)
print("📊 리스크 지표")
print("=" * 80)

sharpe = trades_df['pnl'].mean() / trades_df['pnl'].std() if trades_df['pnl'].std() > 0 else 0
profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if trades_df[trades_df['pnl'] < 0]['pnl'].sum() != 0 else float('inf')

print(f"\nSharpe Ratio: {sharpe:.2f}")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Win/Loss Ratio: {win_trades/loss_trades:.2f}" if loss_trades > 0 else "Win/Loss Ratio: N/A")
print(f"평균 승/평균 패: {abs(trades_df[trades_df['pnl'] > 0]['pnl'].mean() / trades_df[trades_df['pnl'] < 0]['pnl'].mean()):.2f}" if len(trades_df[trades_df['pnl'] < 0]) > 0 else "평균 승/평균 패: N/A")

# Gap별 분석
print("\n" + "=" * 80)
print("📏 Gap 크기별 성과")
print("=" * 80)

gap_ranges = [
    ("0-0.5%", (0, 0.5)),
    ("0.5-1.0%", (0.5, 1.0)),
    ("1.0-1.5%", (1.0, 1.5)),
    ("1.5%+", (1.5, 100))
]

for name, (low, high) in gap_ranges:
    subset = trades_df[(trades_df['gap_pct'] >= low) & (trades_df['gap_pct'] < high)]
    if len(subset) > 0:
        print(f"\n{name}:")
        print(f"  건수: {len(subset)}건 ({len(subset)/total_trades*100:.1f}%)")
        print(f"  평균 수익: {subset['pnl'].mean():.2f}%")
        print(f"  승률: {(subset['pnl'] > 0).mean()*100:.1f}%")
        print(f"  TP 도달: {(subset['exit_reason'] == 'TP').sum()}건 ({(subset['exit_reason'] == 'TP').mean()*100:.1f}%)")

# 최종 요약
print("\n" + "=" * 80)
print("🎯 최종 요약")
print("=" * 80)

print(f"""
전략: H-H 하락추세선 돌파 + HL 확인 + 추세선 마감 손절
기간: {years:.1f}년 ({candles_df['datetime'].min().date()} ~ {candles_df['datetime'].max().date()})

📊 거래 통계:
  총 거래: {total_trades}건 (연 {total_trades/years:.1f}건, 월 {total_trades/years/12:.1f}건)
  승률: {win_trades/total_trades*100:.1f}%
  평균 수익: {trades_df['pnl'].mean():.2f}%
  평균 홀딩: {trades_df['hold_hours'].mean():.1f}시간

🎯 청산 분석:
  TP (5%): {(trades_df['exit_reason'] == 'TP').sum()}건 ({(trades_df['exit_reason'] == 'TP').mean()*100:.1f}%)
  추세선 이탈: {(trades_df['exit_reason'] == 'TRENDLINE_BREAK').sum()}건 ({(trades_df['exit_reason'] == 'TRENDLINE_BREAK').mean()*100:.1f}%)
  SL (-2%): {(trades_df['exit_reason'] == 'SL').sum()}건 ({(trades_df['exit_reason'] == 'SL').mean()*100:.1f}%)
  시간종료: {(trades_df['exit_reason'] == 'TIME').sum()}건 ({(trades_df['exit_reason'] == 'TIME').mean()*100:.1f}%)

💰 수익성:
  총 수익률: {total_return:.2f}%
  연평균 (CAGR): {cagr*100:.2f}%
  최대 낙폭 (MDD): {mdd:.2f}%
  Sharpe Ratio: {sharpe:.2f}
  Profit Factor: {profit_factor:.2f}

✅ 핵심 인사이트:
  • 추세선 터치: {len(touched)}건 중 {len(touched_recovered)}건이 회복하여 수익
  • 터치 후 회복 시 평균 수익: {touched_recovered['pnl'].mean():.2f}%
  • "터치 ≠ 손절, 마감 = 손절" 로직이 효과적
""")

# 차트 저장
print("\n📊 차트 생성 중...")

# 저장
trades_df.to_csv('final_backtest_results.csv', index=False)
print("\n✅ 결과 저장: final_backtest_results.csv")

# 복리 차트
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Capital curve
ax1.plot(dates, capital_history[1:], linewidth=2)
ax1.set_title('복리 자본 곡선', fontsize=14, weight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Capital')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

# PnL distribution
ax2.hist(trades_df['pnl'], bins=50, edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
ax2.axvline(x=trades_df['pnl'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {trades_df["pnl"].mean():.2f}%')
ax2.set_title('수익률 분포', fontsize=14, weight='bold')
ax2.set_xlabel('PnL (%)')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('final_backtest_charts.png', dpi=150, bbox_inches='tight')
print("✅ 차트 저장: final_backtest_charts.png")

