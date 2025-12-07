import pandas as pd
import numpy as np

print("=" * 80)
print("🔍 확정 공간 백테스트 실패 사유 상세 분석")
print("=" * 80)
print()

# Load backtest results
df_trades = pd.read_csv('backtest_confirmation_space_results.csv')
df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])

print(f"총 거래: {len(df_trades)}개")
print()

# Separate by result
sl_trades = df_trades[df_trades['exit_reason'] == 'SL'].copy()
tp1_be_trades = df_trades[df_trades['exit_reason'] == 'TP1_Breakeven'].copy()
tp2_trades = df_trades[df_trades['exit_reason'] == 'TP2_Full'].copy()

print("=" * 80)
print("1단계: 각 결과별 손익 분석")
print("=" * 80)
print()

print(f"🔴 SL 손절: {len(sl_trades)}개 (18.0%)")
print(f"   총 손실: {sl_trades['pnl_pct'].sum():.2f}%")
print(f"   평균 손실: {sl_trades['pnl_pct'].mean():.2f}%")
print()

print(f"🟡 TP1 Breakeven: {len(tp1_be_trades)}개 (36.3%)")
print(f"   총 손익: {tp1_be_trades['pnl_pct'].sum():.2f}%")
print(f"   평균 손익: {tp1_be_trades['pnl_pct'].mean():.2f}%")
print()

print(f"🟢 TP2 Full: {len(tp2_trades)}개 (45.7%)")
print(f"   총 수익: {tp2_trades['pnl_pct'].sum():.2f}%")
print(f"   평균 수익: {tp2_trades['pnl_pct'].mean():.2f}%")
print()

# Calculate net effect
total_pnl = sl_trades['pnl_pct'].sum() + tp1_be_trades['pnl_pct'].sum() + tp2_trades['pnl_pct'].sum()
print(f"📊 총합: {total_pnl:.2f}%")
print()

# Key insight
print("💡 핵심 문제:")
sl_loss = abs(sl_trades['pnl_pct'].sum())
tp2_profit = tp2_trades['pnl_pct'].sum()
tp1_profit = tp1_be_trades['pnl_pct'].sum()

print(f"   SL 손실: -{sl_loss:.2f}%")
print(f"   TP2 수익: +{tp2_profit:.2f}%")
print(f"   TP1 수익: +{tp1_profit:.2f}%")
print(f"   순손익: {tp2_profit + tp1_profit - sl_loss:.2f}%")
print()

if abs(sl_loss) > (tp2_profit + tp1_profit):
    print("   ❌ SL 손실이 TP 수익보다 큼!")
else:
    print("   ✅ TP 수익이 SL 손실보다 큼")
print()

# Analyze TP1 Breakeven problem
print("=" * 80)
print("2단계: TP1 Breakeven 문제 분석")
print("=" * 80)
print()

print(f"TP1 Breakeven {len(tp1_be_trades)}개 중:")
print()

# Calculate how much they earned at TP1 before going back to breakeven
tp1_be_trades['tp1_profit_pct'] = (tp1_be_trades['tp2_price'] - tp1_be_trades['entry_price']) / tp1_be_trades['entry_price'] * 100

# How close did they get to TP2?
# We need to load candle data to check max price
df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])

tp1_be_analysis = []

for idx, trade in tp1_be_trades.head(50).iterrows():  # Sample 50 for speed
    entry_time = trade['entry_time']
    exit_time = trade['exit_time']
    tp2_price = trade['tp2_price']
    entry_price = trade['entry_price']
    
    # Get candles between entry and exit
    candles_between = df_15m[(df_15m['datetime'] > entry_time) & (df_15m['datetime'] <= exit_time)]
    
    if len(candles_between) > 0:
        max_price = candles_between['high'].max()
        distance_to_tp2_pct = (tp2_price - max_price) / entry_price * 100
        reached_tp2_pct = (max_price - entry_price) / (tp2_price - entry_price) * 100 if tp2_price > entry_price else 0
        
        tp1_be_analysis.append({
            'entry_time': entry_time,
            'max_price_reached': max_price,
            'tp2_price': tp2_price,
            'reached_tp2_pct': reached_tp2_pct,
            'distance_to_tp2_pct': distance_to_tp2_pct
        })

df_tp1_be_analysis = pd.DataFrame(tp1_be_analysis)

if len(df_tp1_be_analysis) > 0:
    avg_tp2_reached = df_tp1_be_analysis['reached_tp2_pct'].mean()
    print(f"평균적으로 TP2의 {avg_tp2_reached:.1f}%까지 도달")
    print()
    
    # How many got close to TP2?
    close_to_tp2 = len(df_tp1_be_analysis[df_tp1_be_analysis['reached_tp2_pct'] >= 90])
    print(f"TP2 90% 이상 근접: {close_to_tp2}개 ({close_to_tp2/len(df_tp1_be_analysis)*100:.1f}%)")
    print(f"   → TP2 거의 도달했는데 브레이크이븐으로 끝남")
    print(f"   → **SL을 브레이크이븐으로 이동 타이밍 문제!**")
    print()

# Analyze by year
print("=" * 80)
print("3단계: 연도별 실패 사유")
print("=" * 80)
print()

for year in sorted(df_trades['year'].unique()):
    year_trades = df_trades[df_trades['year'] == year]
    year_sl = year_trades[year_trades['exit_reason'] == 'SL']
    year_tp1 = year_trades[year_trades['exit_reason'] == 'TP1_Breakeven']
    year_tp2 = year_trades[year_trades['exit_reason'] == 'TP2_Full']
    
    total_pnl = year_trades['pnl_pct'].sum()
    
    print(f"{year}년: 총 {total_pnl:+.2f}%")
    print(f"   거래: {len(year_trades)}개")
    print(f"   SL: {len(year_sl)}개 ({len(year_sl)/len(year_trades)*100:.1f}%) → {year_sl['pnl_pct'].sum():.2f}%")
    print(f"   TP1 BE: {len(year_tp1)}개 ({len(year_tp1)/len(year_trades)*100:.1f}%) → {year_tp1['pnl_pct'].sum():.2f}%")
    print(f"   TP2: {len(year_tp2)}개 ({len(year_tp2)/len(year_trades)*100:.1f}%) → {year_tp2['pnl_pct'].sum():.2f}%")
    print()

# Analyze 2021 specifically
print("=" * 80)
print("4단계: 2021년 폭망 사유 집중 분석")
print("=" * 80)
print()

year_2021 = df_trades[df_trades['year'] == 2021]
print(f"2021년 총 손실: {year_2021['pnl_pct'].sum():.2f}%")
print(f"2021년 거래: {len(year_2021)}개")
print()

print("2021년 특징:")
sl_2021 = year_2021[year_2021['exit_reason'] == 'SL']
tp1_2021 = year_2021[year_2021['exit_reason'] == 'TP1_Breakeven']
tp2_2021 = year_2021[year_2021['exit_reason'] == 'TP2_Full']

print(f"   SL: {len(sl_2021)}개 ({len(sl_2021)/len(year_2021)*100:.1f}%) → {sl_2021['pnl_pct'].sum():.2f}%")
print(f"   TP1 BE: {len(tp1_2021)}개 ({len(tp1_2021)/len(year_2021)*100:.1f}%) → {tp1_2021['pnl_pct'].sum():.2f}%")
print(f"   TP2: {len(tp2_2021)}개 ({len(tp2_2021)/len(year_2021)*100:.1f}%) → {tp2_2021['pnl_pct'].sum():.2f}%")
print()

# Compare with other years
other_years = df_trades[df_trades['year'] != 2021]
print("다른 연도 평균:")
print(f"   SL 비율: {len(other_years[other_years['exit_reason'] == 'SL'])/len(other_years)*100:.1f}%")
print(f"   TP2 비율: {len(other_years[other_years['exit_reason'] == 'TP2_Full'])/len(other_years)*100:.1f}%")
print()

# Power score analysis
print("=" * 80)
print("5단계: Power Score vs 실제 결과")
print("=" * 80)
print()

for score in sorted(df_trades['power_score'].unique()):
    score_trades = df_trades[df_trades['power_score'] == score]
    score_sl = score_trades[score_trades['exit_reason'] == 'SL']
    score_tp2 = score_trades[score_trades['exit_reason'] == 'TP2_Full']
    
    avg_pnl = score_trades['pnl_pct'].mean()
    total_pnl = score_trades['pnl_pct'].sum()
    
    print(f"Score {score}점 ({len(score_trades)}개):")
    print(f"   총 PnL: {total_pnl:+.2f}% | 평균 PnL: {avg_pnl:+.2f}%")
    print(f"   SL: {len(score_sl)}개 ({len(score_sl)/len(score_trades)*100:.1f}%)")
    print(f"   TP2: {len(score_tp2)}개 ({len(score_tp2)/len(score_trades)*100:.1f}%)")
    print()

# Final diagnosis
print("=" * 80)
print("🎯 최종 진단: 실패 사유")
print("=" * 80)
print()

print("1️⃣ **TP1 Breakeven 문제** (36.3%, -25.89%)")
print(f"   - TP1 도달 후 브레이크이븐으로 SL 이동")
print(f"   - 평균 TP2의 {avg_tp2_reached:.1f}%까지만 도달")
print(f"   - TP2 거의 도달했는데 되돌아옴")
print(f"   → 해결책: SL 브레이크이븐 이동 타이밍 지연 or TP1 익절 비중 ↑")
print()

print("2️⃣ **2021년 특이 시장** (-25.22%)")
print(f"   - 51개 거래 중 SL {len(sl_2021)}개, TP2 {len(tp2_2021)}개")
print(f"   - 전체 손실의 48% 차지")
print(f"   → 해결책: 2021년 필터링 or 변동성 필터 추가")
print()

print("3️⃣ **Power Score 역설** (높을수록 손실)")
print(f"   - 10점: 5개, 평균 -0.77%")
print(f"   - 3점: 100개, 평균 -0.04%")
print(f"   → 해결책: Power score 계산 방식 재검토")
print()

print("4️⃣ **TP/SL 비율 문제**")
print(f"   - SL 손실: {sl_loss:.2f}%")
print(f"   - TP 수익: {tp2_profit + tp1_profit:.2f}%")
print(f"   - 차이: {tp2_profit + tp1_profit - sl_loss:.2f}%")
print(f"   → 해결책: TP1 비중 70% or 전량 TP1 익절")
print()

# Save detailed analysis
sl_trades.to_csv('failure_sl_trades.csv', index=False)
tp1_be_trades.to_csv('failure_tp1_breakeven_trades.csv', index=False)

print("✅ 상세 분석 저장:")
print("   - failure_sl_trades.csv (SL 케이스)")
print("   - failure_tp1_breakeven_trades.csv (TP1 BE 케이스)")

