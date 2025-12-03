"""
HL 손절이 왜 이렇게 많은지 분석
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

signals = pd.read_csv('valid_signals.csv')
results = pd.read_csv('correct_logic_backtest.csv')

print("="*80)
print("HL 손절 문제 분석")
print("="*80)

# Entry와 HL 관계 분석
signals['entry_hl_gap'] = (signals['breakout_price'] - signals['hl_price']) / signals['breakout_price'] * 100

print("\n📊 Entry - HL 거리 분석")
print(f"  평균: {signals['entry_hl_gap'].mean():.2f}%")
print(f"  중앙값: {signals['entry_hl_gap'].median():.2f}%")
print(f"  최소: {signals['entry_hl_gap'].min():.2f}%")
print(f"  최대: {signals['entry_hl_gap'].max():.2f}%")

# 분포
print("\n📊 Entry - HL 거리 분포")
bins = [-10, -1, 0, 0.5, 1, 2, 5, 10, 100]
labels = ['<-1%', '-1~0%', '0~0.5%', '0.5~1%', '1~2%', '2~5%', '5~10%', '>10%']
signals['gap_bin'] = pd.cut(signals['entry_hl_gap'], bins=bins, labels=labels)
dist = signals['gap_bin'].value_counts().sort_index()

for cat, count in dist.items():
    pct = count / len(signals) * 100
    print(f"  {cat}: {count}건 ({pct:.1f}%)")

# HL이 Entry보다 높은 경우 (문제!)
hl_above_entry = (signals['hl_price'] > signals['breakout_price']).sum()
print(f"\n⚠️ HL이 Entry보다 높은 경우: {hl_above_entry}건 ({hl_above_entry/len(signals)*100:.1f}%)")

# HL이 Entry와 거의 같은 경우 (0.5% 이내)
hl_very_close = (abs(signals['entry_hl_gap']) < 0.5).sum()
print(f"⚠️ HL이 Entry와 0.5% 이내: {hl_very_close}건 ({hl_very_close/len(signals)*100:.1f}%)")

# 손절된 케이스 vs 익절된 케이스 비교
print("\n" + "="*80)
print("손절 vs 익절 케이스 비교")
print("="*80)

sl_cases = results[results['exit_reason'] == 'SL_HL_BREAK']
tp_cases = results[results['exit_reason'] == 'TP_TRENDLINE']

# 머지해서 비교
signals_merged = signals.copy()
signals_merged['exit_reason'] = results['exit_reason'].values
signals_merged['total_pnl'] = results['total_pnl'].values

print("\n손절 케이스 (SL_HL_BREAK):")
sl_data = signals_merged[signals_merged['exit_reason'] == 'SL_HL_BREAK']
print(f"  Entry-HL 거리 평균: {sl_data['entry_hl_gap'].mean():.2f}%")

print("\n익절 케이스 (TP_TRENDLINE):")
tp_data = signals_merged[signals_merged['exit_reason'] == 'TP_TRENDLINE']
print(f"  Entry-HL 거리 평균: {tp_data['entry_hl_gap'].mean():.2f}%")

# 문제 원인 파악
print("\n" + "="*80)
print("🔍 문제 원인")
print("="*80)

print("""
문제: HL이 Entry와 너무 가까움!

진입가: 10,000원
HL:     9,990원 (0.1% 아래)

→ 조금만 떨어져도 바로 손절됨
→ 정상적인 변동성에도 손절 발생
""")

# 해결책 제안
print("\n" + "="*80)
print("💡 해결책")
print("="*80)

print("""
1. HL 버퍼 추가: HL - 0.5% 아래를 손절가로

2. HL 재정의: 
   - 현재 HL = 돌파 후 확인된 저점
   - 문제: 돌파 직후라 Entry와 거의 같음
   
3. 다른 저점 기준:
   - 돌파 캔들의 Low
   - 최근 N봉 중 최저점
""")

