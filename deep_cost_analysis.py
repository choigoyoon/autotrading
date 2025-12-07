#!/usr/bin/env python3
"""
심층 비용 분석 - 진짜 남는게 있는지
"""

import pandas as pd
import numpy as np

df = pd.read_csv('L_value_backtest.csv')
df = df.drop_duplicates(subset=['entry_time'])

print("="*80)
print("🔬 심층 비용 분석: 진짜 남는 게 있나?")
print("="*80)

# 현실적인 비용 (바이낸스 선물 기준)
# - 메이커 수수료: 0.02%
# - 테이커 수수료: 0.04%
# - 슬리피지: 시장가 0.05% 가정

costs = {
    '바이낸스 VIP0 (테이커)': {'entry': 0.04, 'exit': 0.04, 'slip': 0.05},
    '바이낸스 VIP0 (메이커)': {'entry': 0.02, 'exit': 0.02, 'slip': 0.02},
    '업비트 (현물)': {'entry': 0.05, 'exit': 0.05, 'slip': 0.1},
    '최악의 경우': {'entry': 0.1, 'exit': 0.1, 'slip': 0.15},
}

gross_total = df['total_pnl'].sum()
n_trades = len(df)

print(f"\n기본 정보:")
print(f"  - 총 거래: {n_trades}회 (5년)")
print(f"  - 비용 전 총수익: {gross_total:.1f}%")
print(f"  - 비용 전 평균수익: {df['total_pnl'].mean():.2f}%")

print("\n" + "-"*80)
print(f"{'거래소/조건':<25} {'왕복비용':>10} {'총비용':>10} {'순수익':>10} {'연평균':>10}")
print("-"*80)

for name, cost in costs.items():
    round_trip = (cost['entry'] + cost['exit'] + cost['slip'] * 2)
    total_cost = round_trip * n_trades
    net = gross_total - total_cost
    annual = net / 5
    print(f"{name:<25} {round_trip:>9.2f}% {total_cost:>9.1f}% {net:>9.1f}% {annual:>9.1f}%")

# BTC 존버 vs 이 전략
print("\n\n" + "="*80)
print("📊 BTC 존버 vs 매매 전략 비교")
print("="*80)

# BTC 가격 변화 (대략적)
btc_changes = {
    2020: 300,   # 7000 -> 29000
    2021: 60,    # 29000 -> 47000
    2022: -65,   # 47000 -> 16500
    2023: 160,   # 16500 -> 43000
    2024: 120,   # 43000 -> 95000
    2025: 10,    # 추정
}

print(f"\n{'연도':<6} {'BTC 존버':>12} {'매매전략(비용전)':>18} {'매매전략(비용후)':>18}")
print("-"*60)

df['entry_time'] = pd.to_datetime(df['entry_time'])
df['year'] = df['entry_time'].dt.year

cost_per_trade = 0.18  # 바이낸스 테이커 기준

for year in [2020, 2021, 2022, 2023, 2024, 2025]:
    btc = btc_changes.get(year, 0)
    year_df = df[df['year'] == year]
    gross = year_df['total_pnl'].sum()
    net = gross - (len(year_df) * cost_per_trade)
    print(f"{year:<6} {btc:>11.0f}% {gross:>17.1f}% {net:>17.1f}%")

# 복리 계산
print("\n\n" + "="*80)
print("💰 복리 계산 (초기 자본 1000만원)")
print("="*80)

initial = 1000  # 만원

# 존버
btc_compound = initial
for year, change in btc_changes.items():
    btc_compound *= (1 + change/100)

# 매매 (단순 합산 - 복리 아님)
trade_total_net = gross_total - (n_trades * 0.18)
trade_compound = initial * (1 + trade_total_net/100)

# 매매 (복리 적용)
trade_compound_real = initial
for year in [2020, 2021, 2022, 2023, 2024, 2025]:
    year_df = df[df['year'] == year]
    year_return = year_df['total_pnl'].sum() - (len(year_df) * 0.18)
    trade_compound_real *= (1 + year_return/100)

print(f"\n5년 후 자산:")
print(f"  - BTC 존버: {btc_compound:,.0f}만원 ({(btc_compound/initial-1)*100:.0f}% 수익)")
print(f"  - 매매 전략 (단순): {trade_compound:,.0f}만원 ({(trade_compound/initial-1)*100:.0f}% 수익)")
print(f"  - 매매 전략 (복리): {trade_compound_real:,.0f}만원 ({(trade_compound_real/initial-1)*100:.0f}% 수익)")

# 결론
print("\n\n" + "="*80)
print("🎯 결론")
print("="*80)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                           냉정한 현실                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【비용 반영 후】 (바이낸스 테이커 기준)                              │
│    - 5년 총 수익: {trade_total_net:.1f}%                                    │
│    - 연평균: {trade_total_net/5:.1f}%                                        │
│    - 거래당 순수익: {(trade_total_net/n_trades):.2f}%                              │
│                                                                     │
│  【문제점】                                                          │
│    - 357번 거래 × 0.18% = {n_trades * 0.18:.1f}% 비용 (5년)                │
│    - 승률 52% → 동전 던지기 수준                                     │
│    - 하락장(2022)에서도 손실                                         │
│                                                                     │
│  【BTC 존버 대비】                                                   │
│    - 존버: 약 {(btc_compound/initial-1)*100:,.0f}% 수익                               │
│    - 매매: 약 {trade_total_net:.0f}% 수익                                    │
│    - 차이: 존버가 {((btc_compound/initial-1)*100 - trade_total_net):.0f}% 더 나음                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

if trade_total_net < (btc_compound/initial-1)*100:
    print("⚠️  결론: 그냥 BTC 존버가 낫습니다")
    print("\n   매매의 의미:")
    print("   - 하락장 헷지? → 2022년에도 손실")  
    print("   - 추가 수익? → 존버보다 못함")
    print("   - 시간 낭비 + 스트레스만 추가")
