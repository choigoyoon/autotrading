#!/usr/bin/env python3
"""
현실적인 수익 분석 - 수수료, 슬리피지 반영
"""

import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('L_value_backtest.csv')
df = df.drop_duplicates(subset=['entry_time'])

print("="*80)
print("💀 현실 직면: 수수료 + 슬리피지 반영 후 실제 수익")
print("="*80)

# 비용 시나리오
scenarios = [
    {'name': '이상적 (비용 0)', 'fee': 0, 'slippage': 0},
    {'name': '저비용 거래소', 'fee': 0.04, 'slippage': 0.02},  # 편도 0.04%, 슬리피지 0.02%
    {'name': '일반 거래소', 'fee': 0.1, 'slippage': 0.05},    # 편도 0.1%, 슬리피지 0.05%
    {'name': '고비용 거래소', 'fee': 0.2, 'slippage': 0.1},   # 편도 0.2%, 슬리피지 0.1%
]

print(f"\n총 거래 횟수: {len(df)}회")
print(f"평균 보유 시간: {df['hold_hours'].mean():.1f}시간")

print("\n" + "-"*80)
print(f"{'시나리오':<20} {'거래당 비용':>12} {'총 비용':>12} {'순수익':>12} {'순이익률':>12}")
print("-"*80)

gross_pnl = df['total_pnl'].sum()
n_trades = len(df)

for s in scenarios:
    # 왕복 비용 = (진입 수수료 + 청산 수수료) + (진입 슬리피지 + 청산 슬리피지)
    cost_per_trade = (s['fee'] * 2) + (s['slippage'] * 2)
    total_cost = cost_per_trade * n_trades
    net_pnl = gross_pnl - total_cost
    net_rate = net_pnl / n_trades
    
    print(f"{s['name']:<20} {cost_per_trade:>11.2f}% {total_cost:>11.1f}% {net_pnl:>11.1f}% {net_rate:>11.2f}%")

# 상세 분석
print("\n\n" + "="*80)
print("📊 거래별 손익 분포 분석")
print("="*80)

# 일반 거래소 기준 (편도 0.1% + 슬리피지 0.05%)
cost = 0.3  # 왕복 총 비용

df['net_pnl'] = df['total_pnl'] - cost
df['net_win'] = df['net_pnl'] > 0

print(f"\n【일반 거래소 기준 (왕복 0.3% 비용)】")
print(f"  - 비용 전 승률: {(df['total_pnl'] > 0).mean()*100:.1f}%")
print(f"  - 비용 후 승률: {df['net_win'].mean()*100:.1f}%")
print(f"  - 비용 전 평균수익: {df['total_pnl'].mean():.2f}%")
print(f"  - 비용 후 평균수익: {df['net_pnl'].mean():.2f}%")

# 손익 구간별 분포
print(f"\n【손익 구간별 분포】")
bins = [-10, -3, -1, 0, 0.3, 1, 2, 3, 5, 10]
labels = ['-10~-3%', '-3~-1%', '-1~0%', '0~0.3%', '0.3~1%', '1~2%', '2~3%', '3~5%', '5%+']
df['pnl_bin'] = pd.cut(df['total_pnl'], bins=bins, labels=labels)

for label in labels:
    count = (df['pnl_bin'] == label).sum()
    pct = count / len(df) * 100
    
    # 이 구간이 비용 후에도 이익인지
    if label in ['0~0.3%']:
        status = "❌ 비용으로 손실 전환"
    elif label in ['-10~-3%', '-3~-1%', '-1~0%']:
        status = "❌ 손실"
    else:
        status = "✅ 이익"
    
    print(f"  {label:>10}: {count:>3}건 ({pct:>5.1f}%) {status}")

# 비용 커버 가능한 거래만
print(f"\n【비용 커버 분석】")
profitable_after_cost = (df['total_pnl'] > cost).sum()
breakeven_zone = ((df['total_pnl'] > 0) & (df['total_pnl'] <= cost)).sum()
loss_trades = (df['total_pnl'] <= 0).sum()

print(f"  - 비용 후 이익: {profitable_after_cost}건 ({profitable_after_cost/len(df)*100:.1f}%)")
print(f"  - 비용으로 손실전환: {breakeven_zone}건 ({breakeven_zone/len(df)*100:.1f}%)")
print(f"  - 원래 손실: {loss_trades}건 ({loss_trades/len(df)*100:.1f}%)")

# 연도별 실제 수익
print("\n\n" + "="*80)
print("📅 연도별 실제 수익 (비용 반영)")
print("="*80)

df['entry_time'] = pd.to_datetime(df['entry_time'])
df['year'] = df['entry_time'].dt.year

yearly = df.groupby('year').agg({
    'total_pnl': ['count', 'sum'],
    'net_pnl': ['mean', 'sum'],
    'net_win': 'mean'
})
yearly.columns = ['거래수', '총수익(비용전)', '평균순익', '총순익', '순승률']
yearly['순승률'] = (yearly['순승률'] * 100).round(1)

print(yearly.round(2))

# 최종 결론
print("\n\n" + "="*80)
print("💡 결론")
print("="*80)

net_total = df['net_pnl'].sum()
net_avg = df['net_pnl'].mean()

print(f"""
┌────────────────────────────────────────────────────────────────────┐
│                        현실적인 수익 분석                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  【비용 전】                                                        │
│    - 총 수익: {gross_pnl:.1f}%                                        │
│    - 평균 수익: {df['total_pnl'].mean():.2f}%                                     │
│    - 승률: {(df['total_pnl'] > 0).mean()*100:.1f}%                                         │
│                                                                    │
│  【비용 후 (왕복 0.3%)】                                             │
│    - 총 수익: {net_total:.1f}%                                        │
│    - 평균 수익: {net_avg:.2f}%                                       │
│    - 승률: {df['net_win'].mean()*100:.1f}%                                         │
│                                                                    │
│  【현실】                                                           │
│    - {n_trades}번 거래 × 0.3% = {n_trades * 0.3:.1f}% 비용                     │
│    - 5년간 순수익: {net_total:.1f}%                                    │
│    - 연평균: {net_total/5:.1f}%                                          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
""")

if net_total < 50:
    print("⚠️  연평균 10% 미만 → 그냥 존버가 나을 수 있음")
if net_avg < 0.5:
    print("⚠️  거래당 평균 0.5% 미만 → 비용 대비 효율 낮음")

# 결과 저장
df.to_csv('realistic_pnl_analysis.csv', index=False)
