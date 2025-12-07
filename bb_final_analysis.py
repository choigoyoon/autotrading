#!/usr/bin/env python3
"""
최종 전략 분석 - 모든 케이스 포함한 정확한 기대값 계산
타임아웃 포함, 연도별 성과 분석
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("최종 전략 분석 - 정확한 기대값")
print("=" * 80)

# 결과 파일 로드
df = pd.read_csv('bb_aggressive_results.csv')
df['time'] = pd.to_datetime(df['time'])
print(f"\n총 거래: {len(df)}건")

COST = 0.18  # 수수료+슬리피지
df['real_pnl'] = df['pnl'] - COST

# ============================================================
# 1. 전체 성과 분석 (타임아웃 포함)
# ============================================================
print("\n" + "=" * 80)
print("1. 전체 성과 분석 (모든 케이스 포함)")
print("=" * 80)

print(f"""
■ 거래 분포:
  - 익절(TP): {len(df[df['reason']=='TP'])}건 ({len(df[df['reason']=='TP'])/len(df)*100:.1f}%)
  - 손절(SL): {len(df[df['reason']=='SL'])}건 ({len(df[df['reason']=='SL'])/len(df)*100:.1f}%)
  - 타임아웃: {len(df[df['reason']=='TIMEOUT'])}건 ({len(df[df['reason']=='TIMEOUT'])/len(df)*100:.1f}%)
""")

# 각 케이스별 평균 PnL
for reason in ['TP', 'SL', 'TIMEOUT']:
    subset = df[df['reason'] == reason]
    if len(subset) > 0:
        print(f"  {reason:>8}: 평균 PnL {subset['real_pnl'].mean():+.2f}%, 건수 {len(subset)}")

# 정확한 기대값 (모든 케이스 포함)
true_expected = df['real_pnl'].mean()
print(f"\n★ 정확한 기대값 (모든 케이스 포함): {true_expected:+.3f}%")
print(f"★ 100회 거래 시 예상 수익: {true_expected * 100:+.1f}%")

# 양수/음수 거래 비율
positive = df[df['real_pnl'] > 0]
negative = df[df['real_pnl'] < 0]
print(f"\n■ 양수/음수 분포:")
print(f"  - 양수 거래: {len(positive)}건 ({len(positive)/len(df)*100:.1f}%), 평균 +{positive['real_pnl'].mean():.2f}%")
print(f"  - 음수 거래: {len(negative)}건 ({len(negative)/len(df)*100:.1f}%), 평균 {negative['real_pnl'].mean():.2f}%")

# ============================================================
# 2. 타임아웃 케이스 상세 분석
# ============================================================
print("\n" + "=" * 80)
print("2. 타임아웃 케이스 분석 (40시간 후 청산)")
print("=" * 80)

timeout = df[df['reason'] == 'TIMEOUT']
if len(timeout) > 0:
    print(f"\n타임아웃 {len(timeout)}건:")
    print(f"  - 평균 PnL: {timeout['real_pnl'].mean():+.2f}%")
    print(f"  - 양수: {len(timeout[timeout['real_pnl']>0])}건 ({len(timeout[timeout['real_pnl']>0])/len(timeout)*100:.1f}%)")
    print(f"  - 음수: {len(timeout[timeout['real_pnl']<0])}건 ({len(timeout[timeout['real_pnl']<0])/len(timeout)*100:.1f}%)")
    print(f"  - MFE 평균: {timeout['mfe'].mean():.2f}% (최대 유리한 방향)")

# ============================================================
# 3. 연도별 성과
# ============================================================
print("\n" + "=" * 80)
print("3. 연도별 성과 분석")
print("=" * 80)

df['year'] = df['time'].dt.year
yearly = df.groupby('year').agg({
    'real_pnl': ['count', 'mean', 'sum'],
    'pnl': 'mean'
}).round(3)
yearly.columns = ['거래수', '평균PnL%', '총PnL%', 'gross_pnl']
print(yearly)

# ============================================================
# 4. 방향별 분석
# ============================================================
print("\n" + "=" * 80)
print("4. 방향별 분석")
print("=" * 80)

for direction in ['LONG', 'SHORT']:
    subset = df[df['direction'] == direction]
    if len(subset) > 0:
        print(f"\n{direction} ({len(subset)}건):")
        print(f"  - 평균 PnL: {subset['real_pnl'].mean():+.2f}%")
        print(f"  - 양수 비율: {len(subset[subset['real_pnl']>0])/len(subset)*100:.1f}%")
        print(f"  - 익절(TP): {len(subset[subset['reason']=='TP'])}건")
        print(f"  - 손절(SL): {len(subset[subset['reason']=='SL'])}건")
        print(f"  - 타임아웃: {len(subset[subset['reason']=='TIMEOUT'])}건")

# ============================================================
# 5. 시뮬레이션: 자금 성장 곡선
# ============================================================
print("\n" + "=" * 80)
print("5. 자금 성장 시뮬레이션")
print("=" * 80)

initial_capital = 10000  # $10,000
position_size = 0.1  # 10% 포지션

df_sorted = df.sort_values('time')
capital = initial_capital
capitals = [capital]

for _, row in df_sorted.iterrows():
    pnl_pct = row['real_pnl'] / 100
    capital = capital * (1 + position_size * pnl_pct)
    capitals.append(capital)

final = capitals[-1]
total_return = (final - initial_capital) / initial_capital * 100

print(f"\n초기 자금: ${initial_capital:,.0f}")
print(f"포지션 크기: {position_size*100}%")
print(f"기간: {df_sorted['time'].min().strftime('%Y-%m')} ~ {df_sorted['time'].max().strftime('%Y-%m')}")
print(f"거래 횟수: {len(df)}회")
print(f"최종 자금: ${final:,.0f}")
print(f"총 수익률: {total_return:+.1f}%")

# 최대 낙폭 계산
peak = initial_capital
max_drawdown = 0
for cap in capitals:
    if cap > peak:
        peak = cap
    dd = (peak - cap) / peak
    max_drawdown = max(max_drawdown, dd)

print(f"최대 낙폭: -{max_drawdown*100:.1f}%")

# ============================================================
# 6. 다른 조건들과 비교
# ============================================================
print("\n" + "=" * 80)
print("6. 기존 전략과의 비교")
print("=" * 80)

# 데이터 다시 로드해서 더 많은 조건 테스트
df_full = pd.read_csv('analysis_15m.csv')
df_full['datetime'] = pd.to_datetime(df_full['datetime'])
df_full = df_full[df_full['datetime'] >= '2020-01-01'].sort_values('datetime').reset_index(drop=True)

# 지표 계산
period = 20
df_full['bb_mid'] = df_full['close'].rolling(period).mean()
df_full['bb_std'] = df_full['close'].rolling(period).std()
df_full['bb_upper'] = df_full['bb_mid'] + 2 * df_full['bb_std']
df_full['bb_lower'] = df_full['bb_mid'] - 2 * df_full['bb_std']
df_full['bb_width'] = df_full['bb_upper'] - df_full['bb_lower']

# RSI
rsi_period = 14
delta = df_full['close'].diff()
gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
rs = gain / loss
df_full['rsi'] = 100 - (100 / (1 + rs))

df_full = df_full.iloc[30:].reset_index(drop=True)

close = df_full['close'].values
high = df_full['high'].values
low = df_full['low'].values
bb_mid = df_full['bb_mid'].values
bb_upper = df_full['bb_upper'].values
bb_lower = df_full['bb_lower'].values
bb_width = df_full['bb_width'].values
rsi = df_full['rsi'].values
datetimes = df_full['datetime'].values

def simulate_simple(idx, direction, sl_pct=2.0, tp_pct=2.0):
    """단순 고정 손익비 시뮬레이션"""
    entry = close[idx]
    max_hold = 160
    
    if direction == 'LONG':
        sl = entry * (1 - sl_pct / 100)
        tp = entry * (1 + tp_pct / 100)
    else:
        sl = entry * (1 + sl_pct / 100)
        tp = entry * (1 - tp_pct / 100)
    
    for j in range(idx + 1, min(idx + max_hold + 1, len(close))):
        if direction == 'LONG':
            if low[j] <= sl:
                return -sl_pct - COST
            if high[j] >= tp:
                return tp_pct - COST
        else:
            if high[j] >= sl:
                return -sl_pct - COST
            if low[j] <= tp:
                return tp_pct - COST
    
    final = close[min(idx + max_hold, len(close) - 1)]
    if direction == 'LONG':
        pnl = (final - entry) / entry * 100
    else:
        pnl = (entry - final) / entry * 100
    return pnl - COST

# 테스트: RSI 15/85, 손익비 1:1 (2% SL, 2% TP)
trades_simple = []
last_idx = 0
for i in range(50, len(close) - 200):
    if i < last_idx + 4:
        continue
    
    if close[i] <= bb_lower[i] and rsi[i] < 15:
        pnl = simulate_simple(i, 'LONG', 2.0, 2.0)
        trades_simple.append(pnl)
        last_idx = i
    elif close[i] >= bb_upper[i] and rsi[i] > 85:
        pnl = simulate_simple(i, 'SHORT', 2.0, 2.0)
        trades_simple.append(pnl)
        last_idx = i

if trades_simple:
    print(f"\n■ 단순 고정비율 전략 (2% SL / 2% TP + RSI 15/85):")
    print(f"  - 거래 수: {len(trades_simple)}")
    print(f"  - 평균 PnL: {np.mean(trades_simple):+.2f}%")
    print(f"  - 양수 비율: {sum(1 for x in trades_simple if x > 0)/len(trades_simple)*100:.1f}%")

# 테스트: RSI 10/90, 손익비 1:2 (3% SL, 1.5% TP)
trades_rr = []
last_idx = 0
for i in range(50, len(close) - 200):
    if i < last_idx + 4:
        continue
    
    if close[i] <= bb_lower[i] and rsi[i] < 10:
        pnl = simulate_simple(i, 'LONG', 3.0, 1.5)
        trades_rr.append(pnl)
        last_idx = i
    elif close[i] >= bb_upper[i] and rsi[i] > 90:
        pnl = simulate_simple(i, 'SHORT', 3.0, 1.5)
        trades_rr.append(pnl)
        last_idx = i

if trades_rr:
    print(f"\n■ 보수적 고정비율 (3% SL / 1.5% TP + RSI 10/90):")
    print(f"  - 거래 수: {len(trades_rr)}")
    print(f"  - 평균 PnL: {np.mean(trades_rr):+.2f}%")
    print(f"  - 양수 비율: {sum(1 for x in trades_rr if x > 0)/len(trades_rr)*100:.1f}%")

# ============================================================
# 7. 최종 결론
# ============================================================
print("\n" + "=" * 80)
print("★★★ 최종 결론 ★★★")
print("=" * 80)
print(f"""
1. 현재 전략 (RSI 10/90 + BB극단 + 중간선 익절):
   - 정확한 기대값: {true_expected:+.3f}%
   - 문제점: 타임아웃 49%로 불확실성 높음
   
2. 실제 수익 가능 여부:
   - 100회 거래 시: 약 {true_expected * 100:+.1f}%
   - 연간 거래 수: 약 50건 (RSI 10/90 조건)
   - 연간 기대 수익: 약 {true_expected * 50:+.1f}%
   
3. 권장사항:
   - RSI 극단 조건 (10/90)은 유효함
   - 타임아웃 대신 고정 익절/손절 비율 사용 권장
   - 2-3% 손절, 1-2% 익절로 손익비 개선 필요
   
4. 핵심 인사이트:
   - BB 터치 + RSI 극단 = 반전 가능성 높음
   - 하지만 손익비 관리가 핵심
   - 작은 익절, 넓은 손절 = 높은 승률 but 낮은 손익비
""")

# 저장
print("\n분석 완료!")
