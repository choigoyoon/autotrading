"""
Phase 2: MACD Histogram 분석
======================================================================
목표: L값 근처 진입을 위한 MACD Hist 파라미터 찾기

테스트 파라미터:
1. Depth (깊이): Hist < -10, -20, -30, -50
2. Reversal (반전): Hist[t] > Hist[t-1] (Lightening Bar)
3. Divergence (다이버전스): Price LL + Hist HL

평가 지표:
- Proximity: L값까지 거리
- Win Rate: 1.5%+ 수익 확률
- Catch Rate: 얼마나 자주 발생하는가
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 70)
print("Phase 2: MACD Histogram 파라미터 분석")
print("=" * 70)
print()

# 데이터 로드
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# L값 로드 (1H Cross 기반)
try:
    df_l = pd.read_csv('l_labels_1h_cross.csv')
    df_l['datetime'] = pd.to_datetime(df_l['datetime'])
    print(f"L값 로드: {len(df_l)}개 (1H Cross 기반)")
except:
    print("⚠️  l_labels_1h_cross.csv 없음 - Phase 1 먼저 실행 필요")
    # 임시로 15min L값 사용
    l_values = []
    for i in range(1, len(df)):
        if df.iloc[i-1]['macd_hist'] < 0 and df.iloc[i]['macd_hist'] >= 0:
            l_values.append({
                'l_idx': i,
                'datetime': df.iloc[i]['datetime'],
                'price': df.iloc[i]['low']
            })
    df_l = pd.DataFrame(l_values)
    print(f"L값 생성: {len(df_l)}개 (15min MACD 기준)")

cutoff = df['datetime'].max() - timedelta(days=1825)
df = df[df['datetime'] >= cutoff].reset_index(drop=True)
df_l = df_l[df_l['datetime'] >= cutoff].reset_index(drop=True)

print(f"분석 기간: {df['datetime'].min().date()} ~ {df['datetime'].max().date()}")
print(f"L값: {len(df_l)}개\n")

# ═══════════════════════════════════════════════════════════════════
# 지표 계산
# ═══════════════════════════════════════════════════════════════════

print("지표 계산 중...")

# MACD Hist는 이미 있음
# Reversal 신호
df['hist_reversal'] = df['macd_hist'].diff() > 0  # Lightening bar

# BB %B
df['bb_middle'] = df['close'].rolling(window=20).mean()
df['bb_std'] = df['close'].rolling(window=20).std()
df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

print("  완료!\n")

# ═══════════════════════════════════════════════════════════════════
# 백테스트 함수
# ═══════════════════════════════════════════════════════════════════

def backtest_entry(entry_idx):
    """1.5% 목표 수익 달성 여부"""
    if entry_idx >= len(df):
        return None

    entry_price = df.iloc[entry_idx]['open']
    target_price = entry_price * 1.015  # 1.5% 목표

    for j in range(entry_idx, min(entry_idx + 50, len(df))):
        if df.iloc[j]['high'] >= target_price:
            return True  # Win

    return False  # Fail

def calc_proximity(entry_idx, l_idx, l_price):
    """L값까지 거리 (%)"""
    entry_price = df.iloc[entry_idx]['open']
    distance = (entry_price - l_price) / l_price * 100
    bar_distance = abs(entry_idx - l_idx)
    return distance, bar_distance

# ═══════════════════════════════════════════════════════════════════
# 파라미터 테스트
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("MACD Histogram 파라미터 테스트")
print("=" * 70)
print()

results = []

# ═══════════════════════════════════════════════════════════════════
# 테스트 1: Hist Depth (깊이)
# ═══════════════════════════════════════════════════════════════════

print("테스트 1: MACD Hist 깊이")
print()

for depth in [-10, -20, -30, -50, -100]:
    entries = []

    # 각 L값에 대해
    for _, l_row in df_l.iterrows():
        l_idx = l_row['l_idx']
        l_time = l_row['datetime']
        l_price = l_row['price']

        # L값 20봉 전부터 L값까지
        for i in range(max(0, l_idx - 20), l_idx + 1):
            if i >= len(df):
                continue

            row = df.iloc[i]

            # Hist < depth
            if row['macd_hist'] < depth:
                win = backtest_entry(i + 1)
                if win is not None:
                    price_dist, bar_dist = calc_proximity(i + 1, l_idx, l_price)
                    entries.append({
                        'win': win,
                        'price_dist': price_dist,
                        'bar_dist': bar_dist
                    })
                break  # 한 L값당 하나만

    if len(entries) > 0:
        win_rate = sum([e['win'] for e in entries]) / len(entries) * 100
        avg_price_dist = np.mean([e['price_dist'] for e in entries])
        avg_bar_dist = np.mean([e['bar_dist'] for e in entries])
        catch_rate = len(entries) / len(df_l) * 100

        results.append({
            'parameter': f'Hist < {depth}',
            'entries': len(entries),
            'catch_rate': catch_rate,
            'win_rate': win_rate,
            'avg_price_dist': avg_price_dist,
            'avg_bar_dist': avg_bar_dist
        })

        print(f"  Hist < {depth:>4}: {len(entries):>4}개 | 캐치 {catch_rate:>5.1f}% | 승률 {win_rate:>5.1f}% | L거리 {avg_price_dist:>+6.2f}% | {avg_bar_dist:>4.1f}봉")

print()

# ═══════════════════════════════════════════════════════════════════
# 테스트 2: Hist Reversal (반전)
# ═══════════════════════════════════════════════════════════════════

print("테스트 2: MACD Hist 반전 (Lightening Bar)")
print()

entries = []

for _, l_row in df_l.iterrows():
    l_idx = l_row['l_idx']
    l_price = l_row['price']

    # L값 20봉 전부터 L값까지
    for i in range(max(1, l_idx - 20), l_idx + 1):
        if i >= len(df):
            continue

        row = df.iloc[i]

        # Hist 반전 (증가 중)
        if row['hist_reversal'] and row['macd_hist'] < 0:
            win = backtest_entry(i + 1)
            if win is not None:
                price_dist, bar_dist = calc_proximity(i + 1, l_idx, l_price)
                entries.append({
                    'win': win,
                    'price_dist': price_dist,
                    'bar_dist': bar_dist
                })
            break

if len(entries) > 0:
    win_rate = sum([e['win'] for e in entries]) / len(entries) * 100
    avg_price_dist = np.mean([e['price_dist'] for e in entries])
    avg_bar_dist = np.mean([e['bar_dist'] for e in entries])
    catch_rate = len(entries) / len(df_l) * 100

    results.append({
        'parameter': 'Hist Reversal',
        'entries': len(entries),
        'catch_rate': catch_rate,
        'win_rate': win_rate,
        'avg_price_dist': avg_price_dist,
        'avg_bar_dist': avg_bar_dist
    })

    print(f"  Hist 반전: {len(entries):>4}개 | 캐치 {catch_rate:>5.1f}% | 승률 {win_rate:>5.1f}% | L거리 {avg_price_dist:>+6.2f}% | {avg_bar_dist:>4.1f}봉")

print()

# ═══════════════════════════════════════════════════════════════════
# 테스트 3: BB %B < 0.1 (참고용)
# ═══════════════════════════════════════════════════════════════════

print("테스트 3: BB %B < 0.1 (참고)")
print()

entries = []

for _, l_row in df_l.iterrows():
    l_idx = l_row['l_idx']
    l_price = l_row['price']

    for i in range(max(0, l_idx - 20), l_idx + 1):
        if i >= len(df):
            continue

        row = df.iloc[i]

        if pd.notna(row['bb_percent_b']) and row['bb_percent_b'] < 0.1:
            win = backtest_entry(i + 1)
            if win is not None:
                price_dist, bar_dist = calc_proximity(i + 1, l_idx, l_price)
                entries.append({
                    'win': win,
                    'price_dist': price_dist,
                    'bar_dist': bar_dist
                })
            break

if len(entries) > 0:
    win_rate = sum([e['win'] for e in entries]) / len(entries) * 100
    avg_price_dist = np.mean([e['price_dist'] for e in entries])
    avg_bar_dist = np.mean([e['bar_dist'] for e in entries])
    catch_rate = len(entries) / len(df_l) * 100

    results.append({
        'parameter': 'BB %B < 0.1',
        'entries': len(entries),
        'catch_rate': catch_rate,
        'win_rate': win_rate,
        'avg_price_dist': avg_price_dist,
        'avg_bar_dist': avg_bar_dist
    })

    print(f"  %B < 0.1: {len(entries):>4}개 | 캐치 {catch_rate:>5.1f}% | 승률 {win_rate:>5.1f}% | L거리 {avg_price_dist:>+6.2f}% | {avg_bar_dist:>4.1f}봉")

print()

# ═══════════════════════════════════════════════════════════════════
# 복합: Hist Depth + Reversal
# ═══════════════════════════════════════════════════════════════════

print("테스트 4: 복합 (Hist < -20 + Reversal)")
print()

entries = []

for _, l_row in df_l.iterrows():
    l_idx = l_row['l_idx']
    l_price = l_row['price']

    for i in range(max(1, l_idx - 20), l_idx + 1):
        if i >= len(df):
            continue

        row = df.iloc[i]

        if row['macd_hist'] < -20 and row['hist_reversal']:
            win = backtest_entry(i + 1)
            if win is not None:
                price_dist, bar_dist = calc_proximity(i + 1, l_idx, l_price)
                entries.append({
                    'win': win,
                    'price_dist': price_dist,
                    'bar_dist': bar_dist
                })
            break

if len(entries) > 0:
    win_rate = sum([e['win'] for e in entries]) / len(entries) * 100
    avg_price_dist = np.mean([e['price_dist'] for e in entries])
    avg_bar_dist = np.mean([e['bar_dist'] for e in entries])
    catch_rate = len(entries) / len(df_l) * 100

    results.append({
        'parameter': 'Hist<-20 + Rev',
        'entries': len(entries),
        'catch_rate': catch_rate,
        'win_rate': win_rate,
        'avg_price_dist': avg_price_dist,
        'avg_bar_dist': avg_bar_dist
    })

    print(f"  복합: {len(entries):>4}개 | 캐치 {catch_rate:>5.1f}% | 승률 {win_rate:>5.1f}% | L거리 {avg_price_dist:>+6.2f}% | {avg_bar_dist:>4.1f}봉")

print()

# ═══════════════════════════════════════════════════════════════════
# 결과 요약
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("결과 요약")
print("=" * 70)
print()

df_results = pd.DataFrame(results)
if len(df_results) > 0:
    df_results = df_results.sort_values('win_rate', ascending=False)

    print(f"{'파라미터':<20} {'진입수':<10} {'캐치율':<10} {'승률':<10} {'L거리(%)':'<12} {'L거리(봉)':<12}")
    print("-" * 70)

    for _, row in df_results.iterrows():
        print(f"{row['parameter']:<20} {row['entries']:<10} {row['catch_rate']:>6.1f}%    {row['win_rate']:>6.1f}%    {row['avg_price_dist']:>+8.2f}%   {row['avg_bar_dist']:>8.1f}봉")

    print()
    print("핵심 지표:")
    print(f"  • 캐치율: 얼마나 자주 신호가 발생하는가 (높을수록 좋음)")
    print(f"  • 승률: 1.5% 목표 달성 확률 (높을수록 좋음)")
    print(f"  • L거리: L값까지 얼마나 가까운가 (0%에 가까울수록 좋음)")

print()
print("✅ Phase 2 완료")
