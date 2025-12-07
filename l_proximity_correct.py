"""
L값 근접도 분석 (올바른 버전)
- 근접도가 높아지면 L값 가격에 가까워지는가?
- 근접도 임계값별 L값 대비 진입가 차이 측정
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("L값 근접도 분석 - 진입가 최적화")
print("=" * 60)

# 데이터 로드 (최근 2년)
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

cutoff_date = df['datetime'].max() - timedelta(days=730)
df = df[df['datetime'] >= cutoff_date].reset_index(drop=True)

print(f"\n데이터: {len(df):,}개 캔들 (최근 2년)")
print(f"기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

# 지표 계산
print("\n지표 계산 중...")
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

df['bb_middle'] = df['close'].rolling(window=20).mean()
df['bb_std'] = df['close'].rolling(window=20).std()
df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

print("  완료!")

# L값 수집
print("\nL값 수집 중...")
l_values = []

for i in range(1, len(df)):
    prev_hist = df.iloc[i-1]['macd_hist']
    curr_hist = df.iloc[i]['macd_hist']

    if prev_hist < 0 and curr_hist >= 0:
        l_values.append({
            'idx': i,
            'datetime': df.iloc[i]['datetime'],
            'price': df.iloc[i]['low']  # L값 = 저가
        })

print(f"  총 L값: {len(l_values)}개")

# 샘플링 (20개 중 1개)
sampled_l = l_values[::20]
print(f"  샘플링: {len(sampled_l)}개")

# ═══════════════════════════════════════════════════════
# 핵심 분석: 근접도와 L값 거리 상관관계
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("분석: 근접도별 L값 도달 정확도")
print("=" * 60)

results_by_threshold = {}

for threshold in [60, 65, 70, 75, 80, 85, 90]:

    distances = []  # L값 대비 진입가 차이
    bars_before = []  # L값 몇 봉 전 진입

    for l_val in sampled_l:
        l_idx = l_val['idx']
        l_price = l_val['price']

        if l_idx < 50:
            continue

        # L값 전 20봉 탐색
        for lookback in range(1, 21):
            check_idx = l_idx - lookback

            if check_idx < 50:
                break

            row = df.iloc[check_idx]

            # 근접도 계산
            rsi_val = row['rsi']
            if not np.isnan(rsi_val) and 20 <= rsi_val <= 40:
                rsi_prox = (40 - rsi_val) / 20 * 100
            else:
                rsi_prox = 0

            bb_low = row['bb_lower']
            bb_mid = row['bb_middle']
            price = row['close']

            if not np.isnan(bb_low) and not np.isnan(bb_mid) and bb_mid > bb_low:
                bb_prox = (1 - (price - bb_low) / (bb_mid - bb_low)) * 100
                bb_prox = max(0, min(100, bb_prox))
            else:
                bb_prox = 0

            hist = row['macd_hist']
            recent_hist_max = df.iloc[max(0, check_idx-20):check_idx]['macd_hist'].abs().max()
            if recent_hist_max > 0:
                macd_prox = (1 - abs(hist) / recent_hist_max) * 100
            else:
                macd_prox = 0

            avg_prox = np.mean([rsi_prox, bb_prox, macd_prox])

            # 임계값 도달 시 진입
            if avg_prox >= threshold:
                entry_price = df.iloc[check_idx + 1]['open']  # 다음 봉 시가
                distance_pct = (entry_price - l_price) / l_price * 100

                distances.append(distance_pct)
                bars_before.append(lookback)
                break  # 첫 진입만

    # 통계
    if len(distances) > 0:
        results_by_threshold[threshold] = {
            'count': len(distances),
            'avg_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'avg_bars_before': np.mean(bars_before),
            'within_05pct': len([d for d in distances if d <= 0.5]) / len(distances) * 100,
            'within_10pct': len([d for d in distances if d <= 1.0]) / len(distances) * 100
        }

# 결과 출력
print("\n임계값별 L값 근접도")
print("-" * 60)
print(f"{'임계값':<8} {'진입수':<8} {'평균차이':<12} {'중간값':<12} {'0.5%이내':<12} {'평균봉수':<8}")
print("-" * 60)

for threshold in sorted(results_by_threshold.keys()):
    r = results_by_threshold[threshold]
    print(f"{threshold}%      {r['count']:<8} {r['avg_distance']:+.3f}%      {r['median_distance']:+.3f}%      {r['within_05pct']:.1f}%        {r['avg_bars_before']:.1f}봉")

# ═══════════════════════════════════════════════════════
# 상세 분석: 근접도 추세 패턴
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("상세: 근접도 변화 패턴 (샘플 10개)")
print("=" * 60)

for sample_idx in range(min(10, len(sampled_l))):
    l_val = sampled_l[sample_idx]
    l_idx = l_val['idx']
    l_price = l_val['price']

    if l_idx < 50:
        continue

    print(f"\nL값 #{sample_idx + 1}: {l_val['datetime']}, ${l_price:,.2f}")
    print(f"{'봉수':<8} {'가격':<12} {'L값차이':<12} {'근접도':<10}")
    print("-" * 50)

    for lookback in range(10, 0, -1):
        check_idx = l_idx - lookback

        if check_idx < 50:
            continue

        row = df.iloc[check_idx]
        price = row['close']

        # 근접도
        rsi_val = row['rsi']
        if not np.isnan(rsi_val) and 20 <= rsi_val <= 40:
            rsi_prox = (40 - rsi_val) / 20 * 100
        else:
            rsi_prox = 0

        bb_low = row['bb_lower']
        bb_mid = row['bb_middle']

        if not np.isnan(bb_low) and not np.isnan(bb_mid) and bb_mid > bb_low:
            bb_prox = (1 - (price - bb_low) / (bb_mid - bb_low)) * 100
            bb_prox = max(0, min(100, bb_prox))
        else:
            bb_prox = 0

        hist = row['macd_hist']
        recent_hist_max = df.iloc[max(0, check_idx-20):check_idx]['macd_hist'].abs().max()
        if recent_hist_max > 0:
            macd_prox = (1 - abs(hist) / recent_hist_max) * 100
        else:
            macd_prox = 0

        avg_prox = np.mean([rsi_prox, bb_prox, macd_prox])

        distance = (price - l_price) / l_price * 100

        print(f"-{lookback}봉     ${price:>10,.2f}  {distance:+.3f}%      {avg_prox:.1f}%")

# ═══════════════════════════════════════════════════════
# 최적 임계값 추천
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("최적 임계값 추천")
print("=" * 60)

best_threshold = None
best_score = -999

for threshold, r in results_by_threshold.items():
    # 점수 = 0.5% 이내 비율 - 평균 거리
    score = r['within_05pct'] - abs(r['avg_distance']) * 10

    if score > best_score:
        best_score = score
        best_threshold = threshold

if best_threshold:
    r = results_by_threshold[best_threshold]
    print(f"\n✅ 추천 임계값: {best_threshold}%")
    print(f"   평균 L값 차이: {r['avg_distance']:+.3f}%")
    print(f"   0.5% 이내: {r['within_05pct']:.1f}%")
    print(f"   평균 {r['avg_bars_before']:.1f}봉 전 진입")
    print(f"   = 약 {r['avg_bars_before'] * 15:.0f}분 전")

print("\n✅ 분석 완료!")
