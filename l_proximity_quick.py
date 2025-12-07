"""
L값 근접도 분석 (빠른 버전 - 샘플링)
- 최근 1년 데이터만 (약 35,040개 캔들)
- L값 샘플링 (10개 중 1개)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("L값 근접도 분석 (빠른 버전)")
print("=" * 60)

# 데이터 로드 (최근 1년만)
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

cutoff_date = df['datetime'].max() - timedelta(days=365)
df = df[df['datetime'] >= cutoff_date].reset_index(drop=True)

print(f"\n데이터: {len(df):,}개 캔들 (최근 1년)")
print(f"기간: {df['datetime'].min()} ~ {df['datetime'].max()}")

# ═══════════════════════════════════════════════════════
# 지표 계산 (벡터화)
# ═══════════════════════════════════════════════════════

print("\n지표 계산 중...")

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# Bollinger Bands
df['bb_middle'] = df['close'].rolling(window=20).mean()
df['bb_std'] = df['close'].rolling(window=20).std()
df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

# FVG 감지
df['fvg'] = False
for i in range(2, len(df)):
    if df.iloc[i-2]['high'] < df.iloc[i]['low']:
        df.iloc[i, df.columns.get_loc('fvg')] = True

print("  완료!")

# ═══════════════════════════════════════════════════════
# L값 수집
# ═══════════════════════════════════════════════════════

print("\nL값 수집 중...")
l_values = []

for i in range(1, len(df)):
    prev_hist = df.iloc[i-1]['macd_hist']
    curr_hist = df.iloc[i]['macd_hist']

    if prev_hist < 0 and curr_hist >= 0:
        l_values.append(i)

print(f"  총 L값: {len(l_values)}개")

# 샘플링 (10개 중 1개)
sampled_l = l_values[::10]
print(f"  샘플링: {len(sampled_l)}개 (10개 중 1개)")

# ═══════════════════════════════════════════════════════
# L값 분석
# ═══════════════════════════════════════════════════════

print("\nL값 분석 중...")

success_count = 0
failure_count = 0

success_features = {
    'fvg': 0,
    'rsi_prox': [],
    'bb_prox': [],
    'macd_prox': []
}

failure_features = {
    'fvg': 0,
    'rsi_prox': [],
    'bb_prox': [],
    'macd_prox': []
}

for idx, l_idx in enumerate(sampled_l):
    if l_idx < 50 or l_idx + 20 >= len(df):
        continue

    # L값 시점 데이터
    l_row = df.iloc[l_idx]

    # 성공 여부 (향후 20봉 내 2% 상승)
    future_max = df.iloc[l_idx:l_idx+20]['high'].max()
    gain = (future_max - l_row['close']) / l_row['close'] * 100
    success = (gain >= 2.0)

    # 근접도 계산
    rsi_val = l_row['rsi']
    if not np.isnan(rsi_val):
        if 20 <= rsi_val <= 40:
            rsi_prox = (40 - rsi_val) / 20 * 100
        else:
            rsi_prox = 0
    else:
        rsi_prox = 0

    # BB 근접도
    bb_low = l_row['bb_lower']
    bb_mid = l_row['bb_middle']
    price = l_row['close']

    if not np.isnan(bb_low) and not np.isnan(bb_mid) and bb_mid > bb_low:
        bb_prox = (1 - (price - bb_low) / (bb_mid - bb_low)) * 100
        bb_prox = max(0, min(100, bb_prox))
    else:
        bb_prox = 0

    # MACD 0 근접도
    hist = l_row['macd_hist']
    recent_hist_max = df.iloc[max(0, l_idx-20):l_idx]['macd_hist'].abs().max()
    if recent_hist_max > 0:
        macd_prox = (1 - abs(hist) / recent_hist_max) * 100
    else:
        macd_prox = 0

    # FVG
    has_fvg = l_row['fvg']

    # 분류
    if success:
        success_count += 1
        success_features['fvg'] += (1 if has_fvg else 0)
        success_features['rsi_prox'].append(rsi_prox)
        success_features['bb_prox'].append(bb_prox)
        success_features['macd_prox'].append(macd_prox)
    else:
        failure_count += 1
        failure_features['fvg'] += (1 if has_fvg else 0)
        failure_features['rsi_prox'].append(rsi_prox)
        failure_features['bb_prox'].append(bb_prox)
        failure_features['macd_prox'].append(macd_prox)

# ═══════════════════════════════════════════════════════
# 결과 출력
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("분석 결과")
print("=" * 60)

print(f"\n성공 L값: {success_count}개 ({success_count/(success_count+failure_count)*100:.1f}%)")
print(f"실패 L값: {failure_count}개 ({failure_count/(success_count+failure_count)*100:.1f}%)")

print("\n" + "-" * 60)
print("성공 L값 특징 (20봉 내 2% 이상 상승)")
print("-" * 60)
if success_count > 0:
    print(f"  FVG 발생률: {success_features['fvg']/success_count*100:.1f}%")
    print(f"  평균 RSI 근접도: {np.mean(success_features['rsi_prox']):.1f}%")
    print(f"  평균 BB 근접도: {np.mean(success_features['bb_prox']):.1f}%")
    print(f"  평균 MACD 근접도: {np.mean(success_features['macd_prox']):.1f}%")
    avg_prox = np.mean([
        np.mean(success_features['rsi_prox']),
        np.mean(success_features['bb_prox']),
        np.mean(success_features['macd_prox'])
    ])
    print(f"  평균 전체 근접도: {avg_prox:.1f}%")

print("\n" + "-" * 60)
print("실패 L값 특징 (2% 미만)")
print("-" * 60)
if failure_count > 0:
    print(f"  FVG 발생률: {failure_features['fvg']/failure_count*100:.1f}%")
    print(f"  평균 RSI 근접도: {np.mean(failure_features['rsi_prox']):.1f}%")
    print(f"  평균 BB 근접도: {np.mean(failure_features['bb_prox']):.1f}%")
    print(f"  평균 MACD 근접도: {np.mean(failure_features['macd_prox']):.1f}%")
    avg_prox = np.mean([
        np.mean(failure_features['rsi_prox']),
        np.mean(failure_features['bb_prox']),
        np.mean(failure_features['macd_prox'])
    ])
    print(f"  평균 전체 근접도: {avg_prox:.1f}%")

if success_count > 0 and failure_count > 0:
    print("\n" + "=" * 60)
    print("차이점 (결정적 요소)")
    print("=" * 60)
    fvg_diff = (success_features['fvg']/success_count - failure_features['fvg']/failure_count) * 100
    rsi_diff = np.mean(success_features['rsi_prox']) - np.mean(failure_features['rsi_prox'])
    bb_diff = np.mean(success_features['bb_prox']) - np.mean(failure_features['bb_prox'])
    macd_diff = np.mean(success_features['macd_prox']) - np.mean(failure_features['macd_prox'])

    print(f"  FVG: {fvg_diff:+.1f}%p")
    print(f"  RSI 근접도: {rsi_diff:+.1f}%p")
    print(f"  BB 근접도: {bb_diff:+.1f}%p")
    print(f"  MACD 근접도: {macd_diff:+.1f}%p")

# ═══════════════════════════════════════════════════════
# 백테스트 (간단 버전)
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("백테스트 (근접도 80% 기준)")
print("=" * 60)

trades = []
last_trade_idx = -999

for i in range(50, len(df) - 20):

    # 간격
    if i - last_trade_idx < 10:
        continue

    row = df.iloc[i]

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
    recent_hist_max = df.iloc[max(0, i-20):i]['macd_hist'].abs().max()
    if recent_hist_max > 0:
        macd_prox = (1 - abs(hist) / recent_hist_max) * 100
    else:
        macd_prox = 0

    avg_prox = np.mean([rsi_prox, bb_prox, macd_prox])

    # 진입 조건
    if avg_prox >= 80 and row['fvg']:

        entry_price = df.iloc[i+1]['open']
        tp_price = entry_price * 1.02
        sl_price = entry_price * 0.98

        # 청산
        for j in range(i+1, min(i+50, len(df))):
            candle = df.iloc[j]

            if candle['low'] <= sl_price:
                pnl = -2.0
                break
            elif candle['high'] >= tp_price:
                pnl = 2.0
                break
        else:
            pnl = (df.iloc[min(i+50, len(df)-1)]['close'] - entry_price) / entry_price * 100

        trades.append(pnl)
        last_trade_idx = i

if len(trades) > 0:
    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100
    avg_pnl = np.mean(trades)

    print(f"\n총 거래: {len(trades)}개")
    print(f"승률: {win_rate:.1f}%")
    print(f"평균 PnL: {avg_pnl:+.2f}%")
else:
    print("\n거래 없음")

print("\n✅ 빠른 분석 완료!")
