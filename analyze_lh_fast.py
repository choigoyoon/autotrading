#!/usr/bin/env python3
"""
LH (Lower High) 값 빠른 분석 - 벡터화 버전
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("LH/HL/HH/LL 값 분석 (Fast Version)")
print("=" * 60)

# 데이터 로드
df = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
print(f"\n📊 데이터 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
print(f"📊 총 15분봉 개수: {len(df):,}개")

# ============================================================
# Swing High/Low 탐지 (scipy 활용)
# ============================================================

print("\n🔍 Swing High/Low 탐지 중...")

order = 10  # 좌우 10개 봉 기준

# Local maxima (Swing High)
high_indices = argrelextrema(df['high'].values, np.greater, order=order)[0]
swing_highs = df.iloc[high_indices].copy()
swing_highs['swing_price'] = swing_highs['high']

# Local minima (Swing Low)
low_indices = argrelextrema(df['low'].values, np.less, order=order)[0]
swing_lows = df.iloc[low_indices].copy()
swing_lows['swing_price'] = swing_lows['low']

print(f"✅ Swing High: {len(swing_highs):,}개")
print(f"✅ Swing Low: {len(swing_lows):,}개")

# ============================================================
# LH/HH 패턴 분류 (고점)
# ============================================================

print("\n🔍 고점 패턴 분류 중...")

swing_highs = swing_highs.reset_index(drop=True)
swing_highs['prev_price'] = swing_highs['swing_price'].shift(1)
swing_highs['diff'] = swing_highs['swing_price'] - swing_highs['prev_price']
swing_highs['diff_pct'] = (swing_highs['diff'] / swing_highs['prev_price']) * 100
swing_highs['pattern'] = np.where(swing_highs['diff'] > 0, 'HH', 'LH')

# NaN 제거
swing_highs = swing_highs.dropna(subset=['prev_price'])

# ============================================================
# HL/LL 패턴 분류 (저점)
# ============================================================

print("🔍 저점 패턴 분류 중...")

swing_lows = swing_lows.reset_index(drop=True)
swing_lows['prev_price'] = swing_lows['swing_price'].shift(1)
swing_lows['diff'] = swing_lows['swing_price'] - swing_lows['prev_price']
swing_lows['diff_pct'] = (swing_lows['diff'] / swing_lows['prev_price']) * 100
swing_lows['pattern'] = np.where(swing_lows['diff'] > 0, 'HL', 'LL')

# NaN 제거
swing_lows = swing_lows.dropna(subset=['prev_price'])

print(f"✅ 고점 패턴: {len(swing_highs):,}개")
print(f"✅ 저점 패턴: {len(swing_lows):,}개")

# ============================================================
# 통계 분석
# ============================================================

print("\n" + "=" * 60)
print("📊 고점 패턴 분석 (HH vs LH)")
print("=" * 60)

for pattern in ['HH', 'LH']:
    subset = swing_highs[swing_highs['pattern'] == pattern]
    if len(subset) > 0:
        print(f"\n[{pattern}] {'Higher High' if pattern == 'HH' else 'Lower High'}")
        print(f"   발생 횟수: {len(subset):,}회 ({len(subset)/len(swing_highs)*100:.1f}%)")
        print(f"   평균 가격 변화: {subset['diff'].mean():+,.2f} USD ({subset['diff_pct'].mean():+.2f}%)")
        print(f"   중앙값: {subset['diff'].median():+,.2f} USD ({subset['diff_pct'].median():+.2f}%)")
        print(f"   표준편차: {subset['diff'].std():,.2f} USD ({subset['diff_pct'].std():.2f}%)")
        print(f"   최대: {subset['diff'].max():+,.2f} USD ({subset['diff_pct'].max():+.2f}%)")
        print(f"   최소: {subset['diff'].min():+,.2f} USD ({subset['diff_pct'].min():+.2f}%)")

print("\n" + "=" * 60)
print("📊 저점 패턴 분석 (HL vs LL)")
print("=" * 60)

for pattern in ['HL', 'LL']:
    subset = swing_lows[swing_lows['pattern'] == pattern]
    if len(subset) > 0:
        print(f"\n[{pattern}] {'Higher Low' if pattern == 'HL' else 'Lower Low'}")
        print(f"   발생 횟수: {len(subset):,}회 ({len(subset)/len(swing_lows)*100:.1f}%)")
        print(f"   평균 가격 변화: {subset['diff'].mean():+,.2f} USD ({subset['diff_pct'].mean():+.2f}%)")
        print(f"   중앙값: {subset['diff'].median():+,.2f} USD ({subset['diff_pct'].median():+.2f}%)")
        print(f"   표준편차: {subset['diff'].std():,.2f} USD ({subset['diff_pct'].std():.2f}%)")
        print(f"   최대: {subset['diff'].max():+,.2f} USD ({subset['diff_pct'].max():+.2f}%)")
        print(f"   최소: {subset['diff'].min():+,.2f} USD ({subset['diff_pct'].min():+.2f}%)")

# ============================================================
# 추세 분석
# ============================================================

print("\n" + "=" * 60)
print("📊 추세 분석")
print("=" * 60)

hh_count = len(swing_highs[swing_highs['pattern'] == 'HH'])
lh_count = len(swing_highs[swing_highs['pattern'] == 'LH'])
hl_count = len(swing_lows[swing_lows['pattern'] == 'HL'])
ll_count = len(swing_lows[swing_lows['pattern'] == 'LL'])

print(f"\n상승 추세 지표:")
print(f"   HH (Higher High): {hh_count:,}회 ({hh_count/(hh_count+lh_count)*100:.1f}%)")
print(f"   HL (Higher Low):  {hl_count:,}회 ({hl_count/(hl_count+ll_count)*100:.1f}%)")

print(f"\n하락 추세 지표:")
print(f"   LH (Lower High):  {lh_count:,}회 ({lh_count/(hh_count+lh_count)*100:.1f}%)")
print(f"   LL (Lower Low):   {ll_count:,}회 ({ll_count/(hl_count+ll_count)*100:.1f}%)")

uptrend_strength = (hh_count + hl_count) / (hh_count + lh_count + hl_count + ll_count) * 100
downtrend_strength = (lh_count + ll_count) / (hh_count + lh_count + hl_count + ll_count) * 100

print(f"\n전체 추세 비율:")
print(f"   상승 추세 강도: {uptrend_strength:.1f}%")
print(f"   하락 추세 강도: {downtrend_strength:.1f}%")

# ============================================================
# LH값의 매개변수 능력치 분석
# ============================================================

print("\n" + "=" * 60)
print("📊 LH (Lower High) 상세 능력치")
print("=" * 60)

lh_data = swing_highs[swing_highs['pattern'] == 'LH'].copy()

if len(lh_data) > 0:
    print(f"\n총 LH 발생: {len(lh_data):,}회")
    print(f"\n가격 하락 크기 분포:")
    print(f"   평균: {lh_data['diff'].mean():,.2f} USD ({lh_data['diff_pct'].mean():.2f}%)")
    print(f"   중앙값: {lh_data['diff'].median():,.2f} USD ({lh_data['diff_pct'].median():.2f}%)")
    print(f"   25 퍼센타일: {lh_data['diff'].quantile(0.25):,.2f} USD ({lh_data['diff_pct'].quantile(0.25):.2f}%)")
    print(f"   75 퍼센타일: {lh_data['diff'].quantile(0.75):,.2f} USD ({lh_data['diff_pct'].quantile(0.75):.2f}%)")
    
    print(f"\n하락 크기 구간별 분포:")
    bins = [0, -500, -1000, -2000, -5000, -10000, -float('inf')]
    labels = ['0~-500', '-500~-1000', '-1000~-2000', '-2000~-5000', '-5000~-10000', '-10000 이하']
    lh_data['bin'] = pd.cut(lh_data['diff'], bins=bins[::-1], labels=labels[::-1])
    
    for label in labels[::-1]:
        count = len(lh_data[lh_data['bin'] == label])
        if count > 0:
            pct = count / len(lh_data) * 100
            print(f"   {label:15s}: {count:4d}회 ({pct:5.1f}%)")
    
    print(f"\n하락률(%) 구간별 분포:")
    pct_bins = [0, -1, -2, -5, -10, -20, -float('inf')]
    pct_labels = ['0~-1%', '-1~-2%', '-2~-5%', '-5~-10%', '-10~-20%', '-20% 이하']
    lh_data['pct_bin'] = pd.cut(lh_data['diff_pct'], bins=pct_bins[::-1], labels=pct_labels[::-1])
    
    for label in pct_labels[::-1]:
        count = len(lh_data[lh_data['pct_bin'] == label])
        if count > 0:
            pct = count / len(lh_data) * 100
            print(f"   {label:15s}: {count:4d}회 ({pct:5.1f}%)")

# ============================================================
# 파일 저장
# ============================================================

swing_highs[['datetime', 'swing_price', 'prev_price', 'diff', 'diff_pct', 'pattern']].to_csv('high_patterns_analysis.csv', index=False)
swing_lows[['datetime', 'swing_price', 'prev_price', 'diff', 'diff_pct', 'pattern']].to_csv('low_patterns_analysis.csv', index=False)

print("\n💾 결과 저장:")
print("   - high_patterns_analysis.csv")
print("   - low_patterns_analysis.csv")

print("\n" + "=" * 60)
