#!/usr/bin/env python3
"""
LH (Lower High) 값 분석
- 가격의 고점이 낮아지는 패턴 (하락 추세의 특징)
- 저점이 낮아지는 패턴 (LL: Lower Low)
- 고점이 높아지는 패턴 (HH: Higher High)
- 저점이 높아지는 패턴 (HL: Higher Low - 상승 추세)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("LH/HL/HH/LL 값 분석")
print("=" * 60)

# 데이터 로드
df = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
print(f"\n📊 데이터 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
print(f"📊 총 15분봉 개수: {len(df):,}개")

# ============================================================
# Swing High/Low 탐지
# ============================================================

def detect_swing_points(df, left_bars=5, right_bars=5):
    """
    Swing High/Low 탐지
    - Swing High: 좌우 N개 봉보다 고가가 높은 지점
    - Swing Low: 좌우 N개 봉보다 저가가 낮은 지점
    """
    df = df.copy()
    
    df['swing_high'] = False
    df['swing_low'] = False
    df['swing_high_price'] = np.nan
    df['swing_low_price'] = np.nan
    
    for i in range(left_bars, len(df) - right_bars):
        # Swing High 체크
        is_swing_high = True
        current_high = df.iloc[i]['high']
        
        for j in range(i - left_bars, i):
            if df.iloc[j]['high'] >= current_high:
                is_swing_high = False
                break
        
        if is_swing_high:
            for j in range(i + 1, i + right_bars + 1):
                if df.iloc[j]['high'] >= current_high:
                    is_swing_high = False
                    break
        
        if is_swing_high:
            df.loc[i, 'swing_high'] = True
            df.loc[i, 'swing_high_price'] = current_high
        
        # Swing Low 체크
        is_swing_low = True
        current_low = df.iloc[i]['low']
        
        for j in range(i - left_bars, i):
            if df.iloc[j]['low'] <= current_low:
                is_swing_low = False
                break
        
        if is_swing_low:
            for j in range(i + 1, i + right_bars + 1):
                if df.iloc[j]['low'] <= current_low:
                    is_swing_low = False
                    break
        
        if is_swing_low:
            df.loc[i, 'swing_low'] = True
            df.loc[i, 'swing_low_price'] = current_low
    
    return df

print("\n🔍 Swing High/Low 탐지 중...")
df = detect_swing_points(df, left_bars=10, right_bars=10)

swing_highs = df[df['swing_high']].copy()
swing_lows = df[df['swing_low']].copy()

print(f"✅ Swing High: {len(swing_highs):,}개")
print(f"✅ Swing Low: {len(swing_lows):,}개")

# ============================================================
# LH/HL/HH/LL 패턴 탐지
# ============================================================

def classify_patterns(swing_highs, swing_lows):
    """
    연속된 Swing Point 간의 패턴 분류
    - HH (Higher High): 고점이 이전 고점보다 높음 (상승)
    - LH (Lower High): 고점이 이전 고점보다 낮음 (하락)
    - HL (Higher Low): 저점이 이전 저점보다 높음 (상승)
    - LL (Lower Low): 저점이 이전 저점보다 낮음 (하락)
    """
    
    # High 패턴
    high_patterns = []
    for i in range(1, len(swing_highs)):
        prev_high = swing_highs.iloc[i-1]['swing_high_price']
        curr_high = swing_highs.iloc[i]['swing_high_price']
        
        if curr_high > prev_high:
            pattern = 'HH'  # Higher High
        else:
            pattern = 'LH'  # Lower High
        
        high_patterns.append({
            'datetime': swing_highs.iloc[i]['datetime'],
            'pattern': pattern,
            'prev_price': prev_high,
            'curr_price': curr_high,
            'diff': curr_high - prev_high,
            'diff_pct': (curr_high - prev_high) / prev_high * 100
        })
    
    # Low 패턴
    low_patterns = []
    for i in range(1, len(swing_lows)):
        prev_low = swing_lows.iloc[i-1]['swing_low_price']
        curr_low = swing_lows.iloc[i]['swing_low_price']
        
        if curr_low > prev_low:
            pattern = 'HL'  # Higher Low
        else:
            pattern = 'LL'  # Lower Low
        
        low_patterns.append({
            'datetime': swing_lows.iloc[i]['datetime'],
            'pattern': pattern,
            'prev_price': prev_low,
            'curr_price': curr_low,
            'diff': curr_low - prev_low,
            'diff_pct': (curr_low - prev_low) / prev_low * 100
        })
    
    return pd.DataFrame(high_patterns), pd.DataFrame(low_patterns)

print("\n🔍 LH/HL/HH/LL 패턴 분류 중...")
df_high_patterns, df_low_patterns = classify_patterns(swing_highs, swing_lows)

print(f"✅ 고점 패턴: {len(df_high_patterns):,}개")
print(f"✅ 저점 패턴: {len(df_low_patterns):,}개")

# ============================================================
# 통계 분석
# ============================================================

print("\n" + "=" * 60)
print("📊 고점 패턴 분석 (HH vs LH)")
print("=" * 60)

for pattern in ['HH', 'LH']:
    subset = df_high_patterns[df_high_patterns['pattern'] == pattern]
    if len(subset) > 0:
        print(f"\n[{pattern}] Higher High" if pattern == 'HH' else f"\n[{pattern}] Lower High")
        print(f"   발생 횟수: {len(subset):,}회 ({len(subset)/len(df_high_patterns)*100:.1f}%)")
        print(f"   평균 가격 변화: {subset['diff'].mean():+,.2f} USD ({subset['diff_pct'].mean():+.2f}%)")
        print(f"   중앙값: {subset['diff'].median():+,.2f} USD ({subset['diff_pct'].median():+.2f}%)")
        print(f"   표준편차: {subset['diff'].std():,.2f} USD ({subset['diff_pct'].std():.2f}%)")
        print(f"   최대: {subset['diff'].max():+,.2f} USD ({subset['diff_pct'].max():+.2f}%)")
        print(f"   최소: {subset['diff'].min():+,.2f} USD ({subset['diff_pct'].min():+.2f}%)")

print("\n" + "=" * 60)
print("📊 저점 패턴 분석 (HL vs LL)")
print("=" * 60)

for pattern in ['HL', 'LL']:
    subset = df_low_patterns[df_low_patterns['pattern'] == pattern]
    if len(subset) > 0:
        print(f"\n[{pattern}] Higher Low" if pattern == 'HL' else f"\n[{pattern}] Lower Low")
        print(f"   발생 횟수: {len(subset):,}회 ({len(subset)/len(df_low_patterns)*100:.1f}%)")
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

# 상승 추세: HH + HL
# 하락 추세: LH + LL

hh_count = len(df_high_patterns[df_high_patterns['pattern'] == 'HH'])
lh_count = len(df_high_patterns[df_high_patterns['pattern'] == 'LH'])
hl_count = len(df_low_patterns[df_low_patterns['pattern'] == 'HL'])
ll_count = len(df_low_patterns[df_low_patterns['pattern'] == 'LL'])

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
# 파일 저장
# ============================================================

df_high_patterns.to_csv('high_patterns_analysis.csv', index=False)
df_low_patterns.to_csv('low_patterns_analysis.csv', index=False)

print("\n💾 결과 저장:")
print("   - high_patterns_analysis.csv")
print("   - low_patterns_analysis.csv")

print("\n" + "=" * 60)
