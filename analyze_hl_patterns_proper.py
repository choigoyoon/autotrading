#!/usr/bin/env python3
"""
단계별 HLHLHLHL 패턴 분석 (제대로)

Step 1: L값과 H값을 모두 찾기
Step 2: HLHLHLHL 패턴 분석 (Lower High, Lower Low / Higher High, Higher Low)
Step 3: 추세선 그리기
Step 4: 브레이크아웃 감지
"""

import pandas as pd
import numpy as np
from datetime import datetime

def find_swing_points(df, left_bars=10, right_bars=10):
    """
    Step 1: Swing Low (L값)와 Swing High (H값) 찾기
    
    Parameters:
    - left_bars: 좌측 확인 캔들 수
    - right_bars: 우측 확인 캔들 수 (Look-Ahead 포함)
    
    Returns:
    - DataFrame with swing_low and swing_high columns
    """
    
    print("="*80)
    print("Step 1: Swing Low (L값)와 Swing High (H값) 찾기")
    print("="*80)
    
    df = df.copy()
    df['swing_low'] = False
    df['swing_high'] = False
    df['L_value'] = np.nan
    df['H_value'] = np.nan
    
    for i in range(left_bars, len(df) - right_bars):
        # Swing Low 확인
        current_low = df.iloc[i]['low']
        left_lows = df.iloc[i-left_bars:i]['low'].values
        right_lows = df.iloc[i+1:i+right_bars+1]['low'].values
        
        if (current_low < left_lows.min()) and (current_low < right_lows.min()):
            df.loc[df.index[i], 'swing_low'] = True
            df.loc[df.index[i], 'L_value'] = current_low
        
        # Swing High 확인
        current_high = df.iloc[i]['high']
        left_highs = df.iloc[i-left_bars:i]['high'].values
        right_highs = df.iloc[i+1:i+right_bars+1]['high'].values
        
        if (current_high > left_highs.max()) and (current_high > right_highs.max()):
            df.loc[df.index[i], 'swing_high'] = True
            df.loc[df.index[i], 'H_value'] = current_high
    
    L_count = df['swing_low'].sum()
    H_count = df['swing_high'].sum()
    
    print(f"총 Swing Low (L값) 개수: {L_count}")
    print(f"총 Swing High (H값) 개수: {H_count}")
    print()
    
    return df


def analyze_hl_patterns(df):
    """
    Step 2: HLHLHLHL 패턴 분석
    
    - Lower High, Lower Low (LL) → 하락 추세
    - Higher High, Higher Low (HH) → 상승 추세
    - 횡보 → H값과 L값이 비슷한 범위
    """
    
    print("="*80)
    print("Step 2: HLHLHLHL 패턴 분석")
    print("="*80)
    
    # L값 추출
    L_points = df[df['swing_low'] == True][['datetime', 'L_value']].copy()
    L_points = L_points.reset_index(drop=True)
    L_points['L_num'] = range(1, len(L_points) + 1)
    
    # H값 추출
    H_points = df[df['swing_high'] == True][['datetime', 'H_value']].copy()
    H_points = H_points.reset_index(drop=True)
    H_points['H_num'] = range(1, len(H_points) + 1)
    
    print(f"\n총 L값 개수: {len(L_points)}")
    print(f"총 H값 개수: {len(H_points)}")
    
    # L값 패턴 분석 (Lower Low vs Higher Low)
    print("\n" + "="*80)
    print("L값 패턴 분석 (Lower Low vs Higher Low)")
    print("="*80)
    
    L_patterns = []
    for i in range(1, len(L_points)):
        prev_L = L_points.iloc[i-1]['L_value']
        curr_L = L_points.iloc[i]['L_value']
        
        if curr_L < prev_L:
            pattern = "LL (Lower Low)"
        elif curr_L > prev_L:
            pattern = "HL (Higher Low)"
        else:
            pattern = "같음"
        
        L_patterns.append({
            'L_num': i + 1,
            'datetime': L_points.iloc[i]['datetime'],
            'prev_L': prev_L,
            'curr_L': curr_L,
            'change_pct': (curr_L - prev_L) / prev_L * 100,
            'pattern': pattern
        })
    
    L_patterns_df = pd.DataFrame(L_patterns)
    
    # H값 패턴 분석 (Lower High vs Higher High)
    print("\n" + "="*80)
    print("H값 패턴 분석 (Lower High vs Higher High)")
    print("="*80)
    
    H_patterns = []
    for i in range(1, len(H_points)):
        prev_H = H_points.iloc[i-1]['H_value']
        curr_H = H_points.iloc[i]['H_value']
        
        if curr_H < prev_H:
            pattern = "LH (Lower High)"
        elif curr_H > prev_H:
            pattern = "HH (Higher High)"
        else:
            pattern = "같음"
        
        H_patterns.append({
            'H_num': i + 1,
            'datetime': H_points.iloc[i]['datetime'],
            'prev_H': prev_H,
            'curr_H': curr_H,
            'change_pct': (curr_H - prev_H) / prev_H * 100,
            'pattern': pattern
        })
    
    H_patterns_df = pd.DataFrame(H_patterns)
    
    # 통계
    print("\nL값 패턴 통계:")
    print(L_patterns_df['pattern'].value_counts())
    print(f"\nLL (하락) 평균 변화: {L_patterns_df[L_patterns_df['pattern']=='LL (Lower Low)']['change_pct'].mean():.2f}%")
    print(f"HL (상승) 평균 변화: {L_patterns_df[L_patterns_df['pattern']=='HL (Higher Low)']['change_pct'].mean():.2f}%")
    
    print("\n" + "="*80)
    print("\nH값 패턴 통계:")
    print(H_patterns_df['pattern'].value_counts())
    print(f"\nLH (하락) 평균 변화: {H_patterns_df[H_patterns_df['pattern']=='LH (Lower High)']['change_pct'].mean():.2f}%")
    print(f"HH (상승) 평균 변화: {H_patterns_df[H_patterns_df['pattern']=='HH (Higher High)']['change_pct'].mean():.2f}%")
    
    return L_points, H_points, L_patterns_df, H_patterns_df


def detect_consolidation_with_hl(df, L_points, H_points, window=10, tolerance_pct=1.5):
    """
    Step 3: 횡보 구간 감지 (L값과 H값 기반)
    
    횡보 = L값들이 비슷한 범위에 밀집 + H값들도 비슷한 범위에 밀집
    """
    
    print("\n" + "="*80)
    print("Step 3: 횡보 구간 감지 (L값과 H값 기반)")
    print("="*80)
    
    consolidations = []
    
    # L값 기준 횡보 감지
    for i in range(window, len(L_points)):
        recent_Ls = L_points.iloc[i-window:i]['L_value'].values
        L_high = recent_Ls.max()
        L_low = recent_Ls.min()
        L_mid = (L_high + L_low) / 2
        L_range_pct = (L_high - L_low) / L_mid * 100
        
        if L_range_pct <= tolerance_pct:
            # 같은 기간 H값들 확인
            start_time = L_points.iloc[i-window]['datetime']
            end_time = L_points.iloc[i]['datetime']
            
            H_in_period = H_points[(H_points['datetime'] >= start_time) & (H_points['datetime'] <= end_time)]
            
            if len(H_in_period) >= 3:
                H_high = H_in_period['H_value'].max()
                H_low = H_in_period['H_value'].min()
                H_mid = (H_high + H_low) / 2
                H_range_pct = (H_high - H_low) / H_mid * 100
                
                if H_range_pct <= tolerance_pct * 2:  # H값은 좀 더 여유
                    consolidations.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'L_low': L_low,
                        'L_high': L_high,
                        'L_range_pct': L_range_pct,
                        'H_low': H_low,
                        'H_high': H_high,
                        'H_range_pct': H_range_pct,
                        'box_low': L_low,
                        'box_high': H_high,
                        'L_count': window,
                        'H_count': len(H_in_period)
                    })
    
    print(f"\n감지된 횡보 구간: {len(consolidations)}개")
    
    if len(consolidations) > 0:
        print("\n상위 10개 횡보 구간:")
        for i, box in enumerate(consolidations[:10], 1):
            print(f"\n횡보 #{i}")
            print(f"  기간: {box['start_time']} ~ {box['end_time']}")
            print(f"  박스 범위: ${box['box_low']:.2f} ~ ${box['box_high']:.2f}")
            print(f"  L값 범위: {box['L_range_pct']:.2f}%")
            print(f"  H값 범위: {box['H_range_pct']:.2f}%")
            print(f"  L값 개수: {box['L_count']}, H값 개수: {box['H_count']}")
    
    return consolidations


def detect_trendline_breakouts(df, H_points, L_points):
    """
    Step 4: 추세선 돌파 감지
    
    - 하락 추세선: 연속된 Lower High들을 연결
    - 상승 추세선: 연속된 Higher Low들을 연결
    - 돌파: 가격이 추세선을 뚫고 나감
    """
    
    print("\n" + "="*80)
    print("Step 4: 추세선 돌파 감지")
    print("="*80)
    
    breakouts = []
    
    # 하락 추세선 (Lower High 연결)
    print("\n하락 추세선 분석 중...")
    
    for i in range(2, len(H_points) - 5):
        # 연속된 3개의 Lower High 찾기
        H1 = H_points.iloc[i-2]['H_value']
        H2 = H_points.iloc[i-1]['H_value']
        H3 = H_points.iloc[i]['H_value']
        
        if H2 < H1 and H3 < H2:
            # 하락 추세선 형성
            time1 = H_points.iloc[i-2]['datetime']
            time2 = H_points.iloc[i]['datetime']
            
            # 이후 가격이 추세선 돌파하는지 확인
            future_data = df[df['datetime'] > time2].head(40)
            
            for j, row in future_data.iterrows():
                if row['high'] > H3:
                    # 돌파!
                    breakouts.append({
                        'type': '하락추세선 돌파 (상승)',
                        'trendline_H1': H1,
                        'trendline_H2': H2,
                        'trendline_H3': H3,
                        'breakout_time': row['datetime'],
                        'breakout_price': row['high'],
                        'breakout_pct': (row['high'] - H3) / H3 * 100
                    })
                    break
    
    # 상승 추세선 (Higher Low 연결)
    print("상승 추세선 분석 중...")
    
    for i in range(2, len(L_points) - 5):
        # 연속된 3개의 Higher Low 찾기
        L1 = L_points.iloc[i-2]['L_value']
        L2 = L_points.iloc[i-1]['L_value']
        L3 = L_points.iloc[i]['L_value']
        
        if L2 > L1 and L3 > L2:
            # 상승 추세선 형성
            time1 = L_points.iloc[i-2]['datetime']
            time2 = L_points.iloc[i]['datetime']
            
            # 이후 가격이 추세선 하향 돌파하는지 확인
            future_data = df[df['datetime'] > time2].head(40)
            
            for j, row in future_data.iterrows():
                if row['low'] < L3:
                    # 하향 돌파!
                    breakouts.append({
                        'type': '상승추세선 하향돌파 (하락)',
                        'trendline_L1': L1,
                        'trendline_L2': L2,
                        'trendline_L3': L3,
                        'breakout_time': row['datetime'],
                        'breakout_price': row['low'],
                        'breakout_pct': (row['low'] - L3) / L3 * 100
                    })
                    break
    
    print(f"\n감지된 추세선 돌파: {len(breakouts)}개")
    
    if len(breakouts) > 0:
        breakouts_df = pd.DataFrame(breakouts)
        print("\n추세선 돌파 타입별 통계:")
        print(breakouts_df['type'].value_counts())
        
        print("\n상위 10개 추세선 돌파:")
        for i, bo in enumerate(breakouts[:10], 1):
            print(f"\n돌파 #{i}")
            print(f"  타입: {bo['type']}")
            print(f"  돌파 시간: {bo['breakout_time']}")
            print(f"  돌파 가격: ${bo['breakout_price']:.2f}")
            print(f"  돌파폭: {bo['breakout_pct']:.2f}%")
    
    return breakouts


def main():
    print("="*80)
    print("단계별 HLHLHLHL 패턴 분석 (제대로)")
    print("="*80)
    
    # 데이터 로드
    df = pd.read_csv('btc_15m_ohlcv.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"\n데이터 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"총 캔들 수: {len(df):,}")
    
    # Step 1: L값과 H값 찾기
    df = find_swing_points(df, left_bars=10, right_bars=10)
    
    # Step 2: HLHLHLHL 패턴 분석
    L_points, H_points, L_patterns_df, H_patterns_df = analyze_hl_patterns(df)
    
    # Step 3: 횡보 구간 감지 (L값과 H값 기반)
    consolidations = detect_consolidation_with_hl(df, L_points, H_points, window=5, tolerance_pct=1.5)
    
    # Step 4: 추세선 돌파 감지
    breakouts = detect_trendline_breakouts(df, H_points, L_points)
    
    # 결과 저장
    L_points.to_csv('all_L_values.csv', index=False)
    H_points.to_csv('all_H_values.csv', index=False)
    L_patterns_df.to_csv('L_patterns_analysis.csv', index=False)
    H_patterns_df.to_csv('H_patterns_analysis.csv', index=False)
    
    if len(consolidations) > 0:
        pd.DataFrame(consolidations).to_csv('HL_consolidation_boxes.csv', index=False)
    
    if len(breakouts) > 0:
        pd.DataFrame(breakouts).to_csv('trendline_breakouts.csv', index=False)
    
    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80)
    print("\n생성된 파일:")
    print("  - all_L_values.csv: 모든 L값")
    print("  - all_H_values.csv: 모든 H값")
    print("  - L_patterns_analysis.csv: L값 패턴 (LL vs HL)")
    print("  - H_patterns_analysis.csv: H값 패턴 (LH vs HH)")
    print("  - HL_consolidation_boxes.csv: 횡보 구간")
    print("  - trendline_breakouts.csv: 추세선 돌파")


if __name__ == "__main__":
    main()
