#!/usr/bin/env python3
"""
추세선 내려오면서 생긴 H/L값의 가로선(수평선) 분석

핵심: 
- 하락 추세선 (LH-LH-LH)
- 각 H값에서 떨어진 L값들
- 이 L값들이 만드는 가로선 = 지지선/저항선
"""

import pandas as pd
import numpy as np

def main():
    print("="*80)
    print("추세선 H/L값의 가로선 (수평선) 분석")
    print("="*80)
    
    # 데이터 로드
    L_points = pd.read_csv('all_L_values.csv')
    L_points['datetime'] = pd.to_datetime(L_points['datetime'])
    L_points = L_points[L_points['datetime'] >= '2024-01-01'].reset_index(drop=True)
    
    H_points = pd.read_csv('all_H_values.csv')
    H_points['datetime'] = pd.to_datetime(H_points['datetime'])
    H_points = H_points[H_points['datetime'] >= '2024-01-01'].reset_index(drop=True)
    
    df = pd.read_csv('btc_15m_ohlcv.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['datetime'] >= '2024-01-01'].reset_index(drop=True)
    
    print(f"\nL값: {len(L_points)}개")
    print(f"H값: {len(H_points)}개")
    
    # 1. 하락 추세선 찾기
    print("\n" + "="*80)
    print("Step 1: 하락 추세선 찾기")
    print("="*80)
    
    trendlines = []
    
    for i in range(2, len(H_points) - 10):
        H1 = H_points.iloc[i-2]
        H2 = H_points.iloc[i-1]
        H3 = H_points.iloc[i]
        
        # LH-LH-LH
        if H2['H_value'] < H1['H_value'] and H3['H_value'] < H2['H_value']:
            
            # H1-H3 구간의 L값들 찾기
            L_between = L_points[
                (L_points['datetime'] >= H1['datetime']) & 
                (L_points['datetime'] <= H3['datetime'])
            ]
            
            if len(L_between) >= 3:  # 최소 3개 L값
                trendlines.append({
                    'H1': H1,
                    'H2': H2,
                    'H3': H3,
                    'L_values': L_between
                })
    
    print(f"\n하락 추세선 개수: {len(trendlines)}개")
    
    # 2. 각 추세선의 L값 가로선 분석
    print("\n" + "="*80)
    print("Step 2: 추세선 내 L값 가로선 분석")
    print("="*80)
    
    detailed_analysis = []
    
    for idx, tl in enumerate(trendlines[:20], 1):  # 최근 20개
        H1 = tl['H1']
        H2 = tl['H2']
        H3 = tl['H3']
        L_vals = tl['L_values']
        
        print(f"\n{'='*80}")
        print(f"추세선 #{idx}")
        print(f"{'='*80}")
        
        print(f"\n[하락 추세선 고점들]")
        print(f"H1: {H1['datetime']} @ ${H1['H_value']:,.2f}")
        print(f"H2: {H2['datetime']} @ ${H2['H_value']:,.2f} (LH: {(H2['H_value']-H1['H_value'])/H1['H_value']*100:+.2f}%)")
        print(f"H3: {H3['datetime']} @ ${H3['H_value']:,.2f} (LH: {(H3['H_value']-H2['H_value'])/H2['H_value']*100:+.2f}%)")
        
        print(f"\n[추세선 내 L값들 (가로선)]")
        for i, row in L_vals.iterrows():
            print(f"L{row['L_num']}: {row['datetime']} @ ${row['L_value']:,.2f}")
        
        # 각 L값을 가로선(수평 지지선)으로 간주
        # 나중에 이 L값 근처에서 반등/저항이 있는지 확인
        
        # H3 이후 가격 움직임 확인 (30개 캔들 = 7.5시간)
        future_data = df[df['datetime'] > H3['datetime']].head(40)
        
        if len(future_data) > 0:
            print(f"\n[추세선 이후 가격 움직임]")
            
            # 각 L값 근처에서 지지/저항 확인
            for i, l_row in L_vals.iterrows():
                l_price = l_row['L_value']
                l_datetime = l_row['datetime']
                
                # 이 L값 근처 (±1%) 터치 확인
                touches = []
                
                for _, candle in future_data.iterrows():
                    # 가격이 L값 근처에 왔는지 확인
                    if abs(candle['low'] - l_price) / l_price < 0.01:  # 1% 이내
                        touches.append({
                            'datetime': candle['datetime'],
                            'low': candle['low'],
                            'close': candle['close'],
                            'bounced': candle['close'] > candle['open']  # 양봉
                        })
                
                if len(touches) > 0:
                    print(f"\n  L값 ${l_price:,.2f} 근처 터치:")
                    for touch in touches[:3]:  # 최대 3개
                        bounce_status = "✅ 반등" if touch['bounced'] else "🔴 하락"
                        print(f"    {touch['datetime']}: ${touch['low']:,.2f} {bounce_status}")
        
        # 분석 저장
        detailed_analysis.append({
            'trendline_num': idx,
            'H1_time': H1['datetime'],
            'H1_price': H1['H_value'],
            'H2_time': H2['datetime'],
            'H2_price': H2['H_value'],
            'H3_time': H3['datetime'],
            'H3_price': H3['H_value'],
            'L_count': len(L_vals),
            'L_prices': L_vals['L_value'].tolist()
        })
    
    # 3. 가로선(수평선) 역할 분석
    print("\n" + "="*80)
    print("Step 3: L값 가로선의 역할")
    print("="*80)
    
    print("""
    핵심 개념:
    
    추세선이 내려오면서:
    H1 → L1 (가로선 1번)
    H2 → L2 (가로선 2번)
    H3 → L3 (가로선 3번)
    
    이 L값들이 나중에:
    1. 지지선 역할 = 가격이 다시 내려왔을 때 반등
    2. 저항선 역할 = 가격이 올라갔을 때 막힘
    3. 리테스트 = 추세 돌파 후 다시 L값 근처로 되돌아옴
    
    매매 포인트:
    - L값 근처 터치 + 반등 = Long 진입
    - L값 깨짐 = 추가 하락 가능성
    - 추세 돌파 후 L값 리테스트 성공 = Long 확신
    """)
    
    # 통계
    print("\n" + "="*80)
    print("통계")
    print("="*80)
    
    total_L_count = sum(d['L_count'] for d in detailed_analysis)
    avg_L_count = total_L_count / len(detailed_analysis) if len(detailed_analysis) > 0 else 0
    
    print(f"\n분석한 추세선: {len(detailed_analysis)}개")
    print(f"총 L값 개수: {total_L_count}개")
    print(f"추세선당 평균 L값: {avg_L_count:.1f}개")
    
    # 저장
    pd.DataFrame(detailed_analysis).to_csv('trendline_horizontal_levels.csv', index=False)
    
    print("\n" + "="*80)
    print("✅ 결과 저장: trendline_horizontal_levels.csv")
    print("="*80)


if __name__ == "__main__":
    main()
