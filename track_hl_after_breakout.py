#!/usr/bin/env python3
"""
추세 돌파 후 H/L 값 추적

1. 하락 추세선 찾기 (LH-LH-LH)
2. 추세선 돌파 시점 찾기
3. 돌파 후 H/L 움직임 추적 (재하락까지)
"""

import pandas as pd
import numpy as np

def main():
    print("="*80)
    print("추세 돌파 후 H/L 값 추적")
    print("="*80)
    
    # L값, H값 로드
    L_points = pd.read_csv('all_L_values.csv')
    L_points['datetime'] = pd.to_datetime(L_points['datetime'])
    L_points = L_points[L_points['datetime'] >= '2024-01-01'].reset_index(drop=True)
    
    H_points = pd.read_csv('all_H_values.csv')
    H_points['datetime'] = pd.to_datetime(H_points['datetime'])
    H_points = H_points[H_points['datetime'] >= '2024-01-01'].reset_index(drop=True)
    
    print(f"\nL값: {len(L_points)}개")
    print(f"H값: {len(H_points)}개")
    
    # 가격 데이터 로드
    df = pd.read_csv('btc_15m_ohlcv.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['datetime'] >= '2024-01-01'].reset_index(drop=True)
    
    print(f"캔들: {len(df):,}개\n")
    
    # 1. 하락 추세선 찾기 (연속된 3개의 LH)
    print("="*80)
    print("Step 1: 하락 추세선 찾기")
    print("="*80)
    
    trendlines = []
    
    for i in range(2, len(H_points) - 10):
        H1 = H_points.iloc[i-2]
        H2 = H_points.iloc[i-1]
        H3 = H_points.iloc[i]
        
        # LH-LH-LH (하락 추세선)
        if H2['H_value'] < H1['H_value'] and H3['H_value'] < H2['H_value']:
            trendlines.append({
                'H1_time': H1['datetime'],
                'H1_price': H1['H_value'],
                'H2_time': H2['datetime'],
                'H2_price': H2['H_value'],
                'H3_time': H3['datetime'],
                'H3_price': H3['H_value'],
                'H3_idx': i
            })
    
    print(f"\n하락 추세선 개수: {len(trendlines)}개")
    
    # 2. 추세선 돌파 찾기
    print("\n" + "="*80)
    print("Step 2: 추세선 돌파 찾기")
    print("="*80)
    
    breakouts = []
    
    for tl in trendlines[:100]:  # 최근 100개만
        H3_time = tl['H3_time']
        H3_price = tl['H3_price']
        
        # H3 이후 데이터 (30개 캔들 = 7.5시간)
        future_candles = df[df['datetime'] > H3_time].head(30)
        
        for idx, candle in future_candles.iterrows():
            # 추세선(H3) 돌파
            if candle['close'] > H3_price:
                breakouts.append({
                    'H3_time': H3_time,
                    'H3_price': H3_price,
                    'breakout_time': candle['datetime'],
                    'breakout_price': candle['close'],
                    'H3_idx': tl['H3_idx']
                })
                break
    
    print(f"\n추세선 돌파 개수: {len(breakouts)}개")
    
    if len(breakouts) == 0:
        print("\n돌파 케이스 없음")
        return
    
    # 3. 돌파 후 H/L 움직임 추적
    print("\n" + "="*80)
    print("Step 3: 돌파 후 H/L 움직임 추적")
    print("="*80)
    
    detailed_cases = []
    
    for bo in breakouts[:50]:  # 최근 50개만 상세 분석
        breakout_time = bo['breakout_time']
        H3_idx = bo['H3_idx']
        
        # 돌파 후 H값들 (다음 10개)
        after_H = H_points[(H_points['datetime'] > breakout_time)].head(10)
        
        # 돌파 후 L값들 (다음 10개)
        after_L = L_points[(L_points['datetime'] > breakout_time)].head(10)
        
        if len(after_H) < 3 or len(after_L) < 3:
            continue
        
        # HL 움직임 분석
        H_sequence = []
        L_sequence = []
        
        for i in range(len(after_H)):
            H_sequence.append({
                'datetime': after_H.iloc[i]['datetime'],
                'price': after_H.iloc[i]['H_value'],
                'order': i + 1
            })
        
        for i in range(len(after_L)):
            L_sequence.append({
                'datetime': after_L.iloc[i]['datetime'],
                'price': after_L.iloc[i]['L_value'],
                'order': i + 1
            })
        
        # 재하락 시점 찾기 (LL 발생)
        retest_failed = None
        for i in range(1, len(L_sequence)):
            if L_sequence[i]['price'] < L_sequence[i-1]['price']:
                retest_failed = {
                    'datetime': L_sequence[i]['datetime'],
                    'price': L_sequence[i]['price'],
                    'order': i + 1
                }
                break
        
        detailed_cases.append({
            'H3_time': bo['H3_time'],
            'H3_price': bo['H3_price'],
            'breakout_time': breakout_time,
            'breakout_price': bo['breakout_price'],
            'H_sequence': H_sequence,
            'L_sequence': L_sequence,
            'retest_failed': retest_failed
        })
    
    print(f"\n상세 분석 케이스: {len(detailed_cases)}개")
    
    # 4. 결과 출력 (상위 10개)
    print("\n" + "="*80)
    print("돌파 후 H/L 움직임 상세 (TOP 10)")
    print("="*80)
    
    for i, case in enumerate(detailed_cases[:10], 1):
        print(f"\n{'='*80}")
        print(f"케이스 #{i}")
        print(f"{'='*80}")
        
        print(f"\n[추세선]")
        print(f"  H3: {case['H3_time']} @ ${case['H3_price']:,.2f}")
        
        print(f"\n[돌파]")
        print(f"  돌파 시간: {case['breakout_time']}")
        print(f"  돌파 가격: ${case['breakout_price']:,.2f} (+{(case['breakout_price']-case['H3_price'])/case['H3_price']*100:.2f}%)")
        
        print(f"\n[돌파 후 H값 움직임]")
        for h in case['H_sequence']:
            print(f"  H{h['order']}: {h['datetime']} @ ${h['price']:,.2f}")
        
        # H값 패턴 분석
        if len(case['H_sequence']) >= 2:
            h_pattern = []
            for j in range(1, len(case['H_sequence'])):
                prev_h = case['H_sequence'][j-1]['price']
                curr_h = case['H_sequence'][j]['price']
                if curr_h > prev_h:
                    h_pattern.append('HH')
                else:
                    h_pattern.append('LH')
            print(f"  H값 패턴: {' → '.join(h_pattern)}")
        
        print(f"\n[돌파 후 L값 움직임]")
        for l in case['L_sequence']:
            print(f"  L{l['order']}: {l['datetime']} @ ${l['price']:,.2f}")
        
        # L값 패턴 분석
        if len(case['L_sequence']) >= 2:
            l_pattern = []
            for j in range(1, len(case['L_sequence'])):
                prev_l = case['L_sequence'][j-1]['price']
                curr_l = case['L_sequence'][j]['price']
                if curr_l > prev_l:
                    l_pattern.append('HL')
                else:
                    l_pattern.append('LL')
            print(f"  L값 패턴: {' → '.join(l_pattern)}")
        
        # 재하락 여부
        if case['retest_failed']:
            print(f"\n[재하락 발생!]")
            print(f"  시간: {case['retest_failed']['datetime']}")
            print(f"  가격: ${case['retest_failed']['price']:,.2f}")
            print(f"  순서: L{case['retest_failed']['order']}")
        else:
            print(f"\n[재하락 없음 - 상승 지속]")
    
    # 통계
    print("\n" + "="*80)
    print("통계 분석")
    print("="*80)
    
    total_cases = len(detailed_cases)
    failed_cases = sum(1 for c in detailed_cases if c['retest_failed'])
    success_cases = total_cases - failed_cases
    
    print(f"\n총 케이스: {total_cases}개")
    print(f"재하락 발생: {failed_cases}개 ({failed_cases/total_cases*100:.1f}%)")
    print(f"상승 지속: {success_cases}개 ({success_cases/total_cases*100:.1f}%)")
    
    # HL 패턴 통계
    hl_patterns = []
    hh_patterns = []
    
    for case in detailed_cases:
        if len(case['H_sequence']) >= 2:
            for j in range(1, len(case['H_sequence'])):
                prev_h = case['H_sequence'][j-1]['price']
                curr_h = case['H_sequence'][j]['price']
                if curr_h > prev_h:
                    hh_patterns.append(1)
                else:
                    hh_patterns.append(0)
        
        if len(case['L_sequence']) >= 2:
            for j in range(1, len(case['L_sequence'])):
                prev_l = case['L_sequence'][j-1]['price']
                curr_l = case['L_sequence'][j]['price']
                if curr_l > prev_l:
                    hl_patterns.append(1)
                else:
                    hl_patterns.append(0)
    
    if len(hh_patterns) > 0:
        hh_rate = sum(hh_patterns) / len(hh_patterns) * 100
        print(f"\nH값 HH(상승) 비율: {hh_rate:.1f}%")
    
    if len(hl_patterns) > 0:
        hl_rate = sum(hl_patterns) / len(hl_patterns) * 100
        print(f"L값 HL(상승) 비율: {hl_rate:.1f}%")
    
    # 저장
    results = []
    for case in detailed_cases:
        results.append({
            'H3_time': case['H3_time'],
            'H3_price': case['H3_price'],
            'breakout_time': case['breakout_time'],
            'breakout_price': case['breakout_price'],
            'retest_failed': case['retest_failed'] is not None,
            'retest_failed_time': case['retest_failed']['datetime'] if case['retest_failed'] else None,
            'retest_failed_price': case['retest_failed']['price'] if case['retest_failed'] else None
        })
    
    pd.DataFrame(results).to_csv('trendline_breakout_hl_tracking.csv', index=False)
    
    print("\n" + "="*80)
    print("✅ 결과 저장: trendline_breakout_hl_tracking.csv")
    print("="*80)


if __name__ == "__main__":
    main()
