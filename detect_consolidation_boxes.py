#!/usr/bin/env python3
"""
박스권 (Consolidation Box) 감지 알고리즘
사용자 차트 기반 분석: 하락 -> 횡보 -> 하락 -> 횡보 패턴
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def detect_consolidation_boxes(df, lookback=20, price_tolerance=0.015, min_duration=8):
    """
    박스권 감지 알고리즘
    
    Parameters:
    - lookback: 박스권 판단을 위한 캔들 수
    - price_tolerance: 가격 범위 허용 오차 (1.5% = 0.015)
    - min_duration: 최소 박스권 지속 기간 (캔들 수)
    
    Returns:
    - boxes: 감지된 박스권 리스트
    """
    
    boxes = []
    in_box = False
    box_start = None
    box_high = None
    box_low = None
    
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]
        
        # 현재 윈도우의 가격 범위 계산
        window_high = window['high'].max()
        window_low = window['low'].min()
        window_range = window_high - window_low
        window_mid = (window_high + window_low) / 2
        
        # 가격 변동폭이 작으면 박스권으로 판단
        range_pct = window_range / window_mid
        
        if range_pct <= price_tolerance:
            # 박스권 시작
            if not in_box:
                in_box = True
                box_start = i - lookback
                box_high = window_high
                box_low = window_low
                box_candles = lookback
            else:
                # 기존 박스권 확장
                box_high = max(box_high, window_high)
                box_low = min(box_low, window_low)
                box_candles += 1
        else:
            # 박스권 종료
            if in_box and box_candles >= min_duration:
                box_end = i - 1
                boxes.append({
                    'start_idx': box_start,
                    'end_idx': box_end,
                    'start_time': df.iloc[box_start]['datetime'],
                    'end_time': df.iloc[box_end]['datetime'],
                    'box_high': box_high,
                    'box_low': box_low,
                    'box_mid': (box_high + box_low) / 2,
                    'box_range_pct': (box_high - box_low) / ((box_high + box_low) / 2) * 100,
                    'duration_candles': box_candles,
                    'duration_hours': box_candles * 0.25  # 15분 캔들
                })
            
            in_box = False
            box_start = None
            box_high = None
            box_low = None
            box_candles = 0
    
    return boxes


def detect_l_values_with_boxes(df):
    """
    L-value와 박스권을 함께 분석
    
    박스권 이후 L-value (Swing Low) 발생 시 잠재적 진입점으로 판단
    """
    
    # 1. 박스권 감지
    boxes = detect_consolidation_boxes(df)
    
    print(f"\n{'='*80}")
    print(f"📦 감지된 박스권: {len(boxes)}개")
    print(f"{'='*80}\n")
    
    for i, box in enumerate(boxes, 1):
        print(f"박스권 #{i}")
        print(f"  기간: {box['start_time']} ~ {box['end_time']}")
        print(f"  고점: {box['box_high']:.2f}")
        print(f"  저점: {box['box_low']:.2f}")
        print(f"  중심: {box['box_mid']:.2f}")
        print(f"  범위: {box['box_range_pct']:.2f}%")
        print(f"  지속: {box['duration_candles']} 캔들 ({box['duration_hours']:.1f} 시간)")
        print()
    
    # 2. 각 박스권 이후 L-value 찾기
    results = []
    
    for box in boxes:
        # 박스권 종료 이후 데이터
        after_box = df.iloc[box['end_idx']+1:box['end_idx']+41]  # 이후 40개 캔들 (10시간)
        
        if len(after_box) < 10:
            continue
        
        # Swing Low 감지 (좌우 5개 캔들 확인)
        for j in range(5, len(after_box)-5):
            current_idx = box['end_idx'] + 1 + j
            current_low = after_box.iloc[j]['low']
            
            # 좌우 5개 캔들보다 낮은지 확인
            left_lows = after_box.iloc[j-5:j]['low'].values
            right_lows = after_box.iloc[j+1:j+6]['low'].values
            
            if (current_low < left_lows.min()) and (current_low < right_lows.min()):
                # L-value 발견!
                l_value_price = current_low
                l_value_time = after_box.iloc[j]['datetime']
                
                # 박스권 하단 대비 하락폭
                drop_from_box = (box['box_low'] - l_value_price) / box['box_low'] * 100
                
                # L-value 이후 최대 상승폭 (향후 40개 캔들 확인)
                future_data = df.iloc[current_idx+1:current_idx+41]
                if len(future_data) > 0:
                    max_gain = (future_data['high'].max() - l_value_price) / l_value_price * 100
                else:
                    max_gain = 0
                
                results.append({
                    'box_num': boxes.index(box) + 1,
                    'box_end_time': box['end_time'],
                    'box_high': box['box_high'],
                    'box_low': box['box_low'],
                    'box_range_pct': box['box_range_pct'],
                    'l_value_time': l_value_time,
                    'l_value_price': l_value_price,
                    'drop_from_box_pct': drop_from_box,
                    'max_gain_after_l_pct': max_gain,
                    'bars_after_box': j
                })
                
                break  # 첫 번째 L-value만 체크
    
    # 3. 결과 DataFrame 생성
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        print(f"\n{'='*80}")
        print(f"📊 박스권 이후 L-value 분석: {len(results_df)}개")
        print(f"{'='*80}\n")
        
        print(f"평균 박스권 이후 하락폭: {results_df['drop_from_box_pct'].mean():.2f}%")
        print(f"평균 L-value 이후 최대 상승: {results_df['max_gain_after_l_pct'].mean():.2f}%")
        print(f"박스권 종료 후 L-value 발생까지 평균 시간: {results_df['bars_after_box'].mean() * 0.25:.1f}시간")
        
        # 성공 케이스 (최대 상승 > 2%)
        success = results_df[results_df['max_gain_after_l_pct'] > 2.0]
        print(f"\n성공 케이스 (최대 상승 > 2%): {len(success)}/{len(results_df)} ({len(success)/len(results_df)*100:.1f}%)")
        
        if len(success) > 0:
            print(f"  평균 상승: {success['max_gain_after_l_pct'].mean():.2f}%")
            print(f"  평균 하락폭: {success['drop_from_box_pct'].mean():.2f}%")
        
        # 상위 10개 케이스 출력
        print(f"\n{'='*80}")
        print(f"🏆 TOP 10 성공 케이스")
        print(f"{'='*80}\n")
        
        top10 = results_df.nlargest(10, 'max_gain_after_l_pct')
        for idx, row in top10.iterrows():
            print(f"박스권 #{int(row['box_num'])} (종료: {row['box_end_time']})")
            print(f"  박스권: {row['box_low']:.2f} ~ {row['box_high']:.2f} (범위: {row['box_range_pct']:.2f}%)")
            print(f"  L-value: {row['l_value_time']} @ ${row['l_value_price']:.2f}")
            print(f"  박스 하단 대비 하락: -{row['drop_from_box_pct']:.2f}%")
            print(f"  이후 최대 상승: +{row['max_gain_after_l_pct']:.2f}%")
            print()
        
        # CSV 저장
        results_df.to_csv('consolidation_box_analysis.csv', index=False)
        print(f"\n✅ 결과 저장: consolidation_box_analysis.csv")
    
    return boxes, results_df


def main():
    print("="*80)
    print("📦 박스권 (Consolidation Box) 감지 및 L-value 분석")
    print("="*80)
    
    # 데이터 로드
    df = pd.read_csv('btc_15m_ohlcv.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"\n데이터 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"총 캔들 수: {len(df):,}")
    
    # 박스권 및 L-value 분석
    boxes, results_df = detect_l_values_with_boxes(df)
    
    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80)


if __name__ == "__main__":
    main()
