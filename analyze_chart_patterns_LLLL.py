#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
차트 패턴 분석: LLLL (4번 연속 Lower Low) 패턴 집중 분석
연속 하락 패턴의 차트 구조 및 반등 가능성 분석
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_indicators(df):
    """지표 계산"""
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - signal
    
    # Bollinger Bands
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['ma20'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['ma20'] - (df['bb_std'] * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    # ATR
    df['h_l'] = df['high'] - df['low']
    df['h_pc'] = abs(df['high'] - df['close'].shift(1))
    df['l_pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # Stochastic
    df['lowest_14'] = df['low'].rolling(window=14).min()
    df['highest_14'] = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - df['lowest_14']) / (df['highest_14'] - df['lowest_14'])
    
    # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - sma) / (0.015 * mad)
    
    return df


def detect_swing_lows(df, left_bars=10, right_bars=10):
    """Swing Low (L값) 감지"""
    swing_lows = []
    
    for i in range(left_bars, len(df) - right_bars):
        current_low = df.iloc[i]['low']
        
        # 왼쪽 10봉 체크
        left_higher = all(df.iloc[i - j]['low'] > current_low for j in range(1, left_bars + 1))
        
        # 오른쪽 10봉 체크
        right_higher = all(df.iloc[i + j]['low'] > current_low for j in range(1, right_bars + 1))
        
        if left_higher and right_higher:
            swing_lows.append({
                'index': i,
                'datetime': df.iloc[i]['datetime'],
                'price': current_low,
                'confirmed_index': i + right_bars
            })
    
    return swing_lows


def classify_swing_low_pattern(swing_lows):
    """LL/HL 패턴 분류 및 연속 LL 카운트"""
    for i, sl in enumerate(swing_lows):
        if i == 0:
            sl['pattern'] = 'HL'
            sl['consecutive_ll'] = 0
        else:
            prev_price = swing_lows[i - 1]['price']
            if sl['price'] < prev_price:
                sl['pattern'] = 'LL'
                # 연속 LL 카운트
                count = 1
                for j in range(i - 1, -1, -1):
                    if swing_lows[j]['pattern'] == 'LL':
                        count += 1
                    else:
                        break
                sl['consecutive_ll'] = count
            else:
                sl['pattern'] = 'HL'
                sl['consecutive_ll'] = 0
    
    return swing_lows


def analyze_LLLL_chart_structure(df, swing_lows):
    """LLLL (4번 연속 LL) 차트 구조 분석"""
    print("=" * 80)
    print("차트 패턴 분석: LLLL (4번 연속 Lower Low)")
    print("=" * 80)
    
    # 4번 연속 LL 필터링
    llll_patterns = [sl for sl in swing_lows if sl['consecutive_ll'] == 4]
    
    print(f"\n총 LLLL 패턴 발견: {len(llll_patterns)}회")
    
    if len(llll_patterns) == 0:
        print("LLLL 패턴이 발견되지 않았습니다.")
        return []
    
    # 각 LLLL 패턴 상세 분석
    pattern_details = []
    
    for idx, llll in enumerate(llll_patterns, 1):
        # 현재 L4 (4번째 LL)
        l4_idx = llll['index']
        l4_price = llll['price']
        l4_time = llll['datetime']
        
        # 이전 L1, L2, L3 찾기
        prev_lows = []
        for sl in reversed(swing_lows[:swing_lows.index(llll)]):
            if sl['pattern'] == 'LL':
                prev_lows.insert(0, sl)
                if len(prev_lows) == 3:
                    break
        
        if len(prev_lows) < 3:
            continue
        
        l1, l2, l3 = prev_lows
        
        # 가격 하락폭 계산
        l1_to_l2 = (l2['price'] - l1['price']) / l1['price'] * 100
        l2_to_l3 = (l3['price'] - l2['price']) / l2['price'] * 100
        l3_to_l4 = (l4_price - l3['price']) / l3['price'] * 100
        total_drop = (l4_price - l1['price']) / l1['price'] * 100
        
        # 시간 간격
        l1_to_l2_hours = (l2['datetime'] - l1['datetime']).total_seconds() / 3600
        l2_to_l3_hours = (l3['datetime'] - l2['datetime']).total_seconds() / 3600
        l3_to_l4_hours = (l4_time - l3['datetime']).total_seconds() / 3600
        total_duration_hours = (l4_time - l1['datetime']).total_seconds() / 3600
        
        # L4 지점의 지표 값
        l4_indicators = df.iloc[l4_idx]
        
        # L4 이후 반등 추적 (20봉, 40봉, 60봉)
        max_gain_20 = 0
        max_gain_40 = 0
        max_gain_60 = 0
        
        if l4_idx + 20 < len(df):
            next_20 = df.iloc[l4_idx:l4_idx+20]
            max_gain_20 = (next_20['high'].max() - l4_price) / l4_price * 100
        
        if l4_idx + 40 < len(df):
            next_40 = df.iloc[l4_idx:l4_idx+40]
            max_gain_40 = (next_40['high'].max() - l4_price) / l4_price * 100
        
        if l4_idx + 60 < len(df):
            next_60 = df.iloc[l4_idx:l4_idx+60]
            max_gain_60 = (next_60['high'].max() - l4_price) / l4_price * 100
        
        # 최종 반등 (다음 HL까지)
        next_hl_gain = None
        for sl in swing_lows[swing_lows.index(llll) + 1:]:
            if sl['pattern'] == 'HL':
                next_hl_gain = (sl['price'] - l4_price) / l4_price * 100
                break
        
        pattern_details.append({
            'pattern_num': idx,
            'l4_datetime': l4_time,
            'l1_price': l1['price'],
            'l2_price': l2['price'],
            'l3_price': l3['price'],
            'l4_price': l4_price,
            'l1_to_l2_pct': l1_to_l2,
            'l2_to_l3_pct': l2_to_l3,
            'l3_to_l4_pct': l3_to_l4,
            'total_drop_pct': total_drop,
            'l1_to_l2_hours': l1_to_l2_hours,
            'l2_to_l3_hours': l2_to_l3_hours,
            'l3_to_l4_hours': l3_to_l4_hours,
            'total_duration_hours': total_duration_hours,
            'l4_rsi': l4_indicators['rsi'],
            'l4_macd_hist': l4_indicators['macd_hist'],
            'l4_bb_position': l4_indicators['bb_position'],
            'l4_volume_ratio': l4_indicators['volume_ratio'],
            'l4_atr_pct': l4_indicators['atr_pct'],
            'l4_stoch_k': l4_indicators['stoch_k'],
            'l4_cci': l4_indicators['cci'],
            'max_gain_20bars': max_gain_20,
            'max_gain_40bars': max_gain_40,
            'max_gain_60bars': max_gain_60,
            'next_hl_gain_pct': next_hl_gain if next_hl_gain else 0
        })
    
    return pattern_details


def print_pattern_statistics(pattern_details):
    """패턴 통계 출력"""
    if len(pattern_details) == 0:
        return
    
    df = pd.DataFrame(pattern_details)
    
    print("\n" + "=" * 80)
    print("LLLL 패턴 통계 분석")
    print("=" * 80)
    
    # 가격 하락폭 통계
    print(f"\n[1] 가격 하락폭 분석")
    print(f"\n  단계별 하락:")
    print(f"    L1 → L2: 평균 {df['l1_to_l2_pct'].mean():.2f}% (중앙값 {df['l1_to_l2_pct'].median():.2f}%)")
    print(f"    L2 → L3: 평균 {df['l2_to_l3_pct'].mean():.2f}% (중앙값 {df['l2_to_l3_pct'].median():.2f}%)")
    print(f"    L3 → L4: 평균 {df['l3_to_l4_pct'].mean():.2f}% (중앙값 {df['l3_to_l4_pct'].median():.2f}%)")
    print(f"\n  전체 하락:")
    print(f"    L1 → L4: 평균 {df['total_drop_pct'].mean():.2f}% (중앙값 {df['total_drop_pct'].median():.2f}%)")
    print(f"    최대 하락: {df['total_drop_pct'].min():.2f}%")
    print(f"    최소 하락: {df['total_drop_pct'].max():.2f}%")
    
    # 시간 간격 통계
    print(f"\n[2] 시간 간격 분석")
    print(f"\n  단계별 소요 시간:")
    print(f"    L1 → L2: 평균 {df['l1_to_l2_hours'].mean():.1f}시간 (중앙값 {df['l1_to_l2_hours'].median():.1f}시간)")
    print(f"    L2 → L3: 평균 {df['l2_to_l3_hours'].mean():.1f}시간 (중앙값 {df['l2_to_l3_hours'].median():.1f}시간)")
    print(f"    L3 → L4: 평균 {df['l3_to_l4_hours'].mean():.1f}시간 (중앙값 {df['l3_to_l4_hours'].median():.1f}시간)")
    print(f"\n  전체 소요 시간:")
    print(f"    L1 → L4: 평균 {df['total_duration_hours'].mean():.1f}시간 (중앙값 {df['total_duration_hours'].median():.1f}시간)")
    print(f"    평균 일수: {df['total_duration_hours'].mean() / 24:.1f}일")
    
    # L4 지점 지표 통계
    print(f"\n[3] L4 지점 지표 분석")
    print(f"\n  평균 지표 값:")
    print(f"    RSI: {df['l4_rsi'].mean():.2f}")
    print(f"    MACD Histogram: {df['l4_macd_hist'].mean():.2f}")
    print(f"    BB Position: {df['l4_bb_position'].mean():.2f}")
    print(f"    Volume Ratio: {df['l4_volume_ratio'].mean():.2f}x")
    print(f"    ATR %: {df['l4_atr_pct'].mean():.2f}%")
    print(f"    Stochastic K: {df['l4_stoch_k'].mean():.2f}")
    print(f"    CCI: {df['l4_cci'].mean():.2f}")
    
    # 극과매도 조건 충족률
    print(f"\n  극과매도 조건 충족률:")
    print(f"    RSI < 30: {len(df[df['l4_rsi'] < 30]) / len(df) * 100:.1f}%")
    print(f"    MACD Hist < -50: {len(df[df['l4_macd_hist'] < -50]) / len(df) * 100:.1f}%")
    print(f"    BB Position < 0.1: {len(df[df['l4_bb_position'] < 0.1]) / len(df) * 100:.1f}%")
    print(f"    Volume Ratio > 3.0: {len(df[df['l4_volume_ratio'] > 3.0]) / len(df) * 100:.1f}%")
    print(f"    ATR % > 0.5: {len(df[df['l4_atr_pct'] > 0.5]) / len(df) * 100:.1f}%")
    
    # 반등 성과
    print(f"\n[4] L4 이후 반등 성과")
    print(f"\n  시간별 최대 반등:")
    print(f"    20봉 내 (5시간): 평균 {df['max_gain_20bars'].mean():.2f}% (중앙값 {df['max_gain_20bars'].median():.2f}%)")
    print(f"    40봉 내 (10시간): 평균 {df['max_gain_40bars'].mean():.2f}% (중앙값 {df['max_gain_40bars'].median():.2f}%)")
    print(f"    60봉 내 (15시간): 평균 {df['max_gain_60bars'].mean():.2f}% (중앙값 {df['max_gain_60bars'].median():.2f}%)")
    
    print(f"\n  다음 HL까지 반등:")
    valid_hl = df[df['next_hl_gain_pct'] > 0]
    if len(valid_hl) > 0:
        print(f"    평균: {valid_hl['next_hl_gain_pct'].mean():.2f}%")
        print(f"    중앙값: {valid_hl['next_hl_gain_pct'].median():.2f}%")
        print(f"    최대: {valid_hl['next_hl_gain_pct'].max():.2f}%")
        print(f"    최소: {valid_hl['next_hl_gain_pct'].min():.2f}%")
    
    # 반등 성공 확률
    print(f"\n[5] 반등 확률 분석")
    success_20 = len(df[df['max_gain_20bars'] > 1.5]) / len(df) * 100
    success_40 = len(df[df['max_gain_40bars'] > 2.0]) / len(df) * 100
    success_60 = len(df[df['max_gain_60bars'] > 2.5]) / len(df) * 100
    
    print(f"    20봉 내 +1.5% 이상 반등: {success_20:.1f}%")
    print(f"    40봉 내 +2.0% 이상 반등: {success_40:.1f}%")
    print(f"    60봉 내 +2.5% 이상 반등: {success_60:.1f}%")
    
    # 하락 패턴별 분류
    print(f"\n[6] 하락 패턴 분류")
    
    # 가속 하락 vs 둔화 하락
    df['acceleration'] = df['l3_to_l4_pct'] < df['l2_to_l3_pct']
    accelerating = df[df['acceleration'] == True]
    decelerating = df[df['acceleration'] == False]
    
    print(f"\n  가속 하락 (L3→L4 하락폭 > L2→L3): {len(accelerating)}회 ({len(accelerating)/len(df)*100:.1f}%)")
    if len(accelerating) > 0:
        print(f"    평균 L4 이후 반등 (20봉): {accelerating['max_gain_20bars'].mean():.2f}%")
        print(f"    평균 L4 이후 반등 (60봉): {accelerating['max_gain_60bars'].mean():.2f}%")
    
    print(f"\n  둔화 하락 (L3→L4 하락폭 < L2→L3): {len(decelerating)}회 ({len(decelerating)/len(df)*100:.1f}%)")
    if len(decelerating) > 0:
        print(f"    평균 L4 이후 반등 (20봉): {decelerating['max_gain_20bars'].mean():.2f}%")
        print(f"    평균 L4 이후 반등 (60봉): {decelerating['max_gain_60bars'].mean():.2f}%")


def print_individual_patterns(pattern_details, top_n=10):
    """개별 패턴 상세 출력"""
    if len(pattern_details) == 0:
        return
    
    print("\n" + "=" * 80)
    print(f"LLLL 개별 패턴 상세 (상위 {min(top_n, len(pattern_details))}개)")
    print("=" * 80)
    
    df = pd.DataFrame(pattern_details)
    
    # 반등 성과 기준으로 정렬
    df_sorted = df.sort_values('max_gain_60bars', ascending=False)
    
    for idx, row in df_sorted.head(top_n).iterrows():
        print(f"\n{'='*80}")
        print(f"패턴 #{int(row['pattern_num'])}: {row['l4_datetime']}")
        print(f"{'='*80}")
        
        print(f"\n[가격 구조]")
        print(f"  L1: ${row['l1_price']:.2f}")
        print(f"  L2: ${row['l2_price']:.2f} ({row['l1_to_l2_pct']:+.2f}%)")
        print(f"  L3: ${row['l3_price']:.2f} ({row['l2_to_l3_pct']:+.2f}%)")
        print(f"  L4: ${row['l4_price']:.2f} ({row['l3_to_l4_pct']:+.2f}%)")
        print(f"  총 하락: {row['total_drop_pct']:.2f}%")
        
        print(f"\n[시간 구조]")
        print(f"  L1→L2: {row['l1_to_l2_hours']:.1f}시간")
        print(f"  L2→L3: {row['l2_to_l3_hours']:.1f}시간")
        print(f"  L3→L4: {row['l3_to_l4_hours']:.1f}시간")
        print(f"  총 기간: {row['total_duration_hours']:.1f}시간 ({row['total_duration_hours']/24:.1f}일)")
        
        print(f"\n[L4 지표]")
        print(f"  RSI: {row['l4_rsi']:.2f}")
        print(f"  MACD Hist: {row['l4_macd_hist']:.2f}")
        print(f"  BB Position: {row['l4_bb_position']:.2f}")
        print(f"  Volume Ratio: {row['l4_volume_ratio']:.2f}x")
        print(f"  ATR %: {row['l4_atr_pct']:.2f}%")
        print(f"  Stoch K: {row['l4_stoch_k']:.2f}")
        print(f"  CCI: {row['l4_cci']:.2f}")
        
        print(f"\n[반등 성과]")
        print(f"  20봉 내 (5시간): +{row['max_gain_20bars']:.2f}%")
        print(f"  40봉 내 (10시간): +{row['max_gain_40bars']:.2f}%")
        print(f"  60봉 내 (15시간): +{row['max_gain_60bars']:.2f}%")
        if row['next_hl_gain_pct'] > 0:
            print(f"  다음 HL까지: +{row['next_hl_gain_pct']:.2f}%")


def main():
    print("=" * 80)
    print("차트 패턴 분석: LLLL (4번 연속 Lower Low)")
    print("=" * 80)
    
    # 데이터 로드
    print("\n[1] 데이터 로딩...")
    df = pd.read_csv('btc_15m_ohlcv.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"데이터 기간: {df.iloc[0]['datetime']} ~ {df.iloc[-1]['datetime']}")
    print(f"총 캔들 수: {len(df):,}")
    
    # 지표 계산
    print("\n[2] 지표 계산...")
    df = calculate_indicators(df)
    
    # Swing Low 감지
    print("\n[3] Swing Low 감지...")
    swing_lows = detect_swing_lows(df, left_bars=10, right_bars=10)
    print(f"총 Swing Low 감지: {len(swing_lows)}개")
    
    # LL/HL 패턴 분류
    print("\n[4] LL/HL 패턴 분류...")
    swing_lows = classify_swing_low_pattern(swing_lows)
    
    # LLLL 패턴 분석
    print("\n[5] LLLL 패턴 상세 분석...")
    pattern_details = analyze_LLLL_chart_structure(df, swing_lows)
    
    if len(pattern_details) > 0:
        # 통계 출력
        print_pattern_statistics(pattern_details)
        
        # 개별 패턴 출력
        print_individual_patterns(pattern_details, top_n=10)
        
        # 결과 저장
        output_df = pd.DataFrame(pattern_details)
        output_file = 'LLLL_chart_patterns_analysis.csv'
        output_df.to_csv(output_file, index=False)
        print(f"\n✅ 결과 저장: {output_file}")
    else:
        print("\n⚠️  LLLL 패턴을 찾을 수 없습니다.")
    
    # 최종 요약
    print("\n" + "=" * 80)
    print("분석 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
