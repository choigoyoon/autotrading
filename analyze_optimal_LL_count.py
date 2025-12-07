#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최적의 연속 LL 개수 자동 판단
각 연속 LL 개수별 성과를 비교하여 최적의 진입 조건을 찾음
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
    
    return df


def detect_swing_lows(df, left_bars=10, right_bars=10):
    """Swing Low (L값) 감지"""
    swing_lows = []
    
    for i in range(left_bars, len(df) - right_bars):
        current_low = df.iloc[i]['low']
        
        left_higher = all(df.iloc[i - j]['low'] > current_low for j in range(1, left_bars + 1))
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


def analyze_LL_count_performance(df, swing_lows, min_ll=1, max_ll=10):
    """각 연속 LL 개수별 성과 분석"""
    print("=" * 80)
    print("연속 LL 개수별 성과 분석")
    print("=" * 80)
    
    results = []
    
    for ll_count in range(min_ll, max_ll + 1):
        # 해당 LL 개수 필터링
        patterns = [sl for sl in swing_lows if sl['consecutive_ll'] == ll_count]
        
        if len(patterns) == 0:
            continue
        
        # 각 패턴에 대해 반등 성과 계산
        pattern_results = []
        
        for pattern in patterns:
            idx = pattern['index']
            price = pattern['price']
            
            # L값 지표
            l_indicators = df.iloc[idx]
            
            # 반등 성과 (20, 40, 60봉)
            max_gain_20 = 0
            max_gain_40 = 0
            max_gain_60 = 0
            
            if idx + 20 < len(df):
                next_20 = df.iloc[idx:idx+20]
                max_gain_20 = (next_20['high'].max() - price) / price * 100
            
            if idx + 40 < len(df):
                next_40 = df.iloc[idx:idx+40]
                max_gain_40 = (next_40['high'].max() - price) / price * 100
            
            if idx + 60 < len(df):
                next_60 = df.iloc[idx:idx+60]
                max_gain_60 = (next_60['high'].max() - price) / price * 100
            
            # 극과매도 조건 체크
            conditions_met = 0
            if l_indicators['rsi'] < 30:
                conditions_met += 1
            if l_indicators['macd_hist'] < -50:
                conditions_met += 1
            if l_indicators['bb_position'] < 0.1:
                conditions_met += 1
            if l_indicators['volume_ratio'] > 3.0:
                conditions_met += 1
            if l_indicators['atr_pct'] > 0.5:
                conditions_met += 1
            
            pattern_results.append({
                'datetime': pattern['datetime'],
                'price': price,
                'rsi': l_indicators['rsi'],
                'macd_hist': l_indicators['macd_hist'],
                'volume_ratio': l_indicators['volume_ratio'],
                'conditions_met': conditions_met,
                'max_gain_20': max_gain_20,
                'max_gain_40': max_gain_40,
                'max_gain_60': max_gain_60
            })
        
        pattern_df = pd.DataFrame(pattern_results)
        
        # 통계 계산
        results.append({
            'll_count': ll_count,
            'occurrences': len(patterns),
            'avg_gain_20': pattern_df['max_gain_20'].mean(),
            'median_gain_20': pattern_df['max_gain_20'].median(),
            'avg_gain_40': pattern_df['max_gain_40'].mean(),
            'median_gain_40': pattern_df['max_gain_40'].median(),
            'avg_gain_60': pattern_df['max_gain_60'].mean(),
            'median_gain_60': pattern_df['max_gain_60'].median(),
            'success_rate_20': len(pattern_df[pattern_df['max_gain_20'] > 1.5]) / len(pattern_df) * 100,
            'success_rate_40': len(pattern_df[pattern_df['max_gain_40'] > 2.0]) / len(pattern_df) * 100,
            'success_rate_60': len(pattern_df[pattern_df['max_gain_60'] > 2.5]) / len(pattern_df) * 100,
            'avg_rsi': pattern_df['rsi'].mean(),
            'avg_macd_hist': pattern_df['macd_hist'].mean(),
            'avg_volume_ratio': pattern_df['volume_ratio'].mean(),
            'avg_conditions_met': pattern_df['conditions_met'].mean(),
            'strong_signal_rate': len(pattern_df[pattern_df['conditions_met'] >= 4]) / len(pattern_df) * 100
        })
    
    return pd.DataFrame(results)


def calculate_score(row):
    """각 LL 개수별 종합 점수 계산"""
    # 가중치 설정
    weight_occurrences = 0.15  # 거래 빈도
    weight_gain = 0.35  # 평균 반등률
    weight_success = 0.30  # 성공률
    weight_signal = 0.20  # 시그널 강도
    
    # 정규화 (0-100)
    score_occurrences = min(row['occurrences'] / 100 * 100, 100)  # 100회 이상이면 만점
    score_gain = min(row['avg_gain_60'] / 5 * 100, 100)  # 5% 이상이면 만점
    score_success = row['success_rate_60']  # 이미 0-100
    score_signal = row['strong_signal_rate']  # 이미 0-100
    
    total_score = (
        score_occurrences * weight_occurrences +
        score_gain * weight_gain +
        score_success * weight_success +
        score_signal * weight_signal
    )
    
    return total_score


def print_comparison_table(results_df):
    """비교 테이블 출력"""
    print("\n" + "=" * 80)
    print("연속 LL 개수별 성과 비교표")
    print("=" * 80)
    
    print(f"\n{'LL 개수':>8} | {'발생':>6} | {'평균 반등(60봉)':>15} | {'성공률(60봉)':>13} | {'강한 신호':>10} | {'종합 점수':>10}")
    print("-" * 80)
    
    for idx, row in results_df.iterrows():
        print(f"{int(row['ll_count']):>8} | {int(row['occurrences']):>6} | "
              f"{row['avg_gain_60']:>14.2f}% | {row['success_rate_60']:>12.1f}% | "
              f"{row['strong_signal_rate']:>9.1f}% | {row['score']:>9.1f}")


def print_detailed_analysis(results_df):
    """상세 분석 출력"""
    print("\n" + "=" * 80)
    print("상세 성과 분석")
    print("=" * 80)
    
    for idx, row in results_df.iterrows():
        print(f"\n{'='*80}")
        print(f"연속 LL: {int(row['ll_count'])}번")
        print(f"{'='*80}")
        
        print(f"\n[발생 빈도]")
        print(f"  총 발생: {int(row['occurrences'])}회")
        print(f"  연평균: {row['occurrences'] / 5.7:.1f}회")
        print(f"  월평균: {row['occurrences'] / 69:.1f}회")
        
        print(f"\n[반등 성과]")
        print(f"  20봉 내: 평균 {row['avg_gain_20']:.2f}% (중앙값 {row['median_gain_20']:.2f}%)")
        print(f"  40봉 내: 평균 {row['avg_gain_40']:.2f}% (중앙값 {row['median_gain_40']:.2f}%)")
        print(f"  60봉 내: 평균 {row['avg_gain_60']:.2f}% (중앙값 {row['median_gain_60']:.2f}%)")
        
        print(f"\n[성공률]")
        print(f"  20봉 내 +1.5% 이상: {row['success_rate_20']:.1f}%")
        print(f"  40봉 내 +2.0% 이상: {row['success_rate_40']:.1f}%")
        print(f"  60봉 내 +2.5% 이상: {row['success_rate_60']:.1f}%")
        
        print(f"\n[지표 평균]")
        print(f"  RSI: {row['avg_rsi']:.2f}")
        print(f"  MACD Hist: {row['avg_macd_hist']:.2f}")
        print(f"  Volume Ratio: {row['avg_volume_ratio']:.2f}x")
        print(f"  평균 조건 충족: {row['avg_conditions_met']:.2f}/5개")
        print(f"  강한 신호 비율 (4개 이상): {row['strong_signal_rate']:.1f}%")
        
        print(f"\n[종합 점수]")
        print(f"  {row['score']:.2f}/100")


def determine_optimal_ll_count(results_df):
    """최적 LL 개수 결정"""
    print("\n" + "=" * 80)
    print("최적 연속 LL 개수 판단")
    print("=" * 80)
    
    # 종합 점수 기준 Top 3
    top3 = results_df.nlargest(3, 'score')
    
    print(f"\n[종합 점수 기준 Top 3]")
    for idx, (i, row) in enumerate(top3.iterrows(), 1):
        print(f"\n{idx}위: {int(row['ll_count'])}번 연속 LL")
        print(f"  종합 점수: {row['score']:.2f}/100")
        print(f"  발생 빈도: {int(row['occurrences'])}회 (월평균 {row['occurrences']/69:.1f}회)")
        print(f"  평균 반등: {row['avg_gain_60']:.2f}%")
        print(f"  성공률: {row['success_rate_60']:.1f}%")
    
    # 최적 선택
    best = top3.iloc[0]
    
    print(f"\n{'='*80}")
    print(f"🏆 최적 연속 LL 개수: {int(best['ll_count'])}번")
    print(f"{'='*80}")
    
    print(f"\n[선정 이유]")
    print(f"  1. 높은 종합 점수: {best['score']:.2f}/100")
    print(f"  2. 충분한 거래 빈도: {int(best['occurrences'])}회 (월평균 {best['occurrences']/69:.1f}회)")
    print(f"  3. 우수한 반등률: 평균 {best['avg_gain_60']:.2f}%")
    print(f"  4. 높은 성공률: {best['success_rate_60']:.1f}%")
    print(f"  5. 강한 신호: {best['strong_signal_rate']:.1f}%가 극과매도 4개 이상 조건 충족")
    
    # 대안 제시
    print(f"\n[대안]")
    if len(top3) >= 2:
        second = top3.iloc[1]
        print(f"\n  2순위: {int(second['ll_count'])}번 연속 LL")
        print(f"    - 더 {'많은' if second['occurrences'] > best['occurrences'] else '적은'} 거래 빈도: {int(second['occurrences'])}회")
        print(f"    - {'더 높은' if second['avg_gain_60'] > best['avg_gain_60'] else '낮은'} 평균 반등: {second['avg_gain_60']:.2f}%")
        print(f"    - {'더 높은' if second['success_rate_60'] > best['success_rate_60'] else '낮은'} 성공률: {second['success_rate_60']:.1f}%")
    
    if len(top3) >= 3:
        third = top3.iloc[2]
        print(f"\n  3순위: {int(third['ll_count'])}번 연속 LL")
        print(f"    - 거래 빈도: {int(third['occurrences'])}회")
        print(f"    - 평균 반등: {third['avg_gain_60']:.2f}%")
        print(f"    - 성공률: {third['success_rate_60']:.1f}%")
    
    return int(best['ll_count'])


def main():
    print("=" * 80)
    print("최적의 연속 LL 개수 자동 판단")
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
    
    # 연속 LL 분포 확인
    ll_distribution = {}
    for sl in swing_lows:
        count = sl['consecutive_ll']
        ll_distribution[count] = ll_distribution.get(count, 0) + 1
    
    print(f"\n연속 LL 분포:")
    for count in sorted(ll_distribution.keys()):
        if count > 0:
            print(f"  {count}번: {ll_distribution[count]}회")
    
    # 최대 LL 개수 확인
    max_ll = max([sl['consecutive_ll'] for sl in swing_lows])
    print(f"\n최대 연속 LL: {max_ll}번")
    
    # 각 LL 개수별 성과 분석
    print("\n[5] 각 연속 LL 개수별 성과 분석...")
    results_df = analyze_LL_count_performance(df, swing_lows, min_ll=1, max_ll=min(max_ll, 10))
    
    # 종합 점수 계산
    results_df['score'] = results_df.apply(calculate_score, axis=1)
    
    # 점수 기준 정렬
    results_df = results_df.sort_values('score', ascending=False)
    
    # 결과 출력
    print_comparison_table(results_df)
    print_detailed_analysis(results_df)
    
    # 최적 LL 개수 결정
    optimal_ll = determine_optimal_ll_count(results_df)
    
    # 결과 저장
    output_file = 'optimal_LL_count_analysis.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ 결과 저장: {output_file}")
    
    # 최종 권장사항
    print("\n" + "=" * 80)
    print("최종 권장 전략")
    print("=" * 80)
    
    optimal_row = results_df[results_df['ll_count'] == optimal_ll].iloc[0]
    
    print(f"""
🎯 권장 전략: {optimal_ll}번 연속 LL 기반 Long 전략

[진입 조건]
  1. {optimal_ll}번 연속 Lower Low 발생
  2. 극과매도 조건 (5개 중 4개 이상 충족):
     - RSI < 30
     - MACD Histogram < -50
     - BB Position < 0.1
     - Volume Ratio > 3.0
     - ATR % > 0.5
  3. L값 확정 후 첫 양봉 2개 연속
  4. RSI > 35 또는 MACD Hist 상승 전환

[청산 조건]
  TP1: +2.0% (50% 청산)
  TP2: +3.5% (50% 청산)
  SL: L값 -1.0%
  Time Stop: 60봉 (15시간)

[예상 성과]
  월평균 거래: {optimal_row['occurrences']/69:.1f}회
  연평균 거래: {optimal_row['occurrences']/5.7:.1f}회
  평균 반등: {optimal_row['avg_gain_60']:.2f}%
  성공률: {optimal_row['success_rate_60']:.1f}%
  
  예상 월평균 수익: {optimal_row['occurrences']/69 * optimal_row['avg_gain_60'] * optimal_row['success_rate_60'] / 100:.2f}%
""")


if __name__ == "__main__":
    main()
