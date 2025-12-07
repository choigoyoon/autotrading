#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4번 연속 LL 반등 사례 상세 분석
각 반등 케이스별로 성공/실패 원인을 분석
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
    """LL/HL 패턴 분류"""
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


def analyze_bounce_pattern(df, l4_idx, l4_price, look_ahead=60):
    """반등 패턴 상세 분석"""
    if l4_idx + look_ahead >= len(df):
        look_ahead = len(df) - l4_idx - 1
    
    bounce_data = df.iloc[l4_idx:l4_idx + look_ahead + 1].copy()
    bounce_data['bars_from_l4'] = range(len(bounce_data))
    bounce_data['gain_from_l4'] = (bounce_data['high'] - l4_price) / l4_price * 100
    bounce_data['close_gain_from_l4'] = (bounce_data['close'] - l4_price) / l4_price * 100
    
    # 반등 단계 분석
    analysis = {
        'immediate_bounce': False,  # 첫 3봉 내 반등
        'first_green_bar': None,  # 첫 양봉 위치
        'consecutive_greens': 0,  # 연속 양봉 개수
        'max_gain': bounce_data['gain_from_l4'].max(),
        'max_gain_bar': bounce_data['gain_from_l4'].idxmax() - l4_idx if len(bounce_data) > 0 else 0,
        'first_1pct_bar': None,  # +1% 첫 도달
        'first_2pct_bar': None,  # +2% 첫 도달
        'first_3pct_bar': None,  # +3% 첫 도달
        'pullback_count': 0,  # 하락 반전 횟수
        'rsi_recovery_bar': None,  # RSI 35 돌파
        'macd_reversal_bar': None,  # MACD Hist 양전환
        'volume_spike': False,  # 볼륨 급증 여부
    }
    
    # 첫 양봉 찾기
    for i in range(1, min(20, len(bounce_data))):
        if bounce_data.iloc[i]['close'] > bounce_data.iloc[i]['open']:
            analysis['first_green_bar'] = i
            break
    
    # 연속 양봉 개수
    if analysis['first_green_bar']:
        consecutive = 0
        for i in range(analysis['first_green_bar'], min(analysis['first_green_bar'] + 10, len(bounce_data))):
            if bounce_data.iloc[i]['close'] > bounce_data.iloc[i]['open']:
                consecutive += 1
            else:
                break
        analysis['consecutive_greens'] = consecutive
    
    # 즉시 반등 (첫 3봉 내 +0.5% 이상)
    if len(bounce_data) >= 3:
        first_3 = bounce_data.iloc[:3]
        if first_3['gain_from_l4'].max() >= 0.5:
            analysis['immediate_bounce'] = True
    
    # 목표가 도달 시점
    for i, row in bounce_data.iterrows():
        gain = row['gain_from_l4']
        bar_num = row['bars_from_l4']
        
        if gain >= 1.0 and analysis['first_1pct_bar'] is None:
            analysis['first_1pct_bar'] = bar_num
        if gain >= 2.0 and analysis['first_2pct_bar'] is None:
            analysis['first_2pct_bar'] = bar_num
        if gain >= 3.0 and analysis['first_3pct_bar'] is None:
            analysis['first_3pct_bar'] = bar_num
    
    # RSI 회복
    for i, row in bounce_data.iterrows():
        if row['rsi'] >= 35 and analysis['rsi_recovery_bar'] is None:
            analysis['rsi_recovery_bar'] = row['bars_from_l4']
            break
    
    # MACD 반전
    for i, row in bounce_data.iterrows():
        if row['macd_hist'] > 0 and analysis['macd_reversal_bar'] is None:
            analysis['macd_reversal_bar'] = row['bars_from_l4']
            break
    
    # 풀백 횟수 (고점 대비 -0.5% 이상 하락)
    max_so_far = l4_price
    for i, row in bounce_data.iterrows():
        current_high = row['high']
        if current_high > max_so_far:
            max_so_far = current_high
        
        current_low = row['low']
        pullback = (current_low - max_so_far) / max_so_far * 100
        if pullback < -0.5:
            analysis['pullback_count'] += 1
    
    # 볼륨 스파이크 (첫 5봉 내 3.0x 이상)
    if len(bounce_data) >= 5:
        first_5 = bounce_data.iloc[:5]
        if (first_5['volume_ratio'] > 3.0).any():
            analysis['volume_spike'] = True
    
    return analysis


def categorize_bounce_case(max_gain_60, analysis):
    """반등 케이스 분류"""
    if max_gain_60 >= 5.0:
        return "대성공 (5%+)"
    elif max_gain_60 >= 3.0:
        return "성공 (3-5%)"
    elif max_gain_60 >= 1.5:
        return "보통 (1.5-3%)"
    elif max_gain_60 >= 0:
        return "약반등 (0-1.5%)"
    else:
        return "실패 (마이너스)"


def analyze_all_llll_cases(df, swing_lows):
    """모든 LLLL 케이스 분석"""
    print("=" * 80)
    print("4번 연속 LL (LLLL) 반등 사례 상세 분석")
    print("=" * 80)
    
    # 4번 연속 LL 필터링
    llll_patterns = [sl for sl in swing_lows if sl['consecutive_ll'] == 4]
    
    print(f"\n총 LLLL 패턴: {len(llll_patterns)}개")
    
    cases = []
    
    for idx, llll in enumerate(llll_patterns, 1):
        l4_idx = llll['index']
        l4_price = llll['price']
        l4_time = llll['datetime']
        
        # L4 지표
        l4_indicators = df.iloc[l4_idx]
        
        # 반등 패턴 분석
        bounce_analysis = analyze_bounce_pattern(df, l4_idx, l4_price, look_ahead=60)
        
        # 60봉 내 최대 반등
        if l4_idx + 60 < len(df):
            next_60 = df.iloc[l4_idx:l4_idx+60]
            max_gain_60 = (next_60['high'].max() - l4_price) / l4_price * 100
        else:
            max_gain_60 = bounce_analysis['max_gain']
        
        # 케이스 분류
        case_type = categorize_bounce_case(max_gain_60, bounce_analysis)
        
        # 극과매도 조건 체크
        conditions_met = 0
        conditions_detail = []
        
        if l4_indicators['rsi'] < 30:
            conditions_met += 1
            conditions_detail.append('RSI<30')
        if l4_indicators['macd_hist'] < -50:
            conditions_met += 1
            conditions_detail.append('MACD<-50')
        if l4_indicators['bb_position'] < 0.1:
            conditions_met += 1
            conditions_detail.append('BB<0.1')
        if l4_indicators['volume_ratio'] > 3.0:
            conditions_met += 1
            conditions_detail.append('Vol>3.0')
        if l4_indicators['atr_pct'] > 0.5:
            conditions_met += 1
            conditions_detail.append('ATR>0.5')
        
        cases.append({
            'case_num': idx,
            'datetime': l4_time,
            'l4_price': l4_price,
            'case_type': case_type,
            'max_gain_60': max_gain_60,
            'max_gain_bar': bounce_analysis['max_gain_bar'],
            'immediate_bounce': bounce_analysis['immediate_bounce'],
            'first_green_bar': bounce_analysis['first_green_bar'],
            'consecutive_greens': bounce_analysis['consecutive_greens'],
            'first_1pct_bar': bounce_analysis['first_1pct_bar'],
            'first_2pct_bar': bounce_analysis['first_2pct_bar'],
            'first_3pct_bar': bounce_analysis['first_3pct_bar'],
            'pullback_count': bounce_analysis['pullback_count'],
            'rsi_recovery_bar': bounce_analysis['rsi_recovery_bar'],
            'macd_reversal_bar': bounce_analysis['macd_reversal_bar'],
            'volume_spike': bounce_analysis['volume_spike'],
            'l4_rsi': l4_indicators['rsi'],
            'l4_macd_hist': l4_indicators['macd_hist'],
            'l4_bb_position': l4_indicators['bb_position'],
            'l4_volume_ratio': l4_indicators['volume_ratio'],
            'l4_atr_pct': l4_indicators['atr_pct'],
            'conditions_met': conditions_met,
            'conditions_detail': ', '.join(conditions_detail)
        })
    
    return pd.DataFrame(cases)


def print_case_type_summary(cases_df):
    """케이스별 요약 통계"""
    print("\n" + "=" * 80)
    print("반등 케이스별 요약")
    print("=" * 80)
    
    case_types = ["대성공 (5%+)", "성공 (3-5%)", "보통 (1.5-3%)", "약반등 (0-1.5%)", "실패 (마이너스)"]
    
    print(f"\n{'케이스':<20} | {'발생':<8} | {'비율':<8} | {'평균 반등':<12} | {'즉시 반등':<10} | {'강한 신호':<10}")
    print("-" * 80)
    
    for case_type in case_types:
        subset = cases_df[cases_df['case_type'] == case_type]
        if len(subset) == 0:
            continue
        
        count = len(subset)
        ratio = count / len(cases_df) * 100
        avg_gain = subset['max_gain_60'].mean()
        immediate = subset['immediate_bounce'].sum() / len(subset) * 100
        strong_signal = len(subset[subset['conditions_met'] >= 4]) / len(subset) * 100
        
        print(f"{case_type:<20} | {count:<8} | {ratio:<7.1f}% | {avg_gain:<11.2f}% | {immediate:<9.1f}% | {strong_signal:<9.1f}%")


def print_detailed_cases(cases_df, case_type, top_n=5):
    """케이스별 상세 사례"""
    print(f"\n{'='*80}")
    print(f"{case_type} 상세 사례 (상위 {top_n}개)")
    print(f"{'='*80}")
    
    subset = cases_df[cases_df['case_type'] == case_type]
    
    if len(subset) == 0:
        print(f"\n{case_type} 케이스가 없습니다.")
        return
    
    subset_sorted = subset.sort_values('max_gain_60', ascending=False)
    
    for idx, (i, row) in enumerate(subset_sorted.head(top_n).iterrows(), 1):
        print(f"\n{'='*80}")
        print(f"사례 #{int(row['case_num'])}: {row['datetime']}")
        print(f"{'='*80}")
        
        print(f"\n[기본 정보]")
        print(f"  L4 가격: ${row['l4_price']:.2f}")
        print(f"  60봉 내 최대 반등: {row['max_gain_60']:.2f}% (at {int(row['max_gain_bar'])}번째 봉)")
        
        print(f"\n[L4 지표]")
        print(f"  RSI: {row['l4_rsi']:.2f}")
        print(f"  MACD Hist: {row['l4_macd_hist']:.2f}")
        print(f"  BB Position: {row['l4_bb_position']:.2f}")
        print(f"  Volume Ratio: {row['l4_volume_ratio']:.2f}x")
        print(f"  ATR %: {row['l4_atr_pct']:.2f}%")
        print(f"  극과매도 조건: {int(row['conditions_met'])}/5개 충족")
        print(f"  충족 조건: {row['conditions_detail']}")
        
        print(f"\n[반등 패턴]")
        print(f"  즉시 반등 (첫 3봉): {'✅ 예' if row['immediate_bounce'] else '❌ 아니오'}")
        print(f"  첫 양봉 위치: {int(row['first_green_bar']) if pd.notna(row['first_green_bar']) else 'N/A'}번째 봉")
        print(f"  연속 양봉: {int(row['consecutive_greens'])}개")
        print(f"  볼륨 스파이크 (첫 5봉): {'✅ 있음' if row['volume_spike'] else '❌ 없음'}")
        
        print(f"\n[목표가 도달 시점]")
        print(f"  +1.0%: {int(row['first_1pct_bar']) if pd.notna(row['first_1pct_bar']) else 'N/A'}번째 봉")
        print(f"  +2.0%: {int(row['first_2pct_bar']) if pd.notna(row['first_2pct_bar']) else 'N/A'}번째 봉")
        print(f"  +3.0%: {int(row['first_3pct_bar']) if pd.notna(row['first_3pct_bar']) else 'N/A'}번째 봉")
        
        print(f"\n[지표 회복]")
        print(f"  RSI > 35 도달: {int(row['rsi_recovery_bar']) if pd.notna(row['rsi_recovery_bar']) else 'N/A'}번째 봉")
        print(f"  MACD Hist > 0 전환: {int(row['macd_reversal_bar']) if pd.notna(row['macd_reversal_bar']) else 'N/A'}번째 봉")
        
        print(f"\n[변동성]")
        print(f"  풀백 횟수 (-0.5% 이상): {int(row['pullback_count'])}회")


def analyze_success_factors(cases_df):
    """성공 요인 분석"""
    print("\n" + "=" * 80)
    print("성공 vs 실패 요인 비교")
    print("=" * 80)
    
    # 성공 (3% 이상) vs 실패 (1.5% 미만)
    success = cases_df[cases_df['max_gain_60'] >= 3.0]
    failure = cases_df[cases_df['max_gain_60'] < 1.5]
    
    print(f"\n성공 (3% 이상): {len(success)}개")
    print(f"실패 (1.5% 미만): {len(failure)}개")
    
    print(f"\n[L4 지표 비교]")
    print(f"{'지표':<20} | {'성공 평균':<15} | {'실패 평균':<15} | {'차이':<15}")
    print("-" * 70)
    
    indicators = [
        ('RSI', 'l4_rsi'),
        ('MACD Hist', 'l4_macd_hist'),
        ('BB Position', 'l4_bb_position'),
        ('Volume Ratio', 'l4_volume_ratio'),
        ('ATR %', 'l4_atr_pct'),
        ('조건 충족', 'conditions_met')
    ]
    
    for label, col in indicators:
        success_avg = success[col].mean()
        failure_avg = failure[col].mean()
        diff = success_avg - failure_avg
        print(f"{label:<20} | {success_avg:<15.2f} | {failure_avg:<15.2f} | {diff:+<15.2f}")
    
    print(f"\n[반등 패턴 비교]")
    print(f"{'패턴':<30} | {'성공':<10} | {'실패':<10}")
    print("-" * 55)
    
    patterns = [
        ('즉시 반등 (첫 3봉)', 'immediate_bounce'),
        ('볼륨 스파이크', 'volume_spike'),
    ]
    
    for label, col in patterns:
        success_rate = success[col].sum() / len(success) * 100
        failure_rate = failure[col].sum() / len(failure) * 100
        print(f"{label:<30} | {success_rate:<9.1f}% | {failure_rate:<9.1f}%")
    
    print(f"\n[타이밍 비교 (평균)]")
    timing_cols = [
        ('첫 양봉 위치', 'first_green_bar'),
        ('연속 양봉 개수', 'consecutive_greens'),
        ('RSI 회복 시점', 'rsi_recovery_bar'),
        ('풀백 횟수', 'pullback_count')
    ]
    
    print(f"{'항목':<30} | {'성공':<10} | {'실패':<10}")
    print("-" * 55)
    
    for label, col in timing_cols:
        success_avg = success[col].mean()
        failure_avg = failure[col].mean()
        print(f"{label:<30} | {success_avg:<9.1f} | {failure_avg:<9.1f}")


def print_key_insights(cases_df):
    """핵심 인사이트"""
    print("\n" + "=" * 80)
    print("핵심 인사이트 및 패턴")
    print("=" * 80)
    
    # 즉시 반등의 중요성
    immediate = cases_df[cases_df['immediate_bounce'] == True]
    no_immediate = cases_df[cases_df['immediate_bounce'] == False]
    
    print(f"\n[1] 즉시 반등의 중요성")
    print(f"  즉시 반등 있음: 평균 {immediate['max_gain_60'].mean():.2f}% (n={len(immediate)})")
    print(f"  즉시 반등 없음: 평균 {no_immediate['max_gain_60'].mean():.2f}% (n={len(no_immediate)})")
    
    # 강한 신호의 효과
    strong = cases_df[cases_df['conditions_met'] >= 4]
    weak = cases_df[cases_df['conditions_met'] < 4]
    
    print(f"\n[2] 강한 신호 (4개 이상 조건 충족)의 효과")
    print(f"  강한 신호: 평균 {strong['max_gain_60'].mean():.2f}% (n={len(strong)})")
    print(f"  약한 신호: 평균 {weak['max_gain_60'].mean():.2f}% (n={len(weak)})")
    
    # 볼륨 스파이크
    vol_spike = cases_df[cases_df['volume_spike'] == True]
    no_vol_spike = cases_df[cases_df['volume_spike'] == False]
    
    print(f"\n[3] 볼륨 스파이크의 영향")
    print(f"  볼륨 스파이크 있음: 평균 {vol_spike['max_gain_60'].mean():.2f}% (n={len(vol_spike)})")
    print(f"  볼륨 스파이크 없음: 평균 {no_vol_spike['max_gain_60'].mean():.2f}% (n={len(no_vol_spike)})")
    
    # 최적 조합
    optimal = cases_df[
        (cases_df['immediate_bounce'] == True) &
        (cases_df['conditions_met'] >= 4) &
        (cases_df['volume_spike'] == True)
    ]
    
    print(f"\n[4] 최적 조합 (즉시 반등 + 강한 신호 + 볼륨 스파이크)")
    print(f"  발생: {len(optimal)}회")
    if len(optimal) > 0:
        print(f"  평균 반등: {optimal['max_gain_60'].mean():.2f}%")
        print(f"  성공률 (3% 이상): {len(optimal[optimal['max_gain_60'] >= 3.0]) / len(optimal) * 100:.1f}%")


def main():
    print("=" * 80)
    print("4번 연속 LL (LLLL) 반등 사례 상세 분석")
    print("=" * 80)
    
    # 데이터 로드
    print("\n[1] 데이터 로딩...")
    df = pd.read_csv('btc_15m_ohlcv.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"데이터 기간: {df.iloc[0]['datetime']} ~ {df.iloc[-1]['datetime']}")
    
    # 지표 계산
    print("\n[2] 지표 계산...")
    df = calculate_indicators(df)
    
    # Swing Low 감지
    print("\n[3] Swing Low 감지...")
    swing_lows = detect_swing_lows(df, left_bars=10, right_bars=10)
    swing_lows = classify_swing_low_pattern(swing_lows)
    
    # 모든 LLLL 케이스 분석
    print("\n[4] LLLL 반등 케이스 분석...")
    cases_df = analyze_all_llll_cases(df, swing_lows)
    
    # 결과 출력
    print_case_type_summary(cases_df)
    
    # 케이스별 상세 사례
    print_detailed_cases(cases_df, "대성공 (5%+)", top_n=5)
    print_detailed_cases(cases_df, "성공 (3-5%)", top_n=5)
    print_detailed_cases(cases_df, "보통 (1.5-3%)", top_n=3)
    print_detailed_cases(cases_df, "약반등 (0-1.5%)", top_n=3)
    print_detailed_cases(cases_df, "실패 (마이너스)", top_n=3)
    
    # 성공 요인 분석
    analyze_success_factors(cases_df)
    
    # 핵심 인사이트
    print_key_insights(cases_df)
    
    # 결과 저장
    output_file = 'llll_bounce_cases_detailed.csv'
    cases_df.to_csv(output_file, index=False)
    print(f"\n✅ 결과 저장: {output_file}")


if __name__ == "__main__":
    main()
