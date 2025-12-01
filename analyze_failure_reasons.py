#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Floor Catch Strategy - 실패 이유 분석
왜 상승이 없는지 상세 분석
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_data():
    """데이터 로드"""
    df = pd.read_csv('btc_15m_ohlcv.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    trades = pd.read_csv('floor_catch_strategy_results.csv')
    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
    trades['exit_time'] = pd.to_datetime(trades['exit_time'])
    
    return df, trades


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
    
    # Volume
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    return df


def analyze_entry_to_exit_behavior(df, trades):
    """진입 후 가격 움직임 상세 분석"""
    print("=" * 80)
    print("실패 이유 분석 #1: 진입 후 가격 움직임")
    print("=" * 80)
    
    win_trades = trades[trades['pnl_pct'] > 0]
    loss_trades = trades[trades['pnl_pct'] < 0]
    
    print(f"\n[1] 승리 거래 vs 패배 거래 비교")
    print(f"승리: {len(win_trades)}회 ({len(win_trades)/len(trades)*100:.1f}%)")
    print(f"패배: {len(loss_trades)}회 ({len(loss_trades)/len(trades)*100:.1f}%)")
    
    # 진입 후 최대 상승/하락 분석
    detailed_analysis = []
    
    for idx, trade in trades.iterrows():
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        entry_price = trade['entry_price']
        
        # 해당 기간 데이터 추출
        period_data = df[(df['datetime'] >= entry_time) & (df['datetime'] <= exit_time)]
        
        if len(period_data) == 0:
            continue
        
        # 최대 상승/하락률 계산
        max_high = period_data['high'].max()
        min_low = period_data['low'].min()
        
        max_gain_pct = (max_high - entry_price) / entry_price * 100
        max_loss_pct = (min_low - entry_price) / entry_price * 100
        
        # 첫 N봉 동안의 움직임
        first_5_candles = period_data.head(5)
        if len(first_5_candles) > 0:
            first_5_max = first_5_candles['high'].max()
            first_5_min = first_5_candles['low'].min()
            first_5_gain = (first_5_max - entry_price) / entry_price * 100
            first_5_loss = (first_5_min - entry_price) / entry_price * 100
        else:
            first_5_gain = 0
            first_5_loss = 0
        
        # 즉시 반등 여부 (첫 3봉)
        first_3_candles = period_data.head(3)
        immediate_rally = False
        if len(first_3_candles) >= 3:
            green_count = sum(first_3_candles['close'] > first_3_candles['open'])
            if green_count >= 2 and (first_3_candles.iloc[2]['close'] > entry_price * 1.005):
                immediate_rally = True
        
        detailed_analysis.append({
            'trade_idx': idx,
            'pnl_pct': trade['pnl_pct'],
            'exit_type': trade['exit_type'],
            'consecutive_ll': trade['consecutive_ll'],
            'max_gain_pct': max_gain_pct,
            'max_loss_pct': max_loss_pct,
            'first_5_gain': first_5_gain,
            'first_5_loss': first_5_loss,
            'immediate_rally': immediate_rally,
            'tp1_hit': trade['tp1_hit']
        })
    
    analysis_df = pd.DataFrame(detailed_analysis)
    
    print(f"\n[2] 진입 후 최대 상승/하락 분석")
    print(f"\n전체 거래:")
    print(f"  평균 최대 상승: {analysis_df['max_gain_pct'].mean():.2f}%")
    print(f"  평균 최대 하락: {analysis_df['max_loss_pct'].mean():.2f}%")
    print(f"  첫 5봉 평균 상승: {analysis_df['first_5_gain'].mean():.2f}%")
    print(f"  첫 5봉 평균 하락: {analysis_df['first_5_loss'].mean():.2f}%")
    
    print(f"\n승리 거래:")
    win_analysis = analysis_df[analysis_df['pnl_pct'] > 0]
    print(f"  평균 최대 상승: {win_analysis['max_gain_pct'].mean():.2f}%")
    print(f"  평균 최대 하락: {win_analysis['max_loss_pct'].mean():.2f}%")
    print(f"  첫 5봉 평균 상승: {win_analysis['first_5_gain'].mean():.2f}%")
    print(f"  첫 5봉 평균 하락: {win_analysis['first_5_loss'].mean():.2f}%")
    
    print(f"\n패배 거래:")
    loss_analysis = analysis_df[analysis_df['pnl_pct'] <= 0]
    print(f"  평균 최대 상승: {loss_analysis['max_gain_pct'].mean():.2f}%")
    print(f"  평균 최대 하락: {loss_analysis['max_loss_pct'].mean():.2f}%")
    print(f"  첫 5봉 평균 상승: {loss_analysis['first_5_gain'].mean():.2f}%")
    print(f"  첫 5봉 평균 하락: {loss_analysis['first_5_loss'].mean():.2f}%")
    
    # 즉시 반등 분석
    immediate_rally_count = analysis_df['immediate_rally'].sum()
    print(f"\n[3] 즉시 반등 (첫 3봉 내 +0.5% 이상) 분석")
    print(f"  즉시 반등 발생: {immediate_rally_count}회 ({immediate_rally_count/len(analysis_df)*100:.1f}%)")
    print(f"  즉시 반등 없음: {len(analysis_df) - immediate_rally_count}회 ({(len(analysis_df) - immediate_rally_count)/len(analysis_df)*100:.1f}%)")
    
    return analysis_df


def analyze_sl_hits(df, trades):
    """손절 발생 원인 분석"""
    print("\n" + "=" * 80)
    print("실패 이유 분석 #2: 손절 발생 원인")
    print("=" * 80)
    
    sl_trades = trades[trades['exit_type'] == 'SL']
    
    print(f"\n총 {len(sl_trades)}회 손절 발생 ({len(sl_trades)/len(trades)*100:.1f}%)")
    
    # 손절까지 걸린 시간
    print(f"\n[1] 손절까지 걸린 시간")
    print(f"  평균: {sl_trades['duration_hours'].mean():.1f}시간")
    print(f"  중앙값: {sl_trades['duration_hours'].median():.1f}시간")
    print(f"  최소: {sl_trades['duration_hours'].min():.1f}시간")
    print(f"  최대: {sl_trades['duration_hours'].max():.1f}시간")
    
    # 빠른 손절 (1시간 이내)
    fast_sl = sl_trades[sl_trades['duration_hours'] < 1]
    print(f"\n  1시간 이내 손절: {len(fast_sl)}회 ({len(fast_sl)/len(sl_trades)*100:.1f}%)")
    
    # 연속 LL별 손절률
    print(f"\n[2] 연속 LL 카운트별 손절 발생률")
    for ll_count in sorted(trades['consecutive_ll'].unique()):
        subset = trades[trades['consecutive_ll'] == ll_count]
        sl_subset = subset[subset['exit_type'] == 'SL']
        sl_rate = len(sl_subset) / len(subset) * 100
        print(f"  {ll_count}번 LL: {len(sl_subset)}/{len(subset)}회 ({sl_rate:.1f}%)")
    
    # 추가 하락 분석
    print(f"\n[3] 진입 후 추가 하락 발생")
    continued_drops = []
    
    for idx, trade in sl_trades.iterrows():
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        entry_price = trade['entry_price']
        l_price = trade['l_price']
        
        # 해당 기간 데이터
        period_data = df[(df['datetime'] >= entry_time) & (df['datetime'] <= exit_time)]
        
        if len(period_data) == 0:
            continue
        
        min_low = period_data['low'].min()
        additional_drop = (min_low - l_price) / l_price * 100
        
        continued_drops.append({
            'entry_price': entry_price,
            'l_price': l_price,
            'min_low': min_low,
            'additional_drop_pct': additional_drop,
            'consecutive_ll': trade['consecutive_ll']
        })
    
    drop_df = pd.DataFrame(continued_drops)
    print(f"\n  평균 추가 하락 (L값 대비): {drop_df['additional_drop_pct'].mean():.2f}%")
    print(f"  중앙값 추가 하락: {drop_df['additional_drop_pct'].median():.2f}%")
    print(f"  최대 추가 하락: {drop_df['additional_drop_pct'].min():.2f}%")
    
    return sl_trades, drop_df


def analyze_false_bottom_signals(df, trades):
    """거짓 바닥 신호 분석"""
    print("\n" + "=" * 80)
    print("실패 이유 분석 #3: 거짓 바닥 신호 (False Bottom)")
    print("=" * 80)
    
    # 패배 거래 중 TP1조차 도달하지 못한 경우
    failed_tp1 = trades[(trades['pnl_pct'] < 0) & (trades['tp1_hit'] == False)]
    
    print(f"\n[1] TP1 미달성 실패 거래")
    print(f"  총 {len(failed_tp1)}회 ({len(failed_tp1)/len(trades)*100:.1f}%)")
    print(f"  평균 PNL: {failed_tp1['pnl_pct'].mean():.2f}%")
    
    # 진입 후 즉시 하락
    print(f"\n[2] 진입 타이밍 문제 분석")
    
    early_entries = []
    
    for idx, trade in failed_tp1.iterrows():
        entry_time = trade['entry_time']
        entry_price = trade['entry_price']
        
        # 진입 후 10봉 데이터
        entry_idx = df[df['datetime'] == entry_time].index
        if len(entry_idx) == 0:
            continue
        
        entry_idx = entry_idx[0]
        next_10 = df.iloc[entry_idx:entry_idx+10]
        
        if len(next_10) == 0:
            continue
        
        # 즉시 하락 체크
        first_candle_drop = (next_10.iloc[0]['low'] - entry_price) / entry_price * 100
        min_in_10 = next_10['low'].min()
        max_drop_10 = (min_in_10 - entry_price) / entry_price * 100
        
        early_entries.append({
            'consecutive_ll': trade['consecutive_ll'],
            'first_candle_drop': first_candle_drop,
            'max_drop_10': max_drop_10,
            'pnl': trade['pnl_pct']
        })
    
    early_df = pd.DataFrame(early_entries)
    
    if len(early_df) > 0:
        print(f"\n  첫 봉 평균 하락: {early_df['first_candle_drop'].mean():.2f}%")
        print(f"  10봉 내 최대 하락: {early_df['max_drop_10'].mean():.2f}%")
        
        # 즉시 하락한 거래 비율
        immediate_drop = early_df[early_df['first_candle_drop'] < -0.5]
        print(f"  첫 봉부터 -0.5% 이상 하락: {len(immediate_drop)}회 ({len(immediate_drop)/len(early_df)*100:.1f}%)")
    
    return failed_tp1, early_df


def analyze_time_stop_trades(df, trades):
    """타임스톱 거래 분석"""
    print("\n" + "=" * 80)
    print("실패 이유 분석 #4: 타임스톱 거래 분석")
    print("=" * 80)
    
    time_trades = trades[trades['exit_type'] == 'TIME']
    
    print(f"\n총 {len(time_trades)}회 타임스톱 ({len(time_trades)/len(trades)*100:.1f}%)")
    print(f"평균 PNL: {time_trades['pnl_pct'].mean():.2f}%")
    
    # 타임스톱 중 수익/손실
    time_profit = time_trades[time_trades['pnl_pct'] > 0]
    time_loss = time_trades[time_trades['pnl_pct'] < 0]
    
    print(f"\n[1] 타임스톱 수익/손실 분포")
    print(f"  수익: {len(time_profit)}회 ({len(time_profit)/len(time_trades)*100:.1f}%) - 평균 {time_profit['pnl_pct'].mean():.2f}%")
    print(f"  손실: {len(time_loss)}회 ({len(time_loss)/len(time_trades)*100:.1f}%) - 평균 {time_loss['pnl_pct'].mean():.2f}%")
    
    # 타임스톱 발생 시 TP1 달성 여부
    time_with_tp1 = time_trades[time_trades['tp1_hit'] == True]
    time_without_tp1 = time_trades[time_trades['tp1_hit'] == False]
    
    print(f"\n[2] 타임스톱 시 TP1 달성 여부")
    print(f"  TP1 달성 후 타임스톱: {len(time_with_tp1)}회 - 평균 PNL {time_with_tp1['pnl_pct'].mean():.2f}%")
    print(f"  TP1 미달성 타임스톱: {len(time_without_tp1)}회 - 평균 PNL {time_without_tp1['pnl_pct'].mean():.2f}%")
    
    print(f"\n[3] 타임스톱 원인 분석")
    print(f"  → 24시간 내 TP2 (+3.5%)에 도달하지 못함")
    print(f"  → 상승 속도 부족 또는 횡보")


def analyze_winning_trades(df, trades):
    """승리 거래의 특징 분석"""
    print("\n" + "=" * 80)
    print("성공 패턴 분석: 승리 거래의 특징")
    print("=" * 80)
    
    win_trades = trades[trades['pnl_pct'] > 0]
    
    print(f"\n총 {len(win_trades)}회 승리 ({len(win_trades)/len(trades)*100:.1f}%)")
    print(f"평균 PNL: {win_trades['pnl_pct'].mean():.2f}%")
    
    # 청산 타입별
    print(f"\n[1] 승리 거래 청산 타입")
    for exit_type in win_trades['exit_type'].unique():
        subset = win_trades[win_trades['exit_type'] == exit_type]
        print(f"  {exit_type}: {len(subset)}회 - 평균 PNL {subset['pnl_pct'].mean():.2f}%")
    
    # 연속 LL별
    print(f"\n[2] 승리 거래의 연속 LL 분포")
    for ll_count in sorted(win_trades['consecutive_ll'].unique()):
        subset = win_trades[win_trades['consecutive_ll'] == ll_count]
        print(f"  {ll_count}번 LL: {len(subset)}회 - 평균 PNL {subset['pnl_pct'].mean():.2f}%")
    
    # TP2 달성 거래
    tp2_trades = win_trades[win_trades['exit_type'] == 'TP2']
    print(f"\n[3] TP2 완전 달성 거래")
    print(f"  총 {len(tp2_trades)}회")
    print(f"  평균 소요 시간: {tp2_trades['duration_hours'].mean():.1f}시간")
    print(f"  연속 LL 분포:")
    for ll_count in sorted(tp2_trades['consecutive_ll'].unique()):
        count = len(tp2_trades[tp2_trades['consecutive_ll'] == ll_count])
        print(f"    {ll_count}번 LL: {count}회")


def final_summary():
    """최종 요약 및 결론"""
    print("\n" + "=" * 80)
    print("최종 결론: 전략 실패 이유 요약")
    print("=" * 80)
    
    print("""
🔴 전략 실패의 주요 원인 5가지:

1. **거짓 바닥 신호 (False Bottom)**
   - L값이 실제 바닥이 아닌 경우가 많음
   - 진입 후 추가 하락 발생
   - 3-4번 LL도 충분한 하락이 아님

2. **손절 발생률 46.7%**
   - 거의 절반의 거래가 손절
   - 평균 -2.00% 손실
   - L값 -0.5% SL이 너무 타이트

3. **즉시 반등 부족**
   - 진입 후 즉각적인 반등이 없음
   - 첫 5봉 평균 상승이 미미
   - 횡보 또는 추가 하락 지속

4. **상승 모멘텀 부족**
   - 타임스톡 30.4% 발생
   - 24시간 내 TP2 도달 실패
   - 반등 속도가 너무 느림

5. **진입 타이밍 문제**
   - "첫 양봉" 조건이 너무 이름
   - 실제 반등 시작점을 놓침
   - 더 강한 확인 신호 필요

📊 개선이 필요한 핵심 요소:

✅ **연속 LL 카운트를 5번 이상으로 상향**
   → 더 깊은 하락 후 진입

✅ **진입 조건 강화**
   → RSI 다이버전스, 볼륨 급증, MACD 전환 확인

✅ **손절폭 확대**
   → L값 -0.5% → -1.0% 또는 -1.5%

✅ **타임프레임 상승**
   → 15분봉 → 1시간봉 or 4시간봉

✅ **추가 확인 신호**
   → 2-3개 양봉 연속 + RSI > 40 + MACD Hist 상승
""")


def main():
    print("=" * 80)
    print("Floor Catch Strategy - 실패 이유 심층 분석")
    print("=" * 80)
    
    # 데이터 로드
    df, trades = load_data()
    df = calculate_indicators(df)
    
    print(f"\n분석 기간: {df.iloc[0]['datetime']} ~ {df.iloc[-1]['datetime']}")
    print(f"총 거래 수: {len(trades)}")
    print(f"승률: {len(trades[trades['pnl_pct'] > 0]) / len(trades) * 100:.1f}%")
    print(f"총 PNL: {trades['pnl_pct'].sum():.2f}%")
    
    # 분석 실행
    analysis_df = analyze_entry_to_exit_behavior(df, trades)
    sl_trades, drop_df = analyze_sl_hits(df, trades)
    failed_tp1, early_df = analyze_false_bottom_signals(df, trades)
    analyze_time_stop_trades(df, trades)
    analyze_winning_trades(df, trades)
    
    # 최종 요약
    final_summary()
    
    # 상세 분석 결과 저장
    analysis_df.to_csv('failure_analysis_detailed.csv', index=False)
    print(f"\n✅ 상세 분석 저장: failure_analysis_detailed.csv")


if __name__ == "__main__":
    main()
