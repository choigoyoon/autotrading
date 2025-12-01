#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Floor Catch Strategy Backtest
L값 바닥 캐치 전략 백테스트

전략 로직:
1. Swing Low (L값) 감지 (10봉 좌우 확인)
2. LL 연속 카운트 (3-4번 또는 5번 이상)
3. 극과매도 조건 확인 (RSI < 30, MACD Hist < -50, Volume Ratio > 3.0, ATR % > 0.5, BB Position < 0.1)
4. 첫 양봉 + RSI 회복 대기
5. Long 진입
6. TP1 +1.5% (50%), TP2 +3.5% (50%), SL L값 -0.5%, Time Stop 24h
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
    
    # Volume Ratio
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
    """Swing Low (L값) 감지 - 리페인팅 없이 실시간 감지 가능"""
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
                'confirmed_index': i + right_bars  # 실제 확인 시점
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


def check_extreme_oversold(df, idx):
    """극과매도 조건 체크 (L값 발생 시점)"""
    conditions_met = 0
    
    if df.iloc[idx]['rsi'] < 30:
        conditions_met += 1
    if df.iloc[idx]['macd_hist'] < -50:
        conditions_met += 1
    if df.iloc[idx]['bb_position'] < 0.1:
        conditions_met += 1
    if df.iloc[idx]['volume_ratio'] > 3.0:
        conditions_met += 1
    if df.iloc[idx]['atr_pct'] > 0.5:
        conditions_met += 1
    
    return conditions_met


def check_first_green_candle(df, start_idx):
    """첫 양봉 및 RSI 회복 신호 대기"""
    # 최대 20봉까지만 대기
    for offset in range(1, 21):
        idx = start_idx + offset
        if idx >= len(df):
            return None, None
        
        # 양봉 체크 (Close > Open, 크기 > 0.3%)
        is_green = df.iloc[idx]['close'] > df.iloc[idx]['open']
        candle_size = (df.iloc[idx]['close'] - df.iloc[idx]['open']) / df.iloc[idx]['open'] * 100
        
        # RSI 회복 신호 (RSI > 35 또는 MACD Hist 상승)
        rsi_recovery = df.iloc[idx]['rsi'] > 35
        macd_recovery = df.iloc[idx]['macd_hist'] > df.iloc[start_idx]['macd_hist']
        
        if is_green and candle_size > 0.3 and (rsi_recovery or macd_recovery):
            return idx, df.iloc[idx]['close']
    
    return None, None


def backtest_floor_catch(df, swing_lows):
    """바닥 캐치 전략 백테스트"""
    trades = []
    
    for sl in swing_lows:
        # LL 연속 카운트가 3 이상인 경우만 (3-4번 또는 5번 이상)
        if sl['consecutive_ll'] < 3:
            continue
        
        l_idx = sl['confirmed_index']  # L값 확정 시점
        if l_idx >= len(df):
            continue
        
        # 극과매도 조건 체크 (L값 시점)
        conditions_met = check_extreme_oversold(df, sl['index'])
        if conditions_met < 4:  # 최소 4개 조건 만족
            continue
        
        # 첫 양봉 + RSI 회복 대기
        entry_idx, entry_price = check_first_green_candle(df, l_idx)
        if entry_idx is None:
            continue
        
        # 진입
        entry_time = df.iloc[entry_idx]['datetime']
        l_price = sl['price']
        sl_price = l_price * 0.995  # L값 -0.5%
        tp1_price = entry_price * 1.015  # +1.5%
        tp2_price = entry_price * 1.035  # +3.5%
        time_stop = entry_time + timedelta(hours=24)
        
        # 포지션 추적
        position = 1.0  # 100% 포지션
        tp1_hit = False
        exit_idx = None
        exit_price = None
        exit_type = None
        
        for i in range(entry_idx + 1, len(df)):
            current_time = df.iloc[i]['datetime']
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            current_close = df.iloc[i]['close']
            
            # SL 체크
            if current_low <= sl_price:
                exit_idx = i
                exit_price = sl_price
                exit_type = 'SL'
                break
            
            # TP1 체크 (50% 청산)
            if not tp1_hit and current_high >= tp1_price:
                tp1_hit = True
                position = 0.5
            
            # TP2 체크 (나머지 50% 청산)
            if tp1_hit and current_high >= tp2_price:
                exit_idx = i
                exit_price = tp2_price
                exit_type = 'TP2'
                break
            
            # Time Stop 체크
            if current_time >= time_stop:
                exit_idx = i
                exit_price = current_close
                exit_type = 'TIME'
                break
        
        # 데이터 끝까지 도달
        if exit_idx is None:
            exit_idx = len(df) - 1
            exit_price = df.iloc[exit_idx]['close']
            exit_type = 'END'
        
        # PNL 계산
        if tp1_hit:
            pnl_tp1 = (tp1_price - entry_price) / entry_price * 100 * 0.5
            pnl_tp2 = (exit_price - entry_price) / entry_price * 100 * 0.5
            total_pnl = pnl_tp1 + pnl_tp2
        else:
            total_pnl = (exit_price - entry_price) / entry_price * 100
        
        trades.append({
            'entry_time': entry_time,
            'entry_price': entry_price,
            'l_price': l_price,
            'sl_price': sl_price,
            'tp1_price': tp1_price,
            'tp2_price': tp2_price,
            'exit_time': df.iloc[exit_idx]['datetime'],
            'exit_price': exit_price,
            'exit_type': exit_type,
            'tp1_hit': tp1_hit,
            'consecutive_ll': sl['consecutive_ll'],
            'conditions_met': conditions_met,
            'pnl_pct': total_pnl,
            'duration_hours': (df.iloc[exit_idx]['datetime'] - entry_time).total_seconds() / 3600
        })
    
    return trades


def main():
    print("=" * 80)
    print("Floor Catch Strategy Backtest - L값 바닥 캐치 전략")
    print("=" * 80)
    
    # 데이터 로드
    print("\n[1] Loading Data...")
    df = pd.read_csv('btc_15m_ohlcv.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"Data Period: {df.iloc[0]['datetime']} ~ {df.iloc[-1]['datetime']}")
    print(f"Total Candles: {len(df):,}")
    
    # 지표 계산
    print("\n[2] Calculating Indicators...")
    df = calculate_indicators(df)
    
    # Swing Low 감지
    print("\n[3] Detecting Swing Lows...")
    swing_lows = detect_swing_lows(df, left_bars=10, right_bars=10)
    print(f"Total Swing Lows Detected: {len(swing_lows)}")
    
    # LL/HL 패턴 분류
    print("\n[4] Classifying LL/HL Patterns...")
    swing_lows = classify_swing_low_pattern(swing_lows)
    
    ll_counts = {}
    for sl in swing_lows:
        count = sl['consecutive_ll']
        ll_counts[count] = ll_counts.get(count, 0) + 1
    
    print("\nConsecutive LL Distribution:")
    for count in sorted(ll_counts.keys()):
        if count > 0:
            print(f"  {count}번 연속 LL: {ll_counts[count]}회")
    
    # 백테스트 실행
    print("\n[5] Running Backtest...")
    trades = backtest_floor_catch(df, swing_lows)
    
    print(f"\n총 거래 수 (Total Trades): {len(trades)}")
    
    if len(trades) == 0:
        print("\n⚠️  조건을 만족하는 거래가 없습니다.")
        return
    
    # 결과 분석
    print("\n" + "=" * 80)
    print("전략 성과 분석")
    print("=" * 80)
    
    trades_df = pd.DataFrame(trades)
    
    # 기본 통계
    print(f"\n[거래 통계]")
    print(f"총 거래 수: {len(trades_df)}")
    print(f"승리 거래 (PNL > 0): {len(trades_df[trades_df['pnl_pct'] > 0])} ({len(trades_df[trades_df['pnl_pct'] > 0]) / len(trades_df) * 100:.1f}%)")
    print(f"패배 거래 (PNL < 0): {len(trades_df[trades_df['pnl_pct'] < 0])} ({len(trades_df[trades_df['pnl_pct'] < 0]) / len(trades_df) * 100:.1f}%)")
    
    # 수익 통계
    print(f"\n[수익성 분석]")
    print(f"평균 PNL: {trades_df['pnl_pct'].mean():.2f}%")
    print(f"중앙값 PNL: {trades_df['pnl_pct'].median():.2f}%")
    print(f"총 누적 PNL: {trades_df['pnl_pct'].sum():.2f}%")
    print(f"최대 수익: {trades_df['pnl_pct'].max():.2f}%")
    print(f"최대 손실: {trades_df['pnl_pct'].min():.2f}%")
    
    # 청산 타입별
    print(f"\n[청산 타입별 분포]")
    exit_types = trades_df['exit_type'].value_counts()
    for exit_type, count in exit_types.items():
        pct = count / len(trades_df) * 100
        avg_pnl = trades_df[trades_df['exit_type'] == exit_type]['pnl_pct'].mean()
        print(f"  {exit_type}: {count}회 ({pct:.1f}%) - 평균 PNL: {avg_pnl:.2f}%")
    
    # TP1 성공률
    tp1_success = len(trades_df[trades_df['tp1_hit'] == True])
    print(f"\n[TP1 달성률]")
    print(f"TP1 달성: {tp1_success}회 ({tp1_success / len(trades_df) * 100:.1f}%)")
    
    # 연속 LL별 성과
    print(f"\n[연속 LL 카운트별 성과]")
    for ll_count in sorted(trades_df['consecutive_ll'].unique()):
        subset = trades_df[trades_df['consecutive_ll'] == ll_count]
        win_rate = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
        avg_pnl = subset['pnl_pct'].mean()
        print(f"  {ll_count}번 연속 LL: {len(subset)}회 거래 | 승률 {win_rate:.1f}% | 평균 PNL {avg_pnl:.2f}%")
    
    # 조건 수별 성과
    print(f"\n[조건 충족 개수별 성과]")
    for cond_count in sorted(trades_df['conditions_met'].unique()):
        subset = trades_df[trades_df['conditions_met'] == cond_count]
        win_rate = len(subset[subset['pnl_pct'] > 0]) / len(subset) * 100
        avg_pnl = subset['pnl_pct'].mean()
        print(f"  {cond_count}개 조건: {len(subset)}회 거래 | 승률 {win_rate:.1f}% | 평균 PNL {avg_pnl:.2f}%")
    
    # 평균 거래 지속시간
    print(f"\n[거래 지속시간]")
    print(f"평균 거래 시간: {trades_df['duration_hours'].mean():.1f}시간")
    print(f"중앙값 거래 시간: {trades_df['duration_hours'].median():.1f}시간")
    
    # 월별 거래 수
    trades_df['entry_month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
    monthly_trades = trades_df.groupby('entry_month').size()
    print(f"\n[월별 평균 거래 수]")
    print(f"평균: {monthly_trades.mean():.1f}회/월")
    print(f"최대: {monthly_trades.max()}회/월")
    print(f"최소: {monthly_trades.min()}회/월")
    
    # 결과 저장
    output_file = 'floor_catch_strategy_results.csv'
    trades_df.to_csv(output_file, index=False)
    print(f"\n✅ 결과 저장 완료: {output_file}")
    
    # 최종 결론
    print("\n" + "=" * 80)
    print("최종 결론")
    print("=" * 80)
    
    total_months = (df.iloc[-1]['datetime'] - df.iloc[0]['datetime']).days / 30
    avg_monthly_trades = len(trades_df) / total_months
    avg_monthly_pnl = trades_df['pnl_pct'].sum() / total_months
    
    print(f"\n📊 전체 기간 성과 요약:")
    print(f"  - 백테스트 기간: {total_months:.1f}개월")
    print(f"  - 총 거래 수: {len(trades_df)}회")
    print(f"  - 월평균 거래: {avg_monthly_trades:.1f}회")
    print(f"  - 총 누적 PNL: {trades_df['pnl_pct'].sum():.2f}%")
    print(f"  - 월평균 PNL: {avg_monthly_pnl:.2f}%")
    print(f"  - 실전 승률: {len(trades_df[trades_df['pnl_pct'] > 0]) / len(trades_df) * 100:.1f}%")
    
    if avg_monthly_pnl > 0:
        print(f"\n✅ 전략이 수익성이 있습니다 (+{avg_monthly_pnl:.2f}%/월)")
    else:
        print(f"\n⚠️  전략이 손실을 기록했습니다 ({avg_monthly_pnl:.2f}%/월)")


if __name__ == "__main__":
    main()
