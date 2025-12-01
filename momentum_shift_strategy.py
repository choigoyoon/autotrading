#!/usr/bin/env python3
"""
힘의 변환 (Momentum Shift) 전략

핵심: 하락 힘 → 상승 힘으로 전환되는 순간 포착
확인: 선 (수평선+추세선) + 캔들 (봉마감) + BB (볼린저밴드)
"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    볼린저밴드 계산
    """
    df = df.copy()
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    df['bb_std'] = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
    
    return df


def find_horizontal_levels(df, L_points, H_points, tolerance_pct=0.5):
    """
    수평선 (저항/지지선) 찾기
    
    L값과 H값이 여러 번 터치한 가격대 = 수평선
    """
    print("="*80)
    print("Step 1: 수평선 (저항/지지선) 찾기")
    print("="*80)
    
    # L값 기반 지지선
    L_levels = []
    for i in range(len(L_points) - 5):
        L_group = L_points.iloc[i:i+5]['L_value'].values
        L_mean = L_group.mean()
        L_std = L_group.std()
        
        # 가격이 비슷한 범위에 밀집 = 수평선
        if L_std / L_mean * 100 <= tolerance_pct:
            L_levels.append({
                'level': L_mean,
                'type': 'support',
                'touch_count': 5,
                'first_touch': L_points.iloc[i]['datetime'],
                'last_touch': L_points.iloc[i+4]['datetime']
            })
    
    # H값 기반 저항선
    H_levels = []
    for i in range(len(H_points) - 5):
        H_group = H_points.iloc[i:i+5]['H_value'].values
        H_mean = H_group.mean()
        H_std = H_group.std()
        
        if H_std / H_mean * 100 <= tolerance_pct:
            H_levels.append({
                'level': H_mean,
                'type': 'resistance',
                'touch_count': 5,
                'first_touch': H_points.iloc[i]['datetime'],
                'last_touch': H_points.iloc[i+4]['datetime']
            })
    
    print(f"\n지지선 (Support): {len(L_levels)}개")
    print(f"저항선 (Resistance): {len(H_levels)}개")
    
    return L_levels, H_levels


def detect_horizontal_break(df, levels, level_type='resistance'):
    """
    수평선 돌파 감지
    
    저항선 → 지지선 전환 = 상승 신호
    """
    print(f"\nStep 2: {level_type} 돌파 감지")
    print("="*80)
    
    breakouts = []
    
    for level_info in levels:
        level = level_info['level']
        
        # 해당 수평선 이후 데이터만 확인
        df_after = df[df['datetime'] > level_info['last_touch']].copy()
        
        for i in range(1, len(df_after)):
            prev_close = df_after.iloc[i-1]['close']
            curr_close = df_after.iloc[i]['close']
            
            # 저항선 돌파 (아래 → 위)
            if level_type == 'resistance':
                if prev_close < level and curr_close > level:
                    breakouts.append({
                        'datetime': df_after.iloc[i]['datetime'],
                        'level': level,
                        'type': '저항선 돌파 (상승)',
                        'prev_close': prev_close,
                        'curr_close': curr_close,
                        'breakout_pct': (curr_close - level) / level * 100
                    })
                    break
            
            # 지지선 이탈 (위 → 아래)
            elif level_type == 'support':
                if prev_close > level and curr_close < level:
                    breakouts.append({
                        'datetime': df_after.iloc[i]['datetime'],
                        'level': level,
                        'type': '지지선 이탈 (하락)',
                        'prev_close': prev_close,
                        'curr_close': curr_close,
                        'breakout_pct': (curr_close - level) / level * 100
                    })
                    break
    
    print(f"{level_type} 돌파 감지: {len(breakouts)}개")
    
    return breakouts


def detect_bb_signals(df):
    """
    볼린저밴드 신호 감지
    
    1. BB 하단 이탈 후 양봉 마감 = 반전 신호
    2. BB 상단 돌파 = 강한 상승
    3. BB 찢고 마감 = 극단적 힘
    """
    print("\nStep 3: 볼린저밴드 신호 감지")
    print("="*80)
    
    df = df.copy()
    bb_signals = []
    
    for i in range(1, len(df)):
        prev_row = df.iloc[i-1]
        curr_row = df.iloc[i]
        
        # 1. BB 하단 이탈 후 양봉 마감 (반전 신호)
        if (prev_row['close'] < prev_row['bb_lower'] and 
            curr_row['close'] > curr_row['open'] and
            curr_row['close'] > prev_row['close']):
            
            bb_signals.append({
                'datetime': curr_row['datetime'],
                'type': 'BB 하단 반등 (역추세)',
                'bb_lower': curr_row['bb_lower'],
                'close': curr_row['close'],
                'signal_strength': (curr_row['close'] - curr_row['bb_lower']) / curr_row['bb_lower'] * 100
            })
        
        # 2. BB 상단 돌파 (강한 상승)
        if (prev_row['close'] < prev_row['bb_upper'] and 
            curr_row['close'] > curr_row['bb_upper']):
            
            bb_signals.append({
                'datetime': curr_row['datetime'],
                'type': 'BB 상단 돌파 (추세확정)',
                'bb_upper': curr_row['bb_upper'],
                'close': curr_row['close'],
                'signal_strength': (curr_row['close'] - curr_row['bb_upper']) / curr_row['bb_upper'] * 100
            })
        
        # 3. BB 찢고 마감 (극단적 힘)
        if curr_row['low'] < curr_row['bb_lower'] and curr_row['close'] > curr_row['bb_lower']:
            bb_signals.append({
                'datetime': curr_row['datetime'],
                'type': 'BB 하단 찢고 복귀',
                'bb_lower': curr_row['bb_lower'],
                'low': curr_row['low'],
                'close': curr_row['close'],
                'signal_strength': (curr_row['close'] - curr_row['low']) / curr_row['low'] * 100
            })
    
    print(f"BB 신호 감지: {len(bb_signals)}개")
    
    if len(bb_signals) > 0:
        bb_df = pd.DataFrame(bb_signals)
        print("\nBB 신호 타입별 통계:")
        print(bb_df['type'].value_counts())
    
    return bb_signals


def detect_trendline_with_bb(df, H_points, L_points):
    """
    추세선 돌파 + BB 조합 신호
    
    1. 하락추세선 돌파 + BB 하단 반등 = 추세전환 진입
    2. 상승추세선 형성 + BB 상단 돌파 = 추세확정 진입
    """
    print("\nStep 4: 추세선 + BB 조합 신호")
    print("="*80)
    
    combo_signals = []
    
    # 하락 추세선 돌파 찾기
    for i in range(2, len(H_points) - 5):
        H1 = H_points.iloc[i-2]['H_value']
        H2 = H_points.iloc[i-1]['H_value']
        H3 = H_points.iloc[i]['H_value']
        
        # 하락 추세선 (LH-LH-LH)
        if H2 < H1 and H3 < H2:
            time_after = H_points.iloc[i]['datetime']
            
            # 추세선 돌파 확인
            future_data = df[df['datetime'] > time_after].head(40)
            
            for j, row in future_data.iterrows():
                # 추세선 돌파
                if row['close'] > H3:
                    # 같은 시간대 BB 신호 확인
                    if row['close'] > row['bb_lower']:
                        signal_type = 'unknown'
                        
                        # BB 하단 반등 + 추세선 돌파 = 추세전환
                        if row['low'] < row['bb_lower'] and row['close'] > row['open']:
                            signal_type = '추세전환 진입 (하락추세선 돌파 + BB 반등)'
                        
                        # BB 상단 돌파 + 추세선 돌파 = 강한 상승
                        elif row['close'] > row['bb_upper']:
                            signal_type = '추세확정 진입 (하락추세선 돌파 + BB 상단 돌파)'
                        
                        # 일반 돌파
                        else:
                            signal_type = '일반 돌파 (하락추세선 돌파)'
                        
                        combo_signals.append({
                            'datetime': row['datetime'],
                            'type': signal_type,
                            'trendline_H3': H3,
                            'close': row['close'],
                            'bb_lower': row['bb_lower'],
                            'bb_upper': row['bb_upper'],
                            'breakout_pct': (row['close'] - H3) / H3 * 100
                        })
                        break
    
    print(f"추세선 + BB 조합 신호: {len(combo_signals)}개")
    
    if len(combo_signals) > 0:
        combo_df = pd.DataFrame(combo_signals)
        print("\n조합 신호 타입별 통계:")
        print(combo_df['type'].value_counts())
    
    return combo_signals


def backtest_momentum_shift(df, combo_signals):
    """
    힘의 변환 전략 백테스트
    
    역추세: BB 하단 + 양봉 → 익절 1:1
    추세전환: 수평선 위 마감 → 익절 1:2
    추세확정: 추세선 돌파 + BB 상단 → 익절 1:2~1:3
    """
    print("\nStep 5: 백테스트 (힘의 변환 전략)")
    print("="*80)
    
    trades = []
    
    for signal in combo_signals:
        entry_time = signal['datetime']
        entry_price = signal['close']
        signal_type = signal['type']
        
        # 신호 타입별 익절/손절 설정
        if '추세전환' in signal_type:
            # 추세전환: 1:2
            tp_pct = 2.0
            sl_pct = -1.0
        elif '추세확정' in signal_type:
            # 추세확정: 1:3
            tp_pct = 3.0
            sl_pct = -1.0
        else:
            # 일반 돌파: 1:1
            tp_pct = 1.0
            sl_pct = -1.0
        
        tp_price = entry_price * (1 + tp_pct / 100)
        sl_price = entry_price * (1 + sl_pct / 100)
        
        # 진입 이후 데이터
        future_data = df[df['datetime'] > entry_time].head(96)  # 24시간
        
        exit_type = 'Time Stop'
        exit_price = future_data.iloc[-1]['close'] if len(future_data) > 0 else entry_price
        exit_time = future_data.iloc[-1]['datetime'] if len(future_data) > 0 else entry_time
        
        for idx, row in future_data.iterrows():
            # TP 도달
            if row['high'] >= tp_price:
                exit_type = 'TP'
                exit_price = tp_price
                exit_time = row['datetime']
                break
            
            # SL 도달
            if row['low'] <= sl_price:
                exit_type = 'SL'
                exit_price = sl_price
                exit_time = row['datetime']
                break
        
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        
        trades.append({
            'entry_time': entry_time,
            'entry_price': entry_price,
            'signal_type': signal_type,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_type': exit_type,
            'pnl_pct': pnl_pct
        })
    
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) > 0:
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        win_rate = winning_trades / total_trades * 100
        avg_pnl = trades_df['pnl_pct'].mean()
        total_pnl = trades_df['pnl_pct'].sum()
        
        print(f"\n총 거래 수: {total_trades}")
        print(f"승률: {win_rate:.1f}%")
        print(f"평균 수익: {avg_pnl:.2f}%")
        print(f"누적 수익: {total_pnl:.2f}%")
        
        print("\n청산 타입별 통계:")
        print(trades_df['exit_type'].value_counts())
        
        print("\n신호 타입별 통계:")
        for signal_type in trades_df['signal_type'].unique():
            signal_trades = trades_df[trades_df['signal_type'] == signal_type]
            print(f"\n{signal_type}:")
            print(f"  거래 수: {len(signal_trades)}")
            print(f"  승률: {len(signal_trades[signal_trades['pnl_pct'] > 0]) / len(signal_trades) * 100:.1f}%")
            print(f"  평균 수익: {signal_trades['pnl_pct'].mean():.2f}%")
    
    return trades_df


def main():
    print("="*80)
    print("힘의 변환 (Momentum Shift) 전략")
    print("="*80)
    
    # 데이터 로드
    df = pd.read_csv('btc_15m_ohlcv.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # L값, H값 로드
    L_points = pd.read_csv('all_L_values.csv')
    L_points['datetime'] = pd.to_datetime(L_points['datetime'])
    
    H_points = pd.read_csv('all_H_values.csv')
    H_points['datetime'] = pd.to_datetime(H_points['datetime'])
    
    print(f"\n데이터 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"총 캔들 수: {len(df):,}")
    print(f"L값: {len(L_points):,}개")
    print(f"H값: {len(H_points):,}개")
    
    # 볼린저밴드 계산
    df = calculate_bollinger_bands(df)
    
    # Step 1: 수평선 찾기
    support_levels, resistance_levels = find_horizontal_levels(df, L_points, H_points)
    
    # Step 2: 수평선 돌파 감지
    resistance_breaks = detect_horizontal_break(df, resistance_levels, 'resistance')
    
    # Step 3: BB 신호 감지
    bb_signals = detect_bb_signals(df)
    
    # Step 4: 추세선 + BB 조합 신호
    combo_signals = detect_trendline_with_bb(df, H_points, L_points)
    
    # Step 5: 백테스트
    trades_df = backtest_momentum_shift(df, combo_signals)
    
    # 결과 저장
    if len(trades_df) > 0:
        trades_df.to_csv('momentum_shift_trades.csv', index=False)
        print("\n✅ 결과 저장: momentum_shift_trades.csv")
    
    pd.DataFrame(bb_signals).to_csv('bb_signals.csv', index=False)
    print("✅ 결과 저장: bb_signals.csv")
    
    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80)


if __name__ == "__main__":
    main()
