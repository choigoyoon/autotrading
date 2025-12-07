#!/usr/bin/env python3
"""
힘의 변환 (Momentum Shift) 전략 - 간단 버전

핵심: 선 + 캔들 + BB 조합
"""

import pandas as pd
import numpy as np

def main():
    print("="*80)
    print("힘의 변환 (Momentum Shift) 전략")
    print("="*80)
    
    # 데이터 로드 (최근 2년)
    df = pd.read_csv('btc_15m_ohlcv.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['datetime'] >= '2023-01-01'].copy().reset_index(drop=True)
    
    print(f"\n데이터 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"총 캔들 수: {len(df):,}")
    
    # 볼린저밴드 계산
    print("\nStep 1: 볼린저밴드 계산 중...")
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    
    # L값, H값 로드
    L_points = pd.read_csv('all_L_values.csv')
    L_points['datetime'] = pd.to_datetime(L_points['datetime'])
    L_points = L_points[L_points['datetime'] >= '2023-01-01']
    
    H_points = pd.read_csv('all_H_values.csv')
    H_points['datetime'] = pd.to_datetime(H_points['datetime'])
    H_points = H_points[H_points['datetime'] >= '2023-01-01']
    
    print(f"L값: {len(L_points)}개")
    print(f"H값: {len(H_points)}개")
    
    # Step 2: 힘의 변환 신호 감지
    print("\nStep 2: 힘의 변환 신호 감지 중...")
    
    signals = []
    
    for i in range(21, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        
        # 신호 1: BB 하단 이탈 후 양봉 마감 (역추세 반등)
        if (prev['close'] < prev['bb_lower'] and 
            curr['close'] > curr['open'] and
            curr['close'] > prev['close']):
            
            signals.append({
                'datetime': curr['datetime'],
                'type': '역추세 반등 (BB 하단)',
                'entry_price': curr['close'],
                'bb_lower': curr['bb_lower'],
                'tp_ratio': 1.0,  # 1:1
                'sl_ratio': 1.0
            })
        
        # 신호 2: BB 상단 돌파 (추세 확정)
        if (prev['close'] < prev['bb_upper'] and 
            curr['close'] > curr['bb_upper']):
            
            signals.append({
                'datetime': curr['datetime'],
                'type': '추세확정 (BB 상단 돌파)',
                'entry_price': curr['close'],
                'bb_upper': curr['bb_upper'],
                'tp_ratio': 2.0,  # 1:2
                'sl_ratio': 1.0
            })
        
        # 신호 3: BB 찢고 마감 (강력한 반전)
        if (curr['low'] < curr['bb_lower'] and 
            curr['close'] > curr['bb_lower'] and
            curr['close'] > curr['open']):
            
            signals.append({
                'datetime': curr['datetime'],
                'type': '강력한 반전 (BB 찢고 복귀)',
                'entry_price': curr['close'],
                'bb_lower': curr['bb_lower'],
                'tp_ratio': 1.5,  # 1:1.5
                'sl_ratio': 1.0
            })
    
    print(f"감지된 신호: {len(signals)}개")
    
    signals_df = pd.DataFrame(signals)
    
    if len(signals_df) > 0:
        print("\n신호 타입별 통계:")
        print(signals_df['type'].value_counts())
    
    # Step 3: 백테스트
    print("\nStep 3: 백테스트 중...")
    
    trades = []
    
    for idx, signal in signals_df.iterrows():
        entry_time = signal['datetime']
        entry_price = signal['entry_price']
        signal_type = signal['type']
        tp_ratio = signal['tp_ratio']
        sl_ratio = signal['sl_ratio']
        
        # TP/SL 계산
        tp_pct = tp_ratio
        sl_pct = -sl_ratio
        
        tp_price = entry_price * (1 + tp_pct / 100)
        sl_price = entry_price * (1 + sl_pct / 100)
        
        # 진입 이후 데이터 (24시간)
        future_data = df[df['datetime'] > entry_time].head(96)
        
        if len(future_data) == 0:
            continue
        
        exit_type = 'Time Stop'
        exit_price = future_data.iloc[-1]['close']
        exit_time = future_data.iloc[-1]['datetime']
        
        for _, row in future_data.iterrows():
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
    
    # 결과 출력
    print("\n" + "="*80)
    print("백테스트 결과")
    print("="*80)
    
    if len(trades_df) > 0:
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        win_rate = winning_trades / total_trades * 100
        avg_pnl = trades_df['pnl_pct'].mean()
        total_pnl = trades_df['pnl_pct'].sum()
        
        # 월간 거래 수
        start_date = df['datetime'].min()
        end_date = df['datetime'].max()
        months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        monthly_trades = total_trades / months if months > 0 else 0
        
        print(f"\n총 거래 수: {total_trades}")
        print(f"월간 거래 수: {monthly_trades:.1f}")
        print(f"승률: {win_rate:.1f}%")
        print(f"평균 수익: {avg_pnl:.2f}%")
        print(f"누적 수익: {total_pnl:.2f}%")
        
        print("\n청산 타입별:")
        print(trades_df['exit_type'].value_counts())
        
        print("\n신호 타입별 성과:")
        for signal_type in trades_df['signal_type'].unique():
            signal_trades = trades_df[trades_df['signal_type'] == signal_type]
            signal_wins = len(signal_trades[signal_trades['pnl_pct'] > 0])
            signal_wr = signal_wins / len(signal_trades) * 100 if len(signal_trades) > 0 else 0
            signal_avg = signal_trades['pnl_pct'].mean()
            
            print(f"\n{signal_type}:")
            print(f"  거래 수: {len(signal_trades)}")
            print(f"  승률: {signal_wr:.1f}%")
            print(f"  평균 수익: {signal_avg:.2f}%")
        
        # 상위 10개 수익 트레이드
        print("\n" + "="*80)
        print("TOP 10 수익 트레이드")
        print("="*80)
        
        top10 = trades_df.nlargest(10, 'pnl_pct')
        for i, (idx, trade) in enumerate(top10.iterrows(), 1):
            print(f"\n#{i}: {trade['signal_type']}")
            print(f"  진입: {trade['entry_time']} @ ${trade['entry_price']:.2f}")
            print(f"  청산: {trade['exit_time']} @ ${trade['exit_price']:.2f}")
            print(f"  수익: {trade['pnl_pct']:.2f}% ({trade['exit_type']})")
        
        # 결과 저장
        trades_df.to_csv('momentum_shift_trades.csv', index=False)
        signals_df.to_csv('momentum_shift_signals.csv', index=False)
        
        print("\n" + "="*80)
        print("✅ 결과 저장 완료")
        print("  - momentum_shift_trades.csv")
        print("  - momentum_shift_signals.csv")
        print("="*80)
    
    else:
        print("\n거래 없음")


if __name__ == "__main__":
    main()
