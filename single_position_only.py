import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 데이터 로드
df_15m = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df_4h = pd.read_csv('btc_4h_ohlcv.csv', parse_dates=['datetime'])
df_15m = df_15m.rename(columns={'datetime': 'timestamp'})
df_4h = df_4h.rename(columns={'datetime': 'timestamp'})

print("=" * 70)
print("🎯 동시 포지션 없음 (1개 포지션만) - 현실적 백테스트")
print("=" * 70)

# 전략 설정
TP1 = 0.02      # 2% (본절로 이동)
TP2 = 0.035     # 3.5%
SL = -0.015     # -1.5%
TIME_STOP = 96  # 24시간 (15분 * 96 = 24시간)

# FVG 감지 함수
def detect_fvg_signals(df_4h):
    signals = []
    
    for i in range(2, len(df_4h)):
        # Bullish FVG: 갭 상승
        if df_4h['low'].iloc[i] > df_4h['high'].iloc[i-2]:
            gap_top = df_4h['low'].iloc[i]
            gap_bottom = df_4h['high'].iloc[i-2]
            gap_size = (gap_top - gap_bottom) / gap_bottom * 100
            
            if gap_size >= 0.3:
                signal_time = df_4h['timestamp'].iloc[i]
                signals.append({
                    'type': 'FVG_bull',
                    'direction': 'long',
                    'signal_time': signal_time,
                    'entry_zone': gap_top,
                    'gap_size': gap_size
                })
        
        # Bearish FVG: 갭 하락
        if df_4h['high'].iloc[i] < df_4h['low'].iloc[i-2]:
            gap_top = df_4h['low'].iloc[i-2]
            gap_bottom = df_4h['high'].iloc[i]
            gap_size = (gap_top - gap_bottom) / gap_bottom * 100
            
            if gap_size >= 0.3:
                signal_time = df_4h['timestamp'].iloc[i]
                signals.append({
                    'type': 'FVG_bear',
                    'direction': 'short',
                    'signal_time': signal_time,
                    'entry_zone': gap_bottom,
                    'gap_size': gap_size
                })
    
    return signals

# Order Block 감지 함수
def detect_orderblock_signals(df_4h):
    signals = []
    
    for i in range(3, len(df_4h)):
        # Bullish Order Block: 음봉 후 강한 양봉 돌파
        if (df_4h['close'].iloc[i-2] < df_4h['open'].iloc[i-2] and
            df_4h['close'].iloc[i-1] > df_4h['open'].iloc[i-1] and
            df_4h['close'].iloc[i-1] > df_4h['high'].iloc[i-2]):
            
            ob_high = df_4h['high'].iloc[i-2]
            signal_time = df_4h['timestamp'].iloc[i-1]
            
            signals.append({
                'type': 'orderblock',
                'direction': 'long',
                'signal_time': signal_time,
                'entry_zone': ob_high,
            })
    
    return signals

# 시그널 감지
fvg_signals = detect_fvg_signals(df_4h)
ob_signals = detect_orderblock_signals(df_4h)
all_signals = fvg_signals + ob_signals

print(f"\n📊 감지된 시그널:")
print(f"   - FVG Bull: {len([s for s in fvg_signals if s['direction']=='long'])}개")
print(f"   - FVG Bear: {len([s for s in fvg_signals if s['direction']=='short'])}개")
print(f"   - Order Block: {len(ob_signals)}개")
print(f"   - 총 시그널: {len(all_signals)}개")

# 단일 포지션 시뮬레이션 (캔들 단위 순차 처리)
def simulate_single_position(signals, df_15m):
    """동시 포지션 없이 1개씩만 거래 - 캔들 단위로 정확하게"""
    trades = []
    pending_signals = []
    current_position = None
    
    signals_sorted = sorted(signals, key=lambda x: x['signal_time'])
    signal_idx = 0
    
    for i, candle in df_15m.iterrows():
        candle_time = candle['timestamp']
        
        # 새 시그널 추가
        while signal_idx < len(signals_sorted) and signals_sorted[signal_idx]['signal_time'] <= candle_time:
            sig = signals_sorted[signal_idx]
            expire_time = sig['signal_time'] + timedelta(hours=5)  # 20바 유효
            pending_signals.append({**sig, 'expire_time': expire_time})
            signal_idx += 1
        
        # 만료 시그널 제거
        pending_signals = [s for s in pending_signals if s['expire_time'] > candle_time]
        
        # 현재 포지션 관리
        if current_position is not None:
            pos = current_position
            bars_elapsed = (candle_time - pos['entry_time']).total_seconds() / 900
            closed = False
            
            if pos['direction'] == 'long':
                # TP1 체크
                if not pos['tp1_hit']:
                    tp1_price = pos['entry_price'] * (1 + TP1)
                    if candle['high'] >= tp1_price:
                        pos['tp1_hit'] = True
                        pos['sl_price'] = pos['entry_price']
                
                # TP2 체크
                tp2_price = pos['entry_price'] * (1 + TP2)
                if candle['high'] >= tp2_price:
                    pos['exit_price'] = tp2_price
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TP2'
                    pos['pnl'] = TP2 * 100
                    closed = True
                elif candle['low'] <= pos['sl_price']:
                    pos['exit_price'] = pos['sl_price']
                    pos['exit_time'] = candle_time
                    pos['result'] = 'BE' if pos['tp1_hit'] else 'SL'
                    pos['pnl'] = 0 if pos['tp1_hit'] else SL * 100
                    closed = True
                elif bars_elapsed >= TIME_STOP:
                    pos['exit_price'] = candle['close']
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TIME'
                    pos['pnl'] = (candle['close'] - pos['entry_price']) / pos['entry_price'] * 100
                    closed = True
            
            else:  # short
                if not pos['tp1_hit']:
                    tp1_price = pos['entry_price'] * (1 - TP1)
                    if candle['low'] <= tp1_price:
                        pos['tp1_hit'] = True
                        pos['sl_price'] = pos['entry_price']
                
                tp2_price = pos['entry_price'] * (1 - TP2)
                if candle['low'] <= tp2_price:
                    pos['exit_price'] = tp2_price
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TP2'
                    pos['pnl'] = TP2 * 100
                    closed = True
                elif candle['high'] >= pos['sl_price']:
                    pos['exit_price'] = pos['sl_price']
                    pos['exit_time'] = candle_time
                    pos['result'] = 'BE' if pos['tp1_hit'] else 'SL'
                    pos['pnl'] = 0 if pos['tp1_hit'] else SL * 100
                    closed = True
                elif bars_elapsed >= TIME_STOP:
                    pos['exit_price'] = candle['close']
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TIME'
                    pos['pnl'] = (pos['entry_price'] - candle['close']) / pos['entry_price'] * 100
                    closed = True
            
            if closed:
                trades.append(pos)
                current_position = None
        
        # 새 진입 (포지션 없을 때만)
        if current_position is None and pending_signals:
            for sig in pending_signals[:]:
                entry_zone = sig['entry_zone']
                direction = sig['direction']
                
                touched = False
                if direction == 'long' and candle['low'] <= entry_zone:
                    touched = True
                elif direction == 'short' and candle['high'] >= entry_zone:
                    touched = True
                
                if touched:
                    current_position = {
                        'strategy': sig['type'],
                        'direction': direction,
                        'signal_time': sig['signal_time'],
                        'entry_time': candle_time,
                        'entry_price': entry_zone,
                        'sl_price': entry_zone * (1 + SL) if direction == 'long' else entry_zone * (1 - SL),
                        'tp1_hit': False
                    }
                    pending_signals.remove(sig)
                    break
    
    return trades

# 시뮬레이션 실행
print("\n⏳ 단일 포지션 시뮬레이션 실행 중...")
trades = simulate_single_position(all_signals, df_15m)

# 결과 분석
df_trades = pd.DataFrame(trades)
print(f"\n{'='*70}")
print("📈 단일 포지션 백테스트 결과 (동시 포지션 없음)")
print(f"{'='*70}")

if len(df_trades) > 0:
    total_trades = len(df_trades)
    
    results = df_trades['result'].value_counts()
    sl_count = results.get('SL', 0)
    tp2_count = results.get('TP2', 0)
    be_count = results.get('BE', 0)
    time_count = results.get('TIME', 0)
    
    sl_rate = sl_count / total_trades * 100
    win_rate = (tp2_count + be_count + time_count) / total_trades * 100
    
    total_pnl = df_trades['pnl'].sum()
    
    first_trade = df_trades['entry_time'].min()
    last_trade = df_trades['exit_time'].max()
    months = (last_trade - first_trade).days / 30
    monthly_pnl = total_pnl / months if months > 0 else 0
    monthly_trades = total_trades / months if months > 0 else 0
    
    print(f"\n📊 전체 성과:")
    print(f"   총 거래: {total_trades}회")
    print(f"   월평균 거래: {monthly_trades:.1f}회")
    print(f"   SL 비율: {sl_rate:.1f}%")
    print(f"   승률: {win_rate:.1f}%")
    print(f"   총 수익: {total_pnl:.1f}%")
    print(f"   월평균 수익: {monthly_pnl:.2f}%")
    
    print(f"\n📋 결과 분포:")
    print(f"   TP2 (목표가): {tp2_count}회 ({tp2_count/total_trades*100:.1f}%)")
    print(f"   BE (본절): {be_count}회 ({be_count/total_trades*100:.1f}%)")
    print(f"   TIME (시간스탑): {time_count}회 ({time_count/total_trades*100:.1f}%)")
    print(f"   SL (손절): {sl_count}회 ({sl_count/total_trades*100:.1f}%)")
    
    # 전략별 성과
    print(f"\n📈 전략별 성과:")
    for strategy in df_trades['strategy'].unique():
        strat_df = df_trades[df_trades['strategy'] == strategy]
        strat_trades = len(strat_df)
        strat_pnl = strat_df['pnl'].sum()
        strat_monthly = strat_pnl / months if months > 0 else 0
        strat_sl = len(strat_df[strat_df['result'] == 'SL']) / strat_trades * 100
        print(f"   {strategy}: {strat_trades}회, SL {strat_sl:.1f}%, 월평균 {strat_monthly:.2f}%")
    
    # 연도별 성과
    print(f"\n📅 연도별 성과:")
    df_trades['year'] = pd.to_datetime(df_trades['entry_time']).dt.year
    for year in sorted(df_trades['year'].unique()):
        year_df = df_trades[df_trades['year'] == year]
        year_trades = len(year_df)
        year_pnl = year_df['pnl'].sum()
        year_months = 12 if year < 2025 else (last_trade.month if hasattr(last_trade, 'month') else 11)
        year_monthly = year_pnl / year_months
        year_sl = len(year_df[year_df['result'] == 'SL']) / year_trades * 100 if year_trades > 0 else 0
        print(f"   {year}: {year_trades}회, SL {year_sl:.1f}%, 총 {year_pnl:.1f}%, 월평균 {year_monthly:.2f}%")
    
    # MDD
    cumulative = df_trades['pnl'].cumsum()
    peak = cumulative.expanding().max()
    drawdown = cumulative - peak
    mdd = drawdown.min()
    print(f"\n📉 최대 낙폭 (MDD): {mdd:.1f}%")

print(f"\n{'='*70}")
print("✅ 검증 완료: 동시 포지션 없음 (1개씩만 거래)")
print(f"{'='*70}")
