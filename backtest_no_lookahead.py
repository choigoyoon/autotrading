"""
Look-Ahead Bias 수정된 백테스트
- FVG 시그널 발생 후 다음 봉부터 진입 허용
- 미래 정보 사용 방지
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 80)
print("🔧 Look-Ahead Bias 수정된 백테스트")
print("=" * 80)

# 데이터 로드
df_15m = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df_4h = pd.read_csv('btc_4h_ohlcv.csv', parse_dates=['datetime'])
df_15m = df_15m.rename(columns={'datetime': 'timestamp'})
df_4h = df_4h.rename(columns={'datetime': 'timestamp'})

print(f"\n데이터 기간: {df_15m['timestamp'].min()} ~ {df_15m['timestamp'].max()}")
print(f"4H 캔들: {len(df_4h):,}개, 15M 캔들: {len(df_15m):,}개")

# FVG 감지
def detect_fvg_signals(df_4h):
    signals = []
    for i in range(2, len(df_4h) - 1):  # -1: 다음 봉 필요
        current = df_4h.iloc[i]
        prev2 = df_4h.iloc[i-2]
        next_candle = df_4h.iloc[i+1]  # 다음 봉
        
        # Bullish FVG
        if current['low'] > prev2['high']:
            gap_size = (current['low'] - prev2['high']) / prev2['high'] * 100
            if gap_size >= 0.3:
                signals.append({
                    'type': 'FVG_bull', 
                    'direction': 'long',
                    'detect_time': current['timestamp'],  # 감지 시간
                    'signal_time': next_candle['timestamp'],  # 실제 사용 시간 (다음 봉)
                    'entry_zone': current['low'],
                    'gap_size': gap_size
                })
        
        # Bearish FVG
        if current['high'] < prev2['low']:
            gap_size = (prev2['low'] - current['high']) / current['high'] * 100
            if gap_size >= 0.3:
                signals.append({
                    'type': 'FVG_bear',
                    'direction': 'short',
                    'detect_time': current['timestamp'],
                    'signal_time': next_candle['timestamp'],
                    'entry_zone': current['high'],
                    'gap_size': gap_size
                })
    return signals

# Order Block 감지
def detect_orderblock_signals(df_4h):
    signals = []
    for i in range(3, len(df_4h) - 1):  # -1: 다음 봉 필요
        prev2 = df_4h.iloc[i-2]
        prev1 = df_4h.iloc[i-1]
        current = df_4h.iloc[i]
        next_candle = df_4h.iloc[i+1]
        
        # Bullish Order Block
        if (prev2['close'] < prev2['open'] and
            prev1['close'] > prev1['open'] and
            prev1['close'] > prev2['high']):
            signals.append({
                'type': 'orderblock',
                'direction': 'long',
                'detect_time': prev1['timestamp'],
                'signal_time': next_candle['timestamp'],  # 다음 봉부터
                'entry_zone': prev2['high'],
            })
    return signals

# 시뮬레이션
def simulate(signals, df_15m, tp1, tp2, sl, time_stop):
    trades = []
    pending_signals = []
    current_position = None
    signals_sorted = sorted(signals, key=lambda x: x['signal_time'])
    signal_idx = 0
    
    for i, candle in df_15m.iterrows():
        candle_time = candle['timestamp']
        
        # 새 시그널 추가 (signal_time 기준)
        while signal_idx < len(signals_sorted) and signals_sorted[signal_idx]['signal_time'] <= candle_time:
            sig = signals_sorted[signal_idx]
            expire_time = sig['signal_time'] + timedelta(hours=5)
            pending_signals.append({**sig, 'expire_time': expire_time})
            signal_idx += 1
        
        # 만료된 시그널 제거
        pending_signals = [s for s in pending_signals if s['expire_time'] > candle_time]
        
        # 포지션 관리
        if current_position is not None:
            pos = current_position
            bars_elapsed = (candle_time - pos['entry_time']).total_seconds() / 900
            closed = False
            
            if pos['direction'] == 'long':
                # TP1 체크
                if not pos['tp1_hit']:
                    tp1_price = pos['entry_price'] * (1 + tp1)
                    if candle['high'] >= tp1_price:
                        pos['tp1_hit'] = True
                        pos['sl_price'] = pos['entry_price']
                
                # TP2 체크
                tp2_price = pos['entry_price'] * (1 + tp2)
                if candle['high'] >= tp2_price:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TP2'
                    pos['pnl'] = tp2 * 100
                    closed = True
                # SL 체크
                elif candle['low'] <= pos['sl_price']:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'BE' if pos['tp1_hit'] else 'SL'
                    pos['pnl'] = 0 if pos['tp1_hit'] else sl * 100
                    closed = True
                # 시간 스탑
                elif bars_elapsed >= time_stop:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TIME'
                    pos['pnl'] = (candle['close'] - pos['entry_price']) / pos['entry_price'] * 100
                    closed = True
            
            else:  # short
                if not pos['tp1_hit']:
                    tp1_price = pos['entry_price'] * (1 - tp1)
                    if candle['low'] <= tp1_price:
                        pos['tp1_hit'] = True
                        pos['sl_price'] = pos['entry_price']
                
                tp2_price = pos['entry_price'] * (1 - tp2)
                if candle['low'] <= tp2_price:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TP2'
                    pos['pnl'] = tp2 * 100
                    closed = True
                elif candle['high'] >= pos['sl_price']:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'BE' if pos['tp1_hit'] else 'SL'
                    pos['pnl'] = 0 if pos['tp1_hit'] else sl * 100
                    closed = True
                elif bars_elapsed >= time_stop:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TIME'
                    pos['pnl'] = (pos['entry_price'] - candle['close']) / pos['entry_price'] * 100
                    closed = True
            
            if closed:
                trades.append(pos)
                current_position = None
        
        # 진입 확인
        if current_position is None and pending_signals:
            for sig in pending_signals[:]:
                entry_zone = sig['entry_zone']
                direction = sig['direction']
                touched = (direction == 'long' and candle['low'] <= entry_zone) or \
                          (direction == 'short' and candle['high'] >= entry_zone)
                
                if touched:
                    current_position = {
                        'strategy': sig['type'],
                        'direction': direction,
                        'detect_time': sig['detect_time'],
                        'signal_time': sig['signal_time'],
                        'entry_time': candle_time,
                        'entry_price': entry_zone,
                        'sl_price': entry_zone * (1 + sl) if direction == 'long' else entry_zone * (1 - sl),
                        'tp1_hit': False
                    }
                    pending_signals.remove(sig)
                    break
    
    return trades

# 백테스트 실행
print("\n" + "=" * 80)
print("🔍 시그널 감지 (Look-Ahead Bias 수정)")
print("=" * 80)

fvg_signals = detect_fvg_signals(df_4h)
ob_signals = detect_orderblock_signals(df_4h)
all_signals = fvg_signals + ob_signals

print(f"\nFVG 시그널: {len(fvg_signals)}개")
print(f"Order Block: {len(ob_signals)}개")
print(f"총 시그널: {len(all_signals)}개")

# 첫 시그널 확인
if len(all_signals) > 0:
    first_sig = sorted(all_signals, key=lambda x: x['signal_time'])[0]
    print(f"\n첫 시그널 예시:")
    print(f"  감지 시간: {first_sig['detect_time']}")
    print(f"  사용 시간: {first_sig['signal_time']} ← 다음 봉!")
    time_diff = (first_sig['signal_time'] - first_sig['detect_time']).total_seconds() / 3600
    print(f"  시간차: {time_diff}시간")
    print(f"\n✅ Look-Ahead Bias 수정 완료!")
    print(f"  → 시그널 감지 후 {time_diff}시간 뒤부터 사용")

print("\n" + "=" * 80)
print("⏳ 백테스트 실행 중...")
print("=" * 80)

trades = simulate(all_signals, df_15m, tp1=0.015, tp2=0.035, sl=-0.015, time_stop=48)
df_trades = pd.DataFrame(trades)

print(f"\n총 거래: {len(df_trades)}건")

if len(df_trades) > 0:
    # 기간 계산
    first_time = df_trades['entry_time'].min()
    last_time = df_trades['exit_time'].max()
    months = (last_time - first_time).days / 30
    
    # 수익 계산
    total_pnl = df_trades['pnl'].sum()
    monthly_avg = total_pnl / months
    
    # MDD 계산
    cumsum = df_trades['pnl'].cumsum()
    running_max = cumsum.cummax()
    drawdown = cumsum - running_max
    mdd = drawdown.min()
    
    # 승률 계산
    sl_count = (df_trades['result'] == 'SL').sum()
    win_count = len(df_trades) - sl_count
    win_rate = win_count / len(df_trades) * 100
    
    # 결과 분포
    result_dist = df_trades['result'].value_counts()
    
    print("\n" + "=" * 80)
    print("📊 백테스트 결과 (Look-Ahead Bias 수정)")
    print("=" * 80)
    
    print(f"\n기간: {first_time.date()} ~ {last_time.date()} ({months:.1f}개월)")
    print(f"\n총 수익: {total_pnl:.1f}%")
    print(f"월평균 수익: {monthly_avg:.2f}%")
    print(f"평균 수익/건: {df_trades['pnl'].mean():.3f}%")
    print(f"승률 (SL 회피): {win_rate:.1f}%")
    print(f"MDD: {mdd:.1f}%")
    
    print(f"\n결과 분포:")
    for result, count in result_dist.items():
        pct = count / len(df_trades) * 100
        avg_pnl = df_trades[df_trades['result'] == result]['pnl'].mean()
        print(f"  {result}: {count}건 ({pct:.1f}%), 평균 {avg_pnl:.3f}%")
    
    # 전략별 성과
    print(f"\n전략별 성과:")
    for strat in df_trades['strategy'].unique():
        strat_trades = df_trades[df_trades['strategy'] == strat]
        strat_win = len(strat_trades[strat_trades['result'] != 'SL'])
        strat_winrate = strat_win / len(strat_trades) * 100
        strat_pnl = strat_trades['pnl'].sum()
        strat_avg = strat_trades['pnl'].mean()
        print(f"  {strat}: {len(strat_trades)}건, 승률 {strat_winrate:.1f}%, 총 {strat_pnl:.1f}%, 평균 {strat_avg:.3f}%")
    
    # 저장
    df_trades.to_csv('backtest_no_lookahead.csv', index=False)
    print(f"\n저장: backtest_no_lookahead.csv")
    
    # 기존 결과와 비교
    print("\n" + "=" * 80)
    print("📊 기존 백테스트와 비교")
    print("=" * 80)
    
    print(f"\n[기존 백테스트 (Look-Ahead Bias 있음)]:")
    print(f"  총 거래: 1,920건")
    print(f"  월평균: 21.27%")
    print(f"  승률: 85.9%")
    print(f"  MDD: -5.0%")
    
    print(f"\n[수정 백테스트 (Look-Ahead Bias 없음)]:")
    print(f"  총 거래: {len(df_trades)}건")
    print(f"  월평균: {monthly_avg:.2f}%")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  MDD: {mdd:.1f}%")
    
    print(f"\n변화:")
    print(f"  거래 수: {len(df_trades) - 1920:+d}건 ({(len(df_trades) - 1920) / 1920 * 100:+.1f}%)")
    print(f"  월평균: {monthly_avg - 21.27:+.2f}%p ({(monthly_avg - 21.27) / 21.27 * 100:+.1f}%)")
    print(f"  승률: {win_rate - 85.9:+.1f}%p")
    print(f"  MDD: {mdd - (-5.0):+.1f}%p")
    
    if monthly_avg < 21.27:
        print(f"\n⚠️ 예상대로 성과가 하락했습니다.")
        print(f"  하지만 이제 실전에 적용 가능한 정확한 수치입니다! ✅")
    else:
        print(f"\n🤔 예상과 다르게 성과가 유지/개선되었습니다.")
        print(f"  추가 검증이 필요합니다.")

print("\n" + "=" * 80)
print("✅ Look-Ahead Bias 수정 완료")
print("=" * 80)
