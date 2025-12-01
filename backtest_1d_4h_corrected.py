"""
타임프레임 상향 조정: 1D signal + 4H entry
Look-Ahead Bias 완전 제거
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 80)
print("🔧 타임프레임 상향 백테스트: 1D signal + 4H entry")
print("=" * 80)

# 데이터 로드
df_4h = pd.read_csv('btc_4h_ohlcv.csv', parse_dates=['datetime'])
df_4h = df_4h.rename(columns={'datetime': 'timestamp'})

print(f"\n4H 데이터 기간: {df_4h['timestamp'].min()} ~ {df_4h['timestamp'].max()}")
print(f"4H 캔들: {len(df_4h):,}개")

# 1D 캔들 생성 (4H에서 리샘플링)
df_4h_sorted = df_4h.sort_values('timestamp').copy()
df_4h_sorted.set_index('timestamp', inplace=True)

df_1d = df_4h_sorted.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna().reset_index()

print(f"1D 캔들: {len(df_1d):,}개 (4H에서 생성)")

# FVG 감지 (1D 차트)
def detect_fvg_signals_1d(df_1d):
    signals = []
    for i in range(2, len(df_1d) - 1):  # -1: 다음 봉 필요
        current = df_1d.iloc[i]
        prev2 = df_1d.iloc[i-2]
        next_candle = df_1d.iloc[i+1]  # 다음 일봉 (실제 시그널 사용 가능 시점)
        
        # Bullish FVG
        if current['low'] > prev2['high']:
            gap_size = (current['low'] - prev2['high']) / prev2['high'] * 100
            if gap_size >= 0.3:
                signals.append({
                    'type': 'FVG_bull', 
                    'direction': 'long',
                    'detect_time': current['timestamp'],  # 감지 시간 (일봉 종료)
                    'signal_time': next_candle['timestamp'],  # 실제 사용 시간 (다음 일봉)
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

# Order Block 감지 (1D 차트)
def detect_orderblock_signals_1d(df_1d):
    signals = []
    for i in range(3, len(df_1d) - 1):
        prev2 = df_1d.iloc[i-2]
        prev1 = df_1d.iloc[i-1]
        current = df_1d.iloc[i]
        next_candle = df_1d.iloc[i+1]
        
        # Bullish Order Block
        if (prev2['close'] < prev2['open'] and
            prev1['close'] > prev1['open'] and
            prev1['close'] > prev2['high']):
            signals.append({
                'type': 'orderblock',
                'direction': 'long',
                'detect_time': prev1['timestamp'],
                'signal_time': next_candle['timestamp'],
                'entry_zone': prev2['high'],
            })
    return signals

# 시뮬레이션 (4H 차트 진입)
def simulate_4h_entry(signals, df_4h, tp1, tp2, sl, time_stop_hours):
    trades = []
    pending_signals = []
    current_position = None
    signals_sorted = sorted(signals, key=lambda x: x['signal_time'])
    signal_idx = 0
    
    time_stop_bars = time_stop_hours // 4  # 4H 차트 기준 바 수
    
    for i, candle in df_4h.iterrows():
        candle_time = candle['timestamp']
        
        # 새 시그널 추가 (signal_time 기준)
        while signal_idx < len(signals_sorted) and signals_sorted[signal_idx]['signal_time'] <= candle_time:
            sig = signals_sorted[signal_idx]
            expire_time = sig['signal_time'] + timedelta(hours=24)  # 24시간 유효
            pending_signals.append({**sig, 'expire_time': expire_time})
            signal_idx += 1
        
        # 만료된 시그널 제거
        pending_signals = [s for s in pending_signals if s['expire_time'] > candle_time]
        
        # 포지션 관리
        if current_position is not None:
            pos = current_position
            bars_elapsed = (candle_time - pos['entry_time']).total_seconds() / 14400  # 4시간 = 14400초
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
                elif bars_elapsed >= time_stop_bars:
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
                elif bars_elapsed >= time_stop_bars:
                    pos['exit_time'] = candle_time
                    pos['result'] = 'TIME'
                    pos['pnl'] = (pos['entry_price'] - candle['close']) / pos['entry_price'] * 100
                    closed = True
            
            if closed:
                trades.append(pos)
                current_position = None
        
        # 진입 확인 (4H 차트)
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
print("🔍 1D 시그널 감지 (Look-Ahead Bias 없음)")
print("=" * 80)

fvg_signals = detect_fvg_signals_1d(df_1d)
ob_signals = detect_orderblock_signals_1d(df_1d)
all_signals = fvg_signals + ob_signals

print(f"\nFVG 시그널 (1D): {len(fvg_signals)}개")
print(f"Order Block (1D): {len(ob_signals)}개")
print(f"총 시그널: {len(all_signals)}개")

# 첫 시그널 확인
if len(all_signals) > 0:
    first_sig = sorted(all_signals, key=lambda x: x['signal_time'])[0]
    print(f"\n첫 시그널 예시:")
    print(f"  감지 시간 (1D 종료): {first_sig['detect_time']}")
    print(f"  사용 시간 (다음 1D): {first_sig['signal_time']}")
    time_diff = (first_sig['signal_time'] - first_sig['detect_time']).total_seconds() / 3600
    print(f"  시간차: {time_diff}시간")
    print(f"\n✅ Look-Ahead Bias 완전 제거!")

print("\n" + "=" * 80)
print("⏳ 백테스트 실행 중 (4H 차트 진입)...")
print("=" * 80)

trades = simulate_4h_entry(
    all_signals, 
    df_4h, 
    tp1=0.015,   # TP1: 1.5%
    tp2=0.035,   # TP2: 3.5%
    sl=-0.015,   # SL: -1.5%
    time_stop_hours=48  # 48시간 (12바 on 4H)
)

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
    
    # 승률 계산 (정확한 방법)
    tp2_count = (df_trades['result'] == 'TP2').sum()
    be_count = (df_trades['result'] == 'BE').sum()
    time_positive = df_trades[(df_trades['result'] == 'TIME') & (df_trades['pnl'] > 0)]
    win_count = tp2_count + be_count + len(time_positive)
    
    sl_count = (df_trades['result'] == 'SL').sum()
    time_negative = df_trades[(df_trades['result'] == 'TIME') & (df_trades['pnl'] < 0)]
    loss_count = sl_count + len(time_negative)
    
    win_rate_accurate = win_count / len(df_trades) * 100
    win_rate_standard = (len(df_trades) - sl_count) / len(df_trades) * 100
    
    # 결과 분포
    result_dist = df_trades['result'].value_counts()
    
    print("\n" + "=" * 80)
    print("📊 백테스트 결과 (1D signal + 4H entry)")
    print("=" * 80)
    
    print(f"\n기간: {first_time.date()} ~ {last_time.date()} ({months:.1f}개월, {months/12:.1f}년)")
    print(f"\n💰 수익 성과:")
    print(f"  총 수익: {total_pnl:.1f}%")
    print(f"  월평균 수익: {monthly_avg:.2f}%")
    print(f"  평균 수익/건: {df_trades['pnl'].mean():.3f}%")
    print(f"  MDD: {mdd:.1f}%")
    
    print(f"\n🎯 승률 분석:")
    print(f"  정확한 승률: {win_rate_accurate:.1f}% ({win_count}승 / {loss_count}패)")
    print(f"  표준 승률 (SL회피): {win_rate_standard:.1f}%")
    
    print(f"\n📊 결과 분포:")
    for result in ['TP2', 'BE', 'TIME', 'SL']:
        if result in result_dist.index:
            count = result_dist[result]
            pct = count / len(df_trades) * 100
            result_trades = df_trades[df_trades['result'] == result]
            avg_pnl = result_trades['pnl'].mean()
            
            if result == 'TIME':
                positive = len(result_trades[result_trades['pnl'] > 0])
                negative = len(result_trades[result_trades['pnl'] < 0])
                print(f"  {result}: {count}건 ({pct:.1f}%), 평균 {avg_pnl:.3f}% [+{positive}/-{negative}]")
            else:
                print(f"  {result}: {count}건 ({pct:.1f}%), 평균 {avg_pnl:.3f}%")
    
    # 전략별 성과
    print(f"\n📈 전략별 성과:")
    for strat in df_trades['strategy'].unique():
        strat_trades = df_trades[df_trades['strategy'] == strat]
        strat_tp2 = (strat_trades['result'] == 'TP2').sum()
        strat_be = (strat_trades['result'] == 'BE').sum()
        strat_time_pos = len(strat_trades[(strat_trades['result'] == 'TIME') & (strat_trades['pnl'] > 0)])
        strat_win = strat_tp2 + strat_be + strat_time_pos
        strat_winrate = strat_win / len(strat_trades) * 100
        strat_pnl = strat_trades['pnl'].sum()
        strat_avg = strat_trades['pnl'].mean()
        print(f"  {strat}: {len(strat_trades)}건, 승률 {strat_winrate:.1f}%, 총 {strat_pnl:.1f}%, 평균 {strat_avg:.3f}%")
    
    # 저장
    df_trades.to_csv('backtest_1d_4h_corrected.csv', index=False)
    print(f"\n💾 저장: backtest_1d_4h_corrected.csv")
    
    # 기존 결과와 비교
    print("\n" + "=" * 80)
    print("📊 전략 비교")
    print("=" * 80)
    
    print(f"\n[기존 전략 (4H+15M) - Look-Ahead Bias 있음]:")
    print(f"  총 거래: 1,920건")
    print(f"  월평균: 21.27%")
    print(f"  승률: 85.9% (표준) / 75.5% (정확)")
    print(f"  MDD: -5.0%")
    print(f"  ⚠️ 실전 불가 (미래 데이터 사용)")
    
    print(f"\n[수정 전략 (4H+15M) - Look-Ahead Bias 없음]:")
    print(f"  총 거래: 1,030건")
    print(f"  월평균: -1.03%")
    print(f"  승률: 63.2%")
    print(f"  MDD: -127.8%")
    print(f"  ❌ 전략 실패")
    
    print(f"\n[신규 전략 (1D+4H) - Look-Ahead Bias 없음]:")
    print(f"  총 거래: {len(df_trades)}건")
    print(f"  월평균: {monthly_avg:.2f}%")
    print(f"  승률: {win_rate_standard:.1f}% (표준) / {win_rate_accurate:.1f}% (정확)")
    print(f"  MDD: {mdd:.1f}%")
    
    # 평가
    print(f"\n" + "=" * 80)
    print("🎯 최종 평가")
    print("=" * 80)
    
    if monthly_avg > 5 and mdd > -20:
        print(f"\n✅ 신규 전략 (1D+4H) 사용 가능!")
        print(f"  - 월평균 {monthly_avg:.2f}%는 충분히 실용적")
        print(f"  - MDD {mdd:.1f}%는 관리 가능한 수준")
        print(f"  - Look-Ahead Bias 완전 제거로 실전 적용 가능")
    elif monthly_avg > 0 and mdd > -50:
        print(f"\n⚠️ 신규 전략 (1D+4H) 개선 필요")
        print(f"  - 월평균 {monthly_avg:.2f}%는 낮지만 양수")
        print(f"  - MDD {mdd:.1f}%는 다소 크지만 감내 가능")
        print(f"  - 파라미터 최적화 또는 필터 추가 권장")
    else:
        print(f"\n❌ 신규 전략 (1D+4H)도 실패")
        print(f"  - 월평균 {monthly_avg:.2f}%는 손실")
        print(f"  - MDD {mdd:.1f}%는 치명적")
        print(f"  - 근본적인 전략 재설계 필요")
        print(f"\n다음 옵션 고려:")
        print(f"  1. 진입 유효기간 연장 (현재 24시간)")
        print(f"  2. 다른 시그널 방식 (트렌드 필터, RSI 등)")
        print(f"  3. 파라미터 최적화 (TP/SL 비율)")

print("\n" + "=" * 80)
print("✅ 백테스트 완료")
print("=" * 80)
