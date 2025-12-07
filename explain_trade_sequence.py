"""
매매 진입 순서 상세 설명
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 데이터 로드
df_15m = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df_4h = pd.read_csv('btc_4h_ohlcv.csv', parse_dates=['datetime'])
df_15m = df_15m.rename(columns={'datetime': 'timestamp'})
df_4h = df_4h.rename(columns={'datetime': 'timestamp'})

print("=" * 80)
print("📝 매매 진입 순서 상세 설명")
print("=" * 80)

def detect_fvg_signals(df_4h):
    signals = []
    for i in range(2, len(df_4h)):
        if df_4h['low'].iloc[i] > df_4h['high'].iloc[i-2]:
            gap_top = df_4h['low'].iloc[i]
            gap_bottom = df_4h['high'].iloc[i-2]
            gap_size = (gap_top - gap_bottom) / gap_bottom * 100
            if gap_size >= 0.3:
                signals.append({
                    'type': 'FVG_bull', 'direction': 'long',
                    'signal_time': df_4h['timestamp'].iloc[i],
                    'entry_zone': gap_top, 'gap_size': gap_size,
                    'candle_i-2': i-2, 'candle_i': i,
                    'price_i-2_high': df_4h['high'].iloc[i-2],
                    'price_i_low': df_4h['low'].iloc[i]
                })
        if df_4h['high'].iloc[i] < df_4h['low'].iloc[i-2]:
            gap_top = df_4h['low'].iloc[i-2]
            gap_bottom = df_4h['high'].iloc[i]
            gap_size = (gap_top - gap_bottom) / gap_bottom * 100
            if gap_size >= 0.3:
                signals.append({
                    'type': 'FVG_bear', 'direction': 'short',
                    'signal_time': df_4h['timestamp'].iloc[i],
                    'entry_zone': gap_bottom, 'gap_size': gap_size,
                    'candle_i-2': i-2, 'candle_i': i,
                    'price_i-2_low': df_4h['low'].iloc[i-2],
                    'price_i_high': df_4h['high'].iloc[i]
                })
    return signals

def detect_orderblock_signals(df_4h):
    signals = []
    for i in range(3, len(df_4h)):
        if (df_4h['close'].iloc[i-2] < df_4h['open'].iloc[i-2] and
            df_4h['close'].iloc[i-1] > df_4h['open'].iloc[i-1] and
            df_4h['close'].iloc[i-1] > df_4h['high'].iloc[i-2]):
            signals.append({
                'type': 'orderblock', 'direction': 'long',
                'signal_time': df_4h['timestamp'].iloc[i-1],
                'entry_zone': df_4h['high'].iloc[i-2],
                'candle_i-2': i-2, 'candle_i-1': i-1,
                'bearish_candle': f"Open: {df_4h['open'].iloc[i-2]:.0f}, Close: {df_4h['close'].iloc[i-2]:.0f}",
                'bullish_candle': f"Open: {df_4h['open'].iloc[i-1]:.0f}, Close: {df_4h['close'].iloc[i-1]:.0f}"
            })
    return signals

def simulate_with_tracking(signals, df_15m, tp1, tp2, sl, time_stop, max_trades=5):
    """진입 과정을 추적하는 시뮬레이션"""
    trades = []
    pending_signals = []
    current_position = None
    signals_sorted = sorted(signals, key=lambda x: x['signal_time'])
    signal_idx = 0
    
    entry_logs = []
    
    for i, candle in df_15m.iterrows():
        candle_time = candle['timestamp']
        
        # 새 시그널 추가
        while signal_idx < len(signals_sorted) and signals_sorted[signal_idx]['signal_time'] <= candle_time:
            sig = signals_sorted[signal_idx]
            expire_time = sig['signal_time'] + timedelta(hours=5)
            pending_signals.append({**sig, 'expire_time': expire_time})
            
            entry_logs.append({
                'step': '1_SIGNAL_DETECTED',
                'time': sig['signal_time'],
                'signal_type': sig['type'],
                'direction': sig['direction'],
                'entry_zone': sig['entry_zone'],
                'gap_size': sig.get('gap_size', 0),
                'details': sig
            })
            
            signal_idx += 1
        
        # 만료된 시그널 제거
        expired = [s for s in pending_signals if s['expire_time'] <= candle_time]
        for exp in expired:
            entry_logs.append({
                'step': '2_SIGNAL_EXPIRED',
                'time': candle_time,
                'signal_type': exp['type'],
                'signal_time': exp['signal_time'],
                'expired_after': (candle_time - exp['signal_time']).total_seconds() / 3600
            })
        
        pending_signals = [s for s in pending_signals if s['expire_time'] > candle_time]
        
        # 포지션 관리
        if current_position is not None:
            pos = current_position
            bars_elapsed = (candle_time - pos['entry_time']).total_seconds() / 900
            closed = False
            
            if pos['direction'] == 'long':
                if not pos['tp1_hit']:
                    tp1_price = pos['entry_price'] * (1 + tp1)
                    if candle['high'] >= tp1_price:
                        pos['tp1_hit'] = True
                        pos['tp1_time'] = candle_time
                        pos['sl_price'] = pos['entry_price']
                        
                        entry_logs.append({
                            'step': '4_TP1_HIT',
                            'time': candle_time,
                            'entry_price': pos['entry_price'],
                            'tp1_price': tp1_price,
                            'sl_moved_to': pos['sl_price'],
                            'bars_elapsed': bars_elapsed
                        })
                
                tp2_price = pos['entry_price'] * (1 + tp2)
                if candle['high'] >= tp2_price:
                    pos['exit_time'] = candle_time
                    pos['exit_price'] = tp2_price
                    pos['result'] = 'TP2'
                    pos['pnl'] = tp2 * 100
                    closed = True
                    
                    entry_logs.append({
                        'step': '5_EXIT_TP2',
                        'time': candle_time,
                        'entry_price': pos['entry_price'],
                        'exit_price': tp2_price,
                        'pnl': pos['pnl'],
                        'duration_hours': (candle_time - pos['entry_time']).total_seconds() / 3600
                    })
                    
                elif candle['low'] <= pos['sl_price']:
                    pos['exit_time'] = candle_time
                    pos['exit_price'] = pos['sl_price']
                    pos['result'] = 'BE' if pos['tp1_hit'] else 'SL'
                    pos['pnl'] = 0 if pos['tp1_hit'] else sl * 100
                    closed = True
                    
                    entry_logs.append({
                        'step': f'5_EXIT_{pos["result"]}',
                        'time': candle_time,
                        'entry_price': pos['entry_price'],
                        'exit_price': pos['sl_price'],
                        'pnl': pos['pnl'],
                        'tp1_hit': pos['tp1_hit'],
                        'duration_hours': (candle_time - pos['entry_time']).total_seconds() / 3600
                    })
                    
                elif bars_elapsed >= time_stop:
                    pos['exit_time'] = candle_time
                    pos['exit_price'] = candle['close']
                    pos['result'] = 'TIME'
                    pos['pnl'] = (candle['close'] - pos['entry_price']) / pos['entry_price'] * 100
                    closed = True
                    
                    entry_logs.append({
                        'step': '5_EXIT_TIME',
                        'time': candle_time,
                        'entry_price': pos['entry_price'],
                        'exit_price': candle['close'],
                        'pnl': pos['pnl'],
                        'tp1_hit': pos['tp1_hit'],
                        'duration_hours': bars_elapsed / 4
                    })
            else:  # short
                if not pos['tp1_hit']:
                    tp1_price = pos['entry_price'] * (1 - tp1)
                    if candle['low'] <= tp1_price:
                        pos['tp1_hit'] = True
                        pos['tp1_time'] = candle_time
                        pos['sl_price'] = pos['entry_price']
                        
                        entry_logs.append({
                            'step': '4_TP1_HIT',
                            'time': candle_time,
                            'entry_price': pos['entry_price'],
                            'tp1_price': tp1_price,
                            'sl_moved_to': pos['sl_price'],
                            'bars_elapsed': bars_elapsed
                        })
                
                tp2_price = pos['entry_price'] * (1 - tp2)
                if candle['low'] <= tp2_price:
                    pos['exit_time'] = candle_time
                    pos['exit_price'] = tp2_price
                    pos['result'] = 'TP2'
                    pos['pnl'] = tp2 * 100
                    closed = True
                    
                    entry_logs.append({
                        'step': '5_EXIT_TP2',
                        'time': candle_time,
                        'entry_price': pos['entry_price'],
                        'exit_price': tp2_price,
                        'pnl': pos['pnl'],
                        'duration_hours': (candle_time - pos['entry_time']).total_seconds() / 3600
                    })
                    
                elif candle['high'] >= pos['sl_price']:
                    pos['exit_time'] = candle_time
                    pos['exit_price'] = pos['sl_price']
                    pos['result'] = 'BE' if pos['tp1_hit'] else 'SL'
                    pos['pnl'] = 0 if pos['tp1_hit'] else sl * 100
                    closed = True
                    
                    entry_logs.append({
                        'step': f'5_EXIT_{pos["result"]}',
                        'time': candle_time,
                        'entry_price': pos['entry_price'],
                        'exit_price': pos['sl_price'],
                        'pnl': pos['pnl'],
                        'tp1_hit': pos['tp1_hit'],
                        'duration_hours': (candle_time - pos['entry_time']).total_seconds() / 3600
                    })
                    
                elif bars_elapsed >= time_stop:
                    pos['exit_time'] = candle_time
                    pos['exit_price'] = candle['close']
                    pos['result'] = 'TIME'
                    pos['pnl'] = (pos['entry_price'] - candle['close']) / pos['entry_price'] * 100
                    closed = True
                    
                    entry_logs.append({
                        'step': '5_EXIT_TIME',
                        'time': candle_time,
                        'entry_price': pos['entry_price'],
                        'exit_price': candle['close'],
                        'pnl': pos['pnl'],
                        'tp1_hit': pos['tp1_hit'],
                        'duration_hours': bars_elapsed / 4
                    })
            
            if closed:
                trades.append(pos)
                current_position = None
                
                if len(trades) >= max_trades:
                    return trades, entry_logs
        
        # 새 진입 시도
        if current_position is None and pending_signals:
            for sig in pending_signals[:]:
                entry_zone = sig['entry_zone']
                direction = sig['direction']
                touched = (direction == 'long' and candle['low'] <= entry_zone) or \
                          (direction == 'short' and candle['high'] >= entry_zone)
                
                if touched:
                    current_position = {
                        'strategy': sig['type'], 'direction': direction,
                        'signal_time': sig['signal_time'], 'entry_time': candle_time,
                        'entry_price': entry_zone,
                        'sl_price': entry_zone * (1 + sl) if direction == 'long' else entry_zone * (1 - sl),
                        'tp1_hit': False
                    }
                    
                    entry_logs.append({
                        'step': '3_ENTRY_EXECUTED',
                        'time': candle_time,
                        'signal_time': sig['signal_time'],
                        'wait_time_hours': (candle_time - sig['signal_time']).total_seconds() / 3600,
                        'strategy': sig['type'],
                        'direction': direction,
                        'entry_price': entry_zone,
                        'initial_sl': current_position['sl_price'],
                        'tp1_target': entry_zone * (1 + tp1) if direction == 'long' else entry_zone * (1 - tp1),
                        'tp2_target': entry_zone * (1 + tp2) if direction == 'long' else entry_zone * (1 - tp2),
                        'candle_high': candle['high'],
                        'candle_low': candle['low']
                    })
                    
                    pending_signals.remove(sig)
                    break
    
    return trades, entry_logs

# 시그널 감지
print("\n📊 시그널 감지...")
fvg_signals = detect_fvg_signals(df_4h)
ob_signals = detect_orderblock_signals(df_4h)
all_signals = fvg_signals + ob_signals
print(f"총 시그널: {len(all_signals)}개")

# 샘플 거래 추출
print("\n⏳ 샘플 거래 추출 중...")
trades, logs = simulate_with_tracking(
    all_signals, df_15m,
    tp1=0.015, tp2=0.035, sl=-0.015, time_stop=48,
    max_trades=3
)

print(f"추출된 거래: {len(trades)}개")

# 진입 순서 상세 설명
print("\n" + "=" * 80)
print("📖 매매 진입 순서 상세 설명")
print("=" * 80)

trade_num = 0
for log in logs:
    if log['step'] == '1_SIGNAL_DETECTED':
        print(f"\n{'='*80}")
        print(f"🔔 STEP 1: 시그널 감지")
        print(f"{'='*80}")
        print(f"시간: {log['time']}")
        print(f"전략: {log['signal_type']}")
        print(f"방향: {log['direction'].upper()}")
        print(f"진입존: ${log['entry_zone']:.2f}")
        
        if log['signal_type'].startswith('FVG'):
            print(f"갭 크기: {log['gap_size']:.3f}%")
            details = log['details']
            if log['direction'] == 'long':
                print(f"\n📊 FVG 형성 과정 (Bullish):")
                print(f"  - 4H 캔들[i-2] 고가: ${details['price_i-2_high']:.2f}")
                print(f"  - 4H 캔들[i] 저가: ${details['price_i_low']:.2f}")
                print(f"  - 갭 발생: 캔들[i]의 저가가 캔들[i-2]의 고가보다 높음")
                print(f"  - 진입존 = 갭 상단 = ${details['price_i_low']:.2f}")
            else:
                print(f"\n📊 FVG 형성 과정 (Bearish):")
                print(f"  - 4H 캔들[i-2] 저가: ${details['price_i-2_low']:.2f}")
                print(f"  - 4H 캔들[i] 고가: ${details['price_i_high']:.2f}")
                print(f"  - 갭 발생: 캔들[i]의 고가가 캔들[i-2]의 저가보다 낮음")
                print(f"  - 진입존 = 갭 하단 = ${details['price_i_high']:.2f}")
        else:
            details = log['details']
            print(f"\n📊 Order Block 형성:")
            print(f"  - 약세 캔들[i-2]: {details['bearish_candle']}")
            print(f"  - 강세 캔들[i-1]: {details['bullish_candle']}")
            print(f"  - 조건: 강세 캔들이 약세 캔들의 고가를 돌파")
            print(f"  - 진입존 = 약세 캔들의 고가 = ${log['entry_zone']:.2f}")
        
        print(f"\n⏰ 시그널 유효기간: 5시간 (만료 시간: {log['time'] + timedelta(hours=5)})")
        
    elif log['step'] == '3_ENTRY_EXECUTED':
        trade_num += 1
        print(f"\n{'='*80}")
        print(f"✅ STEP 2: 진입 실행 (거래 #{trade_num})")
        print(f"{'='*80}")
        print(f"진입 시간: {log['time']}")
        print(f"시그널 발생 시간: {log['signal_time']}")
        print(f"대기 시간: {log['wait_time_hours']:.2f}시간")
        print(f"\n전략: {log['strategy']} / 방향: {log['direction'].upper()}")
        print(f"진입가: ${log['entry_price']:.2f}")
        print(f"현재 캔들: High ${log['candle_high']:.2f} / Low ${log['candle_low']:.2f}")
        
        if log['direction'] == 'long':
            print(f"\n💡 진입 조건 충족:")
            print(f"  - 15분봉의 저가(${log['candle_low']:.2f})가 진입존(${log['entry_zone']:.2f}) 터치")
            print(f"  - LONG 진입 실행!")
        else:
            print(f"\n💡 진입 조건 충족:")
            print(f"  - 15분봉의 고가(${log['candle_high']:.2f})가 진입존(${log['entry_zone']:.2f}) 터치")
            print(f"  - SHORT 진입 실행!")
        
        print(f"\n🎯 설정된 청산 레벨:")
        print(f"  - TP1 (1.5%): ${log['tp1_target']:.2f} → 도달 시 SL을 본절(${ log['entry_price']:.2f})로 이동")
        print(f"  - TP2 (3.5%): ${log['tp2_target']:.2f}")
        print(f"  - SL (-1.5%): ${log['initial_sl']:.2f}")
        print(f"  - 시간스탑: 48바 (12시간)")
        
    elif log['step'] == '4_TP1_HIT':
        print(f"\n{'='*80}")
        print(f"🎯 STEP 3: TP1 도달")
        print(f"{'='*80}")
        print(f"시간: {log['time']}")
        print(f"진입가: ${log['entry_price']:.2f}")
        print(f"TP1 가격: ${log['tp1_price']:.2f}")
        print(f"경과 시간: {log['bars_elapsed']:.1f}바 ({log['bars_elapsed']/4:.2f}시간)")
        print(f"\n✅ SL을 본절로 이동: ${log['entry_price']:.2f} → ${log['sl_moved_to']:.2f}")
        print(f"   (이제 손실 위험 제거, 최소 손익분기 보장)")
        
    elif log['step'].startswith('5_EXIT'):
        result = log['step'].split('_')[2]
        print(f"\n{'='*80}")
        print(f"🏁 STEP 4: 청산 ({result})")
        print(f"{'='*80}")
        print(f"청산 시간: {log['time']}")
        print(f"진입가: ${log['entry_price']:.2f}")
        print(f"청산가: ${log['exit_price']:.2f}")
        print(f"보유 시간: {log['duration_hours']:.2f}시간")
        
        if result == 'TP2':
            print(f"\n🎉 목표가 달성!")
            print(f"   PNL: +{log['pnl']:.2f}%")
        elif result == 'BE':
            print(f"\n🟡 본절 청산")
            print(f"   TP1 도달 후: {log['tp1_hit']}")
            print(f"   PNL: {log['pnl']:.2f}%")
        elif result == 'SL':
            print(f"\n❌ 손절")
            print(f"   TP1 도달 여부: {log['tp1_hit']}")
            print(f"   PNL: {log['pnl']:.2f}%")
        elif result == 'TIME':
            print(f"\n⏰ 시간스탑 (12시간 경과)")
            print(f"   TP1 도달 여부: {log['tp1_hit']}")
            print(f"   PNL: {log['pnl']:+.2f}%")
        
        print(f"\n" + "=" * 80)
        print(f"거래 #{trade_num} 완료")
        print(f"=" * 80)
        
    elif log['step'] == '2_SIGNAL_EXPIRED':
        print(f"\n⚠️  시그널 만료: {log['signal_type']} (발생: {log['signal_time']}, {log['expired_after']:.1f}시간 경과)")

# 요약
print("\n\n" + "=" * 80)
print("📋 매매 프로세스 요약")
print("=" * 80)
print("""
1️⃣  시그널 감지 (4H 차트)
   - FVG: 3개 캔들 사이에 갭(공백) 발생
   - Order Block: 약세→강세 반전 패턴
   → 진입존 설정, 5시간 유효

2️⃣  진입 대기 (15M 차트)
   - 15분봉이 진입존을 터치할 때까지 대기
   - Long: 15M 저가가 진입존 터치
   - Short: 15M 고가가 진입존 터치
   → 즉시 진입

3️⃣  포지션 관리
   - TP1 (1.5%) 도달 → SL을 본절로 이동
   - TP2 (3.5%) 도달 → 전체 청산
   - SL (-1.5%) 터치 → 손절 (본절 이동 전) 또는 본절 청산
   - 12시간 경과 → 시간스탑 청산

4️⃣  결과 기록
   - 다음 시그널 대기
   - 한 번에 1개 포지션만 보유
""")

