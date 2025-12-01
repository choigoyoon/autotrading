import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

print("=" * 90)
print("🔍 미래 데이터 사용 여부 & 동시 포지션 체크")
print("=" * 90)

# ===== 1. 각 전략별 미래 데이터 체크 =====
print("\n[1] 전략별 미래 데이터 사용 여부")
print("-" * 60)

strategies_check = {
    "FVG 상승 4H": """
        조건: df[i-2]['high'] < df[i]['low']
        → i-2, i 데이터만 사용 (과거만) ✅
        진입: FVG 발생 후 터치 대기 → 다음 캔들 시가 진입 ✅
    """,
    "FVG 하락 4H 숏": """
        조건: df[i-2]['low'] > df[i]['high']
        → i-2, i 데이터만 사용 (과거만) ✅
        진입: FVG 발생 후 터치 대기 → 다음 캔들 시가 진입 ✅
    """,
    "오더블럭 4H": """
        조건: df[i] 양봉, df[i-1] 음봉, 2% 이상 상승
        → i, i-1 데이터만 사용 (과거만) ✅
        진입: 오더블럭 영역 터치 대기 → 다음 캔들 시가 진입 ✅
    """,
    "브레이커 4H": """
        조건: window[i-15:i]에서 고점 찾고, 현재가가 돌파
        ⚠️ 문제: 고점 돌파 확인 시점 = 현재 캔들 종가
        → 종가 확정 전에 돌파 여부 알 수 없음!
        🔴 미래 데이터 사용 가능성 있음
    """,
    "골든크로스 되돌림": """
        조건: EMA20 > EMA50 이고, 저가가 EMA20 터치
        → EMA는 과거 데이터로 계산 ✅
        → 하지만 EMA 계산에 현재 종가 포함
        ⚠️ 주의: 캔들 완성 전 EMA 값 변동 가능
    """,
    "지지선 리테스트": """
        조건: window[i-30:i-5]에서 지지선 찾고, 현재가 터치
        → i-5까지만 사용하므로 과거만 ✅
        → 현재가 터치 여부는 실시간 확인 가능 ✅
    """,
}

for name, check in strategies_check.items():
    print(f"\n[{name}]")
    print(check)

# ===== 2. 브레이커 전략 수정 =====
print("\n" + "=" * 90)
print("🔧 브레이커 전략 수정 (미래 데이터 제거)")
print("=" * 90)

def detect_breaker_fixed(df):
    """
    수정: 고점 돌파는 '완성된 캔들 종가'로 확인
    진입: 돌파 확인 다음 캔들부터 되돌림 대기
    """
    signals = []
    for i in range(16, len(df)):  # i-1 캔들에서 돌파 확인
        window = df.iloc[i-16:i-1]  # i-1 이전 데이터만
        highs_idx = argrelextrema(window['high'].values, np.greater, order=4)[0]
        
        if len(highs_idx) >= 1:
            high_price = window.iloc[highs_idx[-1]]['high']
            # i-1 캔들 종가가 돌파했는지 (완성된 캔들)
            if df.iloc[i-1]['close'] > high_price * 1.005:
                # i번째 캔들부터 되돌림 대기
                signals.append({
                    'time': df.iloc[i]['datetime'],  # i 캔들 시작 시점
                    'level': high_price,
                    'type': 'breaker_fixed'
                })
    return signals

print("기존: 현재 캔들 종가로 돌파 확인 → 미래 데이터")
print("수정: 이전 캔들 종가로 돌파 확인 → 과거 데이터만 ✅")

# ===== 3. 동시 포지션 체크 =====
print("\n" + "=" * 90)
print("📊 동시 포지션 분석")
print("=" * 90)

# 간단한 시뮬레이션으로 포지션 겹침 체크
def simulate_with_position_tracking(signals, df_15m, tp1, tp2, sl, time_stop, direction='long'):
    positions = []  # (entry_time, exit_time, type)
    times = df_15m['datetime'].values
    lows, highs, opens, closes = df_15m['low'].values, df_15m['high'].values, df_15m['open'].values, df_15m['close'].values
    
    for sig in signals:
        start = np.searchsorted(times, np.datetime64(sig['time']))
        if start >= len(times) - 200:
            continue
        
        touch_idx = None
        for i in range(start+1, min(start+50, len(times))):
            if direction == 'long' and lows[i] <= sig['level']:
                touch_idx = i
                break
            elif direction == 'short' and highs[i] >= sig['level']:
                touch_idx = i
                break
        
        if touch_idx is None:
            continue
        
        entry_idx = touch_idx + 1
        if entry_idx >= len(times):
            continue
        
        ep = opens[entry_idx]
        entry_time = times[entry_idx]
        
        if direction == 'long':
            tp1_p, tp2_p, sl_p = ep*(1+tp1/100), ep*(1+tp2/100), ep*(1+sl/100)
        else:
            tp1_p, tp2_p, sl_p = ep*(1-tp1/100), ep*(1-tp2/100), ep*(1-sl/100)
        
        exit_idx = None
        tp1_hit = False
        
        for i in range(entry_idx+1, min(entry_idx+200, len(times))):
            if time_stop > 0 and (i - entry_idx) >= time_stop and not tp1_hit:
                exit_idx = i
                break
            
            if not tp1_hit:
                if direction == 'long':
                    if lows[i] <= sl_p or highs[i] >= tp1_p:
                        if highs[i] >= tp1_p:
                            tp1_hit = True
                            continue
                        exit_idx = i
                        break
                else:
                    if highs[i] >= sl_p or lows[i] <= tp1_p:
                        if lows[i] <= tp1_p:
                            tp1_hit = True
                            continue
                        exit_idx = i
                        break
            else:
                if direction == 'long':
                    if lows[i] <= ep or highs[i] >= tp2_p:
                        exit_idx = i
                        break
                else:
                    if highs[i] >= ep or lows[i] <= tp2_p:
                        exit_idx = i
                        break
        
        if exit_idx is None:
            exit_idx = min(entry_idx + 200, len(times) - 1)
        
        positions.append({
            'entry_time': entry_time,
            'exit_time': times[exit_idx],
            'type': sig['type'],
            'direction': direction
        })
    
    return positions

# 각 전략 시그널 감지
def detect_bull_fvg(df):
    signals = []
    for i in range(2, len(df)):
        if df.iloc[i-2]['high'] < df.iloc[i]['low']:
            signals.append({'time': df.iloc[i]['datetime'], 'level': df.iloc[i]['low'], 'type': 'fvg_bull'})
    return signals

def detect_bear_fvg(df):
    signals = []
    for i in range(2, len(df)):
        if df.iloc[i-2]['low'] > df.iloc[i]['high']:
            signals.append({'time': df.iloc[i]['datetime'], 'level': df.iloc[i]['high'], 'type': 'fvg_bear'})
    return signals

def detect_orderblock(df):
    signals = []
    for i in range(3, len(df)):
        if df.iloc[i]['close'] > df.iloc[i]['open']:
            if df.iloc[i-1]['close'] < df.iloc[i-1]['open']:
                move = (df.iloc[i]['close'] - df.iloc[i-1]['low']) / df.iloc[i-1]['low'] * 100
                if move > 2:
                    signals.append({'time': df.iloc[i]['datetime'], 'level': df.iloc[i-1]['open'], 'type': 'orderblock'})
    return signals

# 포지션 수집
all_positions = []

# FVG 롱
pos = simulate_with_position_tracking(detect_bull_fvg(df_4h), df_15m, 2.0, 3.5, -1.5, 96, 'long')
all_positions.extend(pos)

# FVG 숏
pos = simulate_with_position_tracking(detect_bear_fvg(df_4h), df_15m, 2.0, 3.5, -1.5, 96, 'short')
all_positions.extend(pos)

# 오더블럭
pos = simulate_with_position_tracking(detect_orderblock(df_4h), df_15m, 1.5, 3.0, -1.5, 96, 'long')
all_positions.extend(pos)

# 브레이커 (수정된 버전)
pos = simulate_with_position_tracking(detect_breaker_fixed(df_4h), df_15m, 1.5, 3.0, -1.5, 96, 'long')
all_positions.extend(pos)

df_pos = pd.DataFrame(all_positions)
df_pos['entry_time'] = pd.to_datetime(df_pos['entry_time'])
df_pos['exit_time'] = pd.to_datetime(df_pos['exit_time'])
df_pos = df_pos.sort_values('entry_time')

print(f"\n총 포지션: {len(df_pos)}개")

# 동시 포지션 계산
def count_concurrent_positions(df_pos):
    events = []
    for _, row in df_pos.iterrows():
        events.append((row['entry_time'], 1, row['type'], row['direction']))
        events.append((row['exit_time'], -1, row['type'], row['direction']))
    
    events.sort(key=lambda x: x[0])
    
    current = 0
    max_concurrent = 0
    max_time = None
    concurrent_history = []
    
    long_count = 0
    short_count = 0
    
    for time, delta, ptype, direction in events:
        current += delta
        if direction == 'long':
            long_count += delta
        else:
            short_count += delta
        
        if current > max_concurrent:
            max_concurrent = current
            max_time = time
        
        concurrent_history.append({
            'time': time,
            'total': current,
            'long': long_count,
            'short': short_count
        })
    
    return max_concurrent, max_time, concurrent_history

max_pos, max_time, history = count_concurrent_positions(df_pos)

print(f"\n[동시 포지션 분석]")
print(f"  최대 동시 포지션: {max_pos}개")
print(f"  발생 시점: {max_time}")

# 동시 포지션 분포
df_hist = pd.DataFrame(history)
pos_dist = df_hist['total'].value_counts().sort_index()

print(f"\n[동시 포지션 분포]")
for n, cnt in pos_dist.items():
    if n > 0:
        print(f"  {n}개 동시: {cnt}회")

# 롱/숏 동시 보유 (헷지 상태)
hedge_count = ((df_hist['long'] > 0) & (df_hist['short'] > 0)).sum()
print(f"\n  롱+숏 동시 보유(헷지): {hedge_count}회")

# ===== 4. 자본 분배 시뮬레이션 =====
print("\n" + "=" * 90)
print("💰 자본 분배 시나리오")
print("=" * 90)

print("""
[시나리오 1: 단일 포지션]
  - 한 번에 1개 포지션만
  - 거래 빈도 ↓, 리스크 ↓
  
[시나리오 2: 전략별 자본 분배]
  - 전략당 25% 자본 할당 (4전략 기준)
  - 같은 전략 내에서는 1포지션
  - 최대 4포지션 동시 가능
  
[시나리오 3: 최대 포지션 제한]
  - 전체 최대 3포지션
  - 선착순 진입
  - 리스크 관리 용이
  
[시나리오 4: 방향별 분리]
  - 롱 최대 2포지션, 숏 최대 2포지션
  - 헷지 효과 가능
""")

# 시나리오별 시뮬레이션
print("\n[시나리오별 예상 성과]")
avg_positions = df_hist['total'].mean()
print(f"  평균 동시 포지션: {avg_positions:.2f}개")
print(f"  최대 동시 포지션: {max_pos}개")

# 단일 포지션 시 예상
single_reduction = 1 / avg_positions if avg_positions > 1 else 1
print(f"\n  단일 포지션 시 거래 감소율: {(1-single_reduction)*100:.1f}%")

