#!/usr/bin/env python3
"""
MTF 전략 v2 - 4시간봉 매매
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("MTF 전략 v2 - 4시간봉 매매, 일봉 추세 필터")
print("=" * 80)

# 데이터 로드
df_1d = pd.read_csv('analysis_1d.csv', parse_dates=['datetime'])
df_4h = pd.read_csv('analysis_4h.csv', parse_dates=['datetime'])

# 지표 계산
def add_indicators(df):
    df = df.copy()
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    return df.iloc[200:].reset_index(drop=True)

df_1d = add_indicators(df_1d)
df_4h = add_indicators(df_4h)

print(f"일봉: {len(df_1d)}개")
print(f"4시간: {len(df_4h)}개")

# 일봉 추세
df_1d['trend'] = np.where(df_1d['ema20'] > df_1d['ema50'], 'UP', 'DOWN')
df_1d['above_200'] = df_1d['close'] > df_1d['ema200']

# 일봉 데이터 매핑
df_1d['date'] = df_1d['datetime'].dt.date
daily_map = df_1d.set_index('date')[['trend', 'above_200', 'rsi', 'ema200']].to_dict('index')

# numpy 배열
close = df_4h['close'].values
high = df_4h['high'].values
low = df_4h['low'].values
bb_upper = df_4h['bb_upper'].values
bb_lower = df_4h['bb_lower'].values
bb_mid = df_4h['bb_mid'].values
rsi = df_4h['rsi'].values
datetimes = df_4h['datetime'].values

COST = 0.18

def get_daily(dt):
    """일봉 데이터 가져오기"""
    d = dt.date()
    for i in range(10):
        check = d - pd.Timedelta(days=i)
        if check in daily_map:
            return daily_map[check]
    return None

def simulate(idx, direction, sl_pct, tp_pct, max_bars=30):
    """매매 시뮬레이션"""
    entry = close[idx]
    
    if direction == 'LONG':
        sl = entry * (1 - sl_pct / 100)
        tp = entry * (1 + tp_pct / 100)
    else:
        sl = entry * (1 + sl_pct / 100)
        tp = entry * (1 - tp_pct / 100)
    
    mfe = 0
    mae = 0
    
    for j in range(idx + 1, min(idx + max_bars + 1, len(close))):
        if direction == 'LONG':
            mfe = max(mfe, (high[j] - entry) / entry * 100)
            mae = min(mae, (low[j] - entry) / entry * 100)
            if low[j] <= sl:
                return {'pnl': -sl_pct - COST, 'reason': 'SL', 'bars': j - idx, 'mfe': mfe, 'mae': mae}
            if high[j] >= tp:
                return {'pnl': tp_pct - COST, 'reason': 'TP', 'bars': j - idx, 'mfe': mfe, 'mae': mae}
        else:
            mfe = max(mfe, (entry - low[j]) / entry * 100)
            mae = min(mae, (entry - high[j]) / entry * 100)
            if high[j] >= sl:
                return {'pnl': -sl_pct - COST, 'reason': 'SL', 'bars': j - idx, 'mfe': mfe, 'mae': mae}
            if low[j] <= tp:
                return {'pnl': tp_pct - COST, 'reason': 'TP', 'bars': j - idx, 'mfe': mfe, 'mae': mae}
    
    exit_price = close[min(idx + max_bars, len(close) - 1)]
    if direction == 'LONG':
        pnl = (exit_price - entry) / entry * 100
    else:
        pnl = (entry - exit_price) / entry * 100
    return {'pnl': pnl - COST, 'reason': 'TIMEOUT', 'bars': max_bars, 'mfe': mfe, 'mae': mae}


# ============================================================
# 전략 1: 추세 방향 + BB 터치 (역추세)
# ============================================================
print("\n" + "=" * 80)
print("전략 1: 일봉 추세 방향 + 4H BB 터치 (역추세 눌림목)")
print("=" * 80)

print(f"\n{'조건':>20} {'SL':>5} {'TP':>5} {'거래':>6} {'승률':>8} {'평균':>8} {'월수익':>8}")
print("-" * 75)

best_trades = None
best_pnl = -999

for condition in ['trend_only', 'trend+ema200', 'ema200_only']:
    for sl in [3, 4, 5]:
        for tp in [5, 7, 10, 15]:
            trades = []
            last_idx = 0
            
            for i in range(len(close) - 50):
                if i < last_idx + 6:  # 24시간 간격
                    continue
                
                dt = pd.Timestamp(datetimes[i])
                daily = get_daily(dt)
                if daily is None:
                    continue
                
                # 조건별 LONG/SHORT 판단
                long_ok = False
                short_ok = False
                
                if condition == 'trend_only':
                    long_ok = daily['trend'] == 'UP'
                    short_ok = daily['trend'] == 'DOWN'
                elif condition == 'trend+ema200':
                    long_ok = daily['trend'] == 'UP' and daily['above_200']
                    short_ok = daily['trend'] == 'DOWN' and not daily['above_200']
                elif condition == 'ema200_only':
                    long_ok = daily['above_200']
                    short_ok = not daily['above_200']
                
                # LONG: 4H BB 하단 터치
                if long_ok and close[i] <= bb_lower[i]:
                    result = simulate(i, 'LONG', sl, tp)
                    result['direction'] = 'LONG'
                    result['datetime'] = dt
                    trades.append(result)
                    last_idx = i
                
                # SHORT: 4H BB 상단 터치
                elif short_ok and close[i] >= bb_upper[i]:
                    result = simulate(i, 'SHORT', sl, tp)
                    result['direction'] = 'SHORT'
                    result['datetime'] = dt
                    trades.append(result)
                    last_idx = i
            
            if len(trades) >= 20:
                df_t = pd.DataFrame(trades)
                win = len(df_t[df_t['reason'] == 'TP']) / len(df_t) * 100
                avg = df_t['pnl'].mean()
                monthly = avg * len(trades) / (5 * 12)  # 5년
                
                if avg > 0.5:
                    print(f"{condition:>20} {sl:>5} {tp:>5} {len(trades):>6} {win:>8.1f}% {avg:>8.2f}% {monthly:>8.2f}%")
                
                if avg > best_pnl:
                    best_pnl = avg
                    best_trades = df_t.copy()
                    best_params = {'condition': condition, 'sl': sl, 'tp': tp}


# ============================================================
# 전략 2: 추세 방향 + BB 터치 + RSI 필터
# ============================================================
print("\n" + "=" * 80)
print("전략 2: + RSI 필터 추가")
print("=" * 80)

print(f"\n{'RSI조건':>12} {'SL':>5} {'TP':>5} {'거래':>6} {'승률':>8} {'평균':>8} {'월수익':>8}")
print("-" * 65)

for rsi_th in [30, 35, 40]:
    for sl in [3, 4, 5]:
        for tp in [5, 7, 10, 15]:
            trades = []
            last_idx = 0
            
            for i in range(len(close) - 50):
                if i < last_idx + 6:
                    continue
                
                dt = pd.Timestamp(datetimes[i])
                daily = get_daily(dt)
                if daily is None:
                    continue
                
                # LONG: 일봉 상승 + 4H BB하단 + RSI 낮음
                if daily['above_200']:
                    if close[i] <= bb_lower[i] and rsi[i] < rsi_th:
                        result = simulate(i, 'LONG', sl, tp)
                        result['direction'] = 'LONG'
                        result['datetime'] = dt
                        result['rsi'] = rsi[i]
                        trades.append(result)
                        last_idx = i
                
                # SHORT: 일봉 하락 + 4H BB상단 + RSI 높음
                elif not daily['above_200']:
                    if close[i] >= bb_upper[i] and rsi[i] > (100 - rsi_th):
                        result = simulate(i, 'SHORT', sl, tp)
                        result['direction'] = 'SHORT'
                        result['datetime'] = dt
                        result['rsi'] = rsi[i]
                        trades.append(result)
                        last_idx = i
            
            if len(trades) >= 10:
                df_t = pd.DataFrame(trades)
                win = len(df_t[df_t['reason'] == 'TP']) / len(df_t) * 100
                avg = df_t['pnl'].mean()
                monthly = avg * len(trades) / (5 * 12)
                
                if avg > 0.5:
                    print(f"RSI<{rsi_th}/>{ 100-rsi_th:>2} {sl:>5} {tp:>5} {len(trades):>6} {win:>8.1f}% {avg:>8.2f}% {monthly:>8.2f}%")
                
                if avg > best_pnl:
                    best_pnl = avg
                    best_trades = df_t.copy()
                    best_params = {'rsi': rsi_th, 'sl': sl, 'tp': tp}


# ============================================================
# 전략 3: 추세 방향으로 진입 (순추세)
# ============================================================
print("\n" + "=" * 80)
print("전략 3: 일봉 추세 방향으로 돌파 진입 (순추세)")
print("=" * 80)

print(f"\n{'SL':>5} {'TP':>5} {'거래':>6} {'승률':>8} {'평균':>8} {'월수익':>8}")
print("-" * 50)

for sl in [3, 4, 5]:
    for tp in [5, 7, 10, 15]:
        trades = []
        last_idx = 0
        
        for i in range(len(close) - 50):
            if i < last_idx + 6:
                continue
            
            dt = pd.Timestamp(datetimes[i])
            daily = get_daily(dt)
            if daily is None:
                continue
            
            # LONG: 상승추세 + BB 상단 돌파 (모멘텀)
            if daily['above_200'] and daily['trend'] == 'UP':
                if close[i] >= bb_upper[i] and close[i-1] < bb_upper[i-1]:
                    result = simulate(i, 'LONG', sl, tp)
                    result['direction'] = 'LONG'
                    result['datetime'] = dt
                    trades.append(result)
                    last_idx = i
            
            # SHORT: 하락추세 + BB 하단 돌파
            elif not daily['above_200'] and daily['trend'] == 'DOWN':
                if close[i] <= bb_lower[i] and close[i-1] > bb_lower[i-1]:
                    result = simulate(i, 'SHORT', sl, tp)
                    result['direction'] = 'SHORT'
                    result['datetime'] = dt
                    trades.append(result)
                    last_idx = i
        
        if len(trades) >= 20:
            df_t = pd.DataFrame(trades)
            win = len(df_t[df_t['reason'] == 'TP']) / len(df_t) * 100
            avg = df_t['pnl'].mean()
            monthly = avg * len(trades) / (5 * 12)
            
            if avg > 0:
                print(f"{sl:>5} {tp:>5} {len(trades):>6} {win:>8.1f}% {avg:>8.2f}% {monthly:>8.2f}%")
            
            if avg > best_pnl:
                best_pnl = avg
                best_trades = df_t.copy()
                best_params = {'strategy': 'breakout', 'sl': sl, 'tp': tp}


# ============================================================
# 최적 결과
# ============================================================
if best_trades is not None and best_pnl > 0:
    print("\n" + "=" * 80)
    print("★★★ 최적 결과 ★★★")
    print("=" * 80)
    
    trades = best_trades
    tp_trades = trades[trades['reason'] == 'TP']
    sl_trades = trades[trades['reason'] == 'SL']
    timeout = trades[trades['reason'] == 'TIMEOUT']
    
    monthly_cnt = len(trades) / (5 * 12)
    monthly_pnl = trades['pnl'].mean() * monthly_cnt
    
    print(f"""
■ 파라미터: {best_params}

■ 결과:
  - 총 거래: {len(trades)}건 (월 {monthly_cnt:.1f}회)
  - 익절: {len(tp_trades)}건 ({len(tp_trades)/len(trades)*100:.1f}%)
  - 손절: {len(sl_trades)}건 ({len(sl_trades)/len(trades)*100:.1f}%)
  - 타임아웃: {len(timeout)}건

■ 손익:
  - 평균 PnL: {trades['pnl'].mean():+.2f}%
  - 월 기대 수익: {monthly_pnl:+.2f}%
  - 연 기대 수익: {monthly_pnl * 12:+.1f}%
  
■ 손익비:
  - 익절 평균: +{tp_trades['pnl'].mean():.2f}%
  - 손절 평균: {sl_trades['pnl'].mean():.2f}%
  - 비율: {abs(tp_trades['pnl'].mean() / sl_trades['pnl'].mean()):.2f}

■ MFE/MAE:
  - MFE 평균: {trades['mfe'].mean():.2f}%
  - MAE 평균: {trades['mae'].mean():.2f}%
  - 손절 전 MFE: {sl_trades['mfe'].mean():.2f}% (놓친 수익)
""")
    
    # 연도별
    trades['year'] = pd.to_datetime(trades['datetime']).dt.year
    print("■ 연도별:")
    for year, grp in trades.groupby('year'):
        win = len(grp[grp['reason']=='TP'])/len(grp)*100
        print(f"  {year}: {len(grp):>3}건, 승률 {win:.0f}%, 평균 {grp['pnl'].mean():+.2f}%, 총 {grp['pnl'].sum():+.1f}%")
    
    # 방향별
    print("\n■ 방향별:")
    for d in ['LONG', 'SHORT']:
        sub = trades[trades['direction'] == d]
        if len(sub) > 0:
            win = len(sub[sub['reason']=='TP'])/len(sub)*100
            print(f"  {d}: {len(sub)}건, 승률 {win:.0f}%, 평균 {sub['pnl'].mean():+.2f}%")
    
    trades.to_csv('mtf_v2_results.csv', index=False)
    print(f"\n저장: mtf_v2_results.csv")

else:
    print("\n\n수익나는 전략 없음")
