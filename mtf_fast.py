#!/usr/bin/env python3
"""
MTF 전략 - 최적화 버전
4시간봉 매매, 일봉 추세 필터
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("MTF 전략 - 4시간봉 매매")
print("=" * 80)

# 데이터 로드
df_1d = pd.read_csv('analysis_1d.csv', parse_dates=['datetime'])
df_4h = pd.read_csv('analysis_4h.csv', parse_dates=['datetime'])

print(f"일봉: {len(df_1d):,}개")
print(f"4시간: {len(df_4h):,}개")

# BB, RSI, EMA 계산
def add_indicators(df):
    # BB
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    
    # EMA
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    return df.dropna().reset_index(drop=True)

df_1d = add_indicators(df_1d)
df_4h = add_indicators(df_4h)

# 일봉 추세를 4시간에 매핑 (빠른 방법)
df_1d['date'] = df_1d['datetime'].dt.date
df_1d['trend'] = np.where(df_1d['ema20'] > df_1d['ema50'], 'UP', 'DOWN')
df_1d['above_200'] = df_1d['close'] > df_1d['ema200']

daily_trend = df_1d.set_index('date')[['trend', 'above_200', 'rsi']].to_dict('index')

def get_daily_trend(dt):
    d = dt.date()
    for i in range(5):  # 최대 5일 이전까지 찾기
        check_date = d - pd.Timedelta(days=i)
        if check_date in daily_trend:
            return daily_trend[check_date]
    return None

# numpy 배열로 변환 (속도)
close_4h = df_4h['close'].values
high_4h = df_4h['high'].values
low_4h = df_4h['low'].values
bb_upper = df_4h['bb_upper'].values
bb_lower = df_4h['bb_lower'].values
rsi_4h = df_4h['rsi'].values
datetimes_4h = df_4h['datetime'].values

COST = 0.18

def simulate(idx, direction, sl_pct, tp_pct, max_bars=30):
    """매매 시뮬레이션"""
    entry = close_4h[idx]
    
    if direction == 'LONG':
        sl = entry * (1 - sl_pct / 100)
        tp = entry * (1 + tp_pct / 100)
    else:
        sl = entry * (1 + sl_pct / 100)
        tp = entry * (1 - tp_pct / 100)
    
    mfe = 0
    for j in range(idx + 1, min(idx + max_bars + 1, len(close_4h))):
        if direction == 'LONG':
            mfe = max(mfe, (high_4h[j] - entry) / entry * 100)
            if low_4h[j] <= sl:
                return {'pnl': -sl_pct - COST, 'reason': 'SL', 'bars': j - idx, 'mfe': mfe}
            if high_4h[j] >= tp:
                return {'pnl': tp_pct - COST, 'reason': 'TP', 'bars': j - idx, 'mfe': mfe}
        else:
            mfe = max(mfe, (entry - low_4h[j]) / entry * 100)
            if high_4h[j] >= sl:
                return {'pnl': -sl_pct - COST, 'reason': 'SL', 'bars': j - idx, 'mfe': mfe}
            if low_4h[j] <= tp:
                return {'pnl': tp_pct - COST, 'reason': 'TP', 'bars': j - idx, 'mfe': mfe}
    
    # 타임아웃
    exit_price = close_4h[min(idx + max_bars, len(close_4h) - 1)]
    if direction == 'LONG':
        pnl = (exit_price - entry) / entry * 100
    else:
        pnl = (entry - exit_price) / entry * 100
    return {'pnl': pnl - COST, 'reason': 'TIMEOUT', 'bars': max_bars, 'mfe': mfe}


# ============================================================
# 전략: 일봉 추세 + 4시간 BB 터치
# ============================================================
print("\n" + "=" * 80)
print("전략: 일봉 추세 방향 + 4시간 BB 터치")
print("=" * 80)

print(f"\n{'SL%':>6} {'TP%':>6} {'거래':>6} {'승률':>8} {'평균PnL':>10} {'월수익':>10}")
print("-" * 60)

best_trades = None
best_expected = -999

for sl_pct in [3.0, 4.0, 5.0]:
    for tp_pct in [5.0, 7.0, 10.0, 15.0]:
        trades = []
        last_idx = 0
        
        for i in range(200, len(close_4h) - 50):
            if i < last_idx + 6:  # 24시간 간격
                continue
            
            dt = pd.Timestamp(datetimes_4h[i])
            daily = get_daily_trend(dt)
            if daily is None:
                continue
            
            # LONG: 일봉 상승추세 + 4H BB 하단 터치
            if daily['trend'] == 'UP' and daily['above_200']:
                if close_4h[i] <= bb_lower[i]:
                    result = simulate(i, 'LONG', sl_pct, tp_pct)
                    result['direction'] = 'LONG'
                    result['datetime'] = dt
                    trades.append(result)
                    last_idx = i
            
            # SHORT: 일봉 하락추세 + 4H BB 상단 터치
            elif daily['trend'] == 'DOWN' and not daily['above_200']:
                if close_4h[i] >= bb_upper[i]:
                    result = simulate(i, 'SHORT', sl_pct, tp_pct)
                    result['direction'] = 'SHORT'
                    result['datetime'] = dt
                    trades.append(result)
                    last_idx = i
        
        if len(trades) >= 20:
            df_t = pd.DataFrame(trades)
            win_rate = len(df_t[df_t['reason'] == 'TP']) / len(df_t) * 100
            avg_pnl = df_t['pnl'].mean()
            monthly = avg_pnl * len(df_t) / (5.5 * 12)
            
            print(f"{sl_pct:>6.1f} {tp_pct:>6.1f} {len(trades):>6} {win_rate:>8.1f}% {avg_pnl:>10.2f}% {monthly:>10.2f}%")
            
            if avg_pnl > best_expected:
                best_expected = avg_pnl
                best_trades = df_t.copy()
                best_params = {'sl': sl_pct, 'tp': tp_pct}


# ============================================================
# 추가: RSI 필터
# ============================================================
print("\n" + "=" * 80)
print("RSI 필터 추가 (4H RSI < 30 LONG, > 70 SHORT)")
print("=" * 80)

print(f"\n{'SL%':>6} {'TP%':>6} {'거래':>6} {'승률':>8} {'평균PnL':>10} {'월수익':>10}")
print("-" * 60)

for sl_pct in [3.0, 4.0, 5.0]:
    for tp_pct in [5.0, 7.0, 10.0, 15.0]:
        trades = []
        last_idx = 0
        
        for i in range(200, len(close_4h) - 50):
            if i < last_idx + 6:
                continue
            
            dt = pd.Timestamp(datetimes_4h[i])
            daily = get_daily_trend(dt)
            if daily is None:
                continue
            
            # LONG: 일봉 상승 + 4H BB하단 + RSI < 30
            if daily['trend'] == 'UP' and daily['above_200']:
                if close_4h[i] <= bb_lower[i] and rsi_4h[i] < 30:
                    result = simulate(i, 'LONG', sl_pct, tp_pct)
                    result['direction'] = 'LONG'
                    result['datetime'] = dt
                    result['rsi'] = rsi_4h[i]
                    trades.append(result)
                    last_idx = i
            
            # SHORT: 일봉 하락 + 4H BB상단 + RSI > 70
            elif daily['trend'] == 'DOWN' and not daily['above_200']:
                if close_4h[i] >= bb_upper[i] and rsi_4h[i] > 70:
                    result = simulate(i, 'SHORT', sl_pct, tp_pct)
                    result['direction'] = 'SHORT'
                    result['datetime'] = dt
                    result['rsi'] = rsi_4h[i]
                    trades.append(result)
                    last_idx = i
        
        if len(trades) >= 10:
            df_t = pd.DataFrame(trades)
            win_rate = len(df_t[df_t['reason'] == 'TP']) / len(df_t) * 100
            avg_pnl = df_t['pnl'].mean()
            monthly = avg_pnl * len(df_t) / (5.5 * 12)
            
            print(f"{sl_pct:>6.1f} {tp_pct:>6.1f} {len(trades):>6} {win_rate:>8.1f}% {avg_pnl:>10.2f}% {monthly:>10.2f}%")
            
            if avg_pnl > best_expected:
                best_expected = avg_pnl
                best_trades = df_t.copy()
                best_params = {'sl': sl_pct, 'tp': tp_pct, 'rsi': True}


# ============================================================
# 최적 결과
# ============================================================
if best_trades is not None and best_expected > 0:
    print("\n" + "=" * 80)
    print("★★★ 최적 결과 ★★★")
    print("=" * 80)
    
    trades = best_trades
    tp = trades[trades['reason'] == 'TP']
    sl = trades[trades['reason'] == 'SL']
    timeout = trades[trades['reason'] == 'TIMEOUT']
    
    monthly_trades = len(trades) / (5.5 * 12)
    monthly_pnl = trades['pnl'].mean() * monthly_trades
    
    print(f"""
■ 파라미터: SL {best_params['sl']}%, TP {best_params['tp']}%

■ 결과:
  - 총 거래: {len(trades)}건 (월 {monthly_trades:.1f}회)
  - 익절: {len(tp)}건 ({len(tp)/len(trades)*100:.1f}%)
  - 손절: {len(sl)}건 ({len(sl)/len(trades)*100:.1f}%)
  - 타임아웃: {len(timeout)}건
  
■ 손익:
  - 평균 PnL: {trades['pnl'].mean():+.2f}%
  - 월 기대 수익: {monthly_pnl:+.2f}%
  - 연 기대 수익: {monthly_pnl * 12:+.1f}%
  
■ MFE 분석:
  - 평균 MFE: {trades['mfe'].mean():.2f}%
  - 손절 전 MFE: {sl['mfe'].mean():.2f}% (놓친 수익)
""")
    
    # 연도별
    trades['year'] = pd.to_datetime(trades['datetime']).dt.year
    print("■ 연도별:")
    for year, grp in trades.groupby('year'):
        print(f"  {year}: {len(grp)}건, {grp['pnl'].mean():+.2f}%, 총 {grp['pnl'].sum():+.1f}%")
    
    # 방향별
    print("\n■ 방향별:")
    for d in ['LONG', 'SHORT']:
        sub = trades[trades['direction'] == d]
        if len(sub) > 0:
            print(f"  {d}: {len(sub)}건, {sub['pnl'].mean():+.2f}%")
    
    trades.to_csv('mtf_fast_results.csv', index=False)
    print(f"\n저장: mtf_fast_results.csv")

else:
    print("\n플러스 기대값 전략 없음")
