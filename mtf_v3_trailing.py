#!/usr/bin/env python3
"""
MTF 전략 v3 - 트레일링 스탑 적용
타임아웃에서 수익 나고있던 것들을 살리기
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("MTF 전략 v3 - 트레일링 스탑")
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
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    return df.iloc[200:].reset_index(drop=True)

df_1d = add_indicators(df_1d)
df_4h = add_indicators(df_4h)

# 일봉 추세
df_1d['trend'] = np.where(df_1d['ema20'] > df_1d['ema50'], 'UP', 'DOWN')
df_1d['above_200'] = df_1d['close'] > df_1d['ema200']
df_1d['date'] = df_1d['datetime'].dt.date
daily_map = df_1d.set_index('date')[['trend', 'above_200']].to_dict('index')

# numpy
close = df_4h['close'].values
high = df_4h['high'].values
low = df_4h['low'].values
bb_upper = df_4h['bb_upper'].values
bb_lower = df_4h['bb_lower'].values
datetimes = df_4h['datetime'].values

COST = 0.18

def get_daily(dt):
    d = dt.date()
    for i in range(10):
        check = d - pd.Timedelta(days=i)
        if check in daily_map:
            return daily_map[check]
    return None


def simulate_trailing(idx, direction, initial_sl_pct, tp_pct, trail_pct, max_bars=30):
    """
    트레일링 스탑 시뮬레이션
    - initial_sl_pct: 초기 손절 %
    - tp_pct: 익절 목표 %
    - trail_pct: 트레일링 간격 % (최고점에서 이만큼 빠지면 청산)
    """
    entry = close[idx]
    
    if direction == 'LONG':
        sl = entry * (1 - initial_sl_pct / 100)
        tp = entry * (1 + tp_pct / 100)
        highest = entry
    else:
        sl = entry * (1 + initial_sl_pct / 100)
        tp = entry * (1 - tp_pct / 100)
        lowest = entry
    
    mfe = 0
    trail_activated = False
    
    for j in range(idx + 1, min(idx + max_bars + 1, len(close))):
        if direction == 'LONG':
            mfe = max(mfe, (high[j] - entry) / entry * 100)
            
            # 최고점 갱신
            if high[j] > highest:
                highest = high[j]
                # 수익 3% 넘으면 트레일링 활성화
                if (highest - entry) / entry * 100 > 3:
                    trail_activated = True
                    new_sl = highest * (1 - trail_pct / 100)
                    sl = max(sl, new_sl)
            
            # 손절
            if low[j] <= sl:
                pnl = (sl - entry) / entry * 100
                reason = 'TRAIL_SL' if trail_activated else 'SL'
                return {'pnl': pnl - COST, 'reason': reason, 'bars': j - idx, 'mfe': mfe}
            
            # 익절
            if high[j] >= tp:
                return {'pnl': tp_pct - COST, 'reason': 'TP', 'bars': j - idx, 'mfe': mfe}
        
        else:  # SHORT
            mfe = max(mfe, (entry - low[j]) / entry * 100)
            
            if low[j] < lowest:
                lowest = low[j]
                if (entry - lowest) / entry * 100 > 3:
                    trail_activated = True
                    new_sl = lowest * (1 + trail_pct / 100)
                    sl = min(sl, new_sl)
            
            if high[j] >= sl:
                pnl = (entry - sl) / entry * 100
                reason = 'TRAIL_SL' if trail_activated else 'SL'
                return {'pnl': pnl - COST, 'reason': reason, 'bars': j - idx, 'mfe': mfe}
            
            if low[j] <= tp:
                return {'pnl': tp_pct - COST, 'reason': 'TP', 'bars': j - idx, 'mfe': mfe}
    
    # 타임아웃
    exit_price = close[min(idx + max_bars, len(close) - 1)]
    if direction == 'LONG':
        pnl = (exit_price - entry) / entry * 100
    else:
        pnl = (entry - exit_price) / entry * 100
    return {'pnl': pnl - COST, 'reason': 'TIMEOUT', 'bars': max_bars, 'mfe': mfe}


# ============================================================
# 테스트: 트레일링 스탑 조합
# ============================================================
print("\n" + "=" * 80)
print("트레일링 스탑 테스트")
print("=" * 80)

print(f"\n{'SL':>4} {'TP':>4} {'Trail':>6} {'거래':>6} {'승률':>8} {'평균':>8} {'월수익':>8}")
print("-" * 55)

best_trades = None
best_pnl = -999

for initial_sl in [4, 5, 6]:
    for tp in [10, 15, 20]:
        for trail in [2, 3, 4]:
            trades = []
            last_idx = 0
            
            for i in range(len(close) - 50):
                if i < last_idx + 6:
                    continue
                
                dt = pd.Timestamp(datetimes[i])
                daily = get_daily(dt)
                if daily is None:
                    continue
                
                # LONG: 상승추세 + BB 상단 돌파
                if daily['above_200'] and daily['trend'] == 'UP':
                    if close[i] >= bb_upper[i] and close[i-1] < bb_upper[i-1]:
                        result = simulate_trailing(i, 'LONG', initial_sl, tp, trail)
                        result['direction'] = 'LONG'
                        result['datetime'] = dt
                        trades.append(result)
                        last_idx = i
                
                # SHORT: 하락추세 + BB 하단 돌파
                elif not daily['above_200'] and daily['trend'] == 'DOWN':
                    if close[i] <= bb_lower[i] and close[i-1] > bb_lower[i-1]:
                        result = simulate_trailing(i, 'SHORT', initial_sl, tp, trail)
                        result['direction'] = 'SHORT'
                        result['datetime'] = dt
                        trades.append(result)
                        last_idx = i
            
            if len(trades) >= 20:
                df_t = pd.DataFrame(trades)
                # 승률 = TP + TRAIL_SL(양수) 
                wins = df_t[(df_t['reason'] == 'TP') | ((df_t['reason'] == 'TRAIL_SL') & (df_t['pnl'] > 0))]
                win = len(wins) / len(df_t) * 100
                avg = df_t['pnl'].mean()
                monthly = avg * len(trades) / (5 * 12)
                
                print(f"{initial_sl:>4} {tp:>4} {trail:>6} {len(trades):>6} {win:>8.1f}% {avg:>8.2f}% {monthly:>8.2f}%")
                
                if avg > best_pnl:
                    best_pnl = avg
                    best_trades = df_t.copy()
                    best_params = {'sl': initial_sl, 'tp': tp, 'trail': trail}


# ============================================================
# 최적 결과
# ============================================================
if best_trades is not None and best_pnl > 0:
    print("\n" + "=" * 80)
    print("★★★ 최적 결과 ★★★")
    print("=" * 80)
    
    trades = best_trades
    
    print(f"\n■ 파라미터:")
    print(f"  - 초기 손절: {best_params['sl']}%")
    print(f"  - 익절 목표: {best_params['tp']}%")
    print(f"  - 트레일링: {best_params['trail']}% (3% 수익 후 활성화)")
    
    print(f"\n■ 청산 분포:")
    for reason in trades['reason'].unique():
        sub = trades[trades['reason'] == reason]
        print(f"  {reason:>10}: {len(sub):>3}건 ({len(sub)/len(trades)*100:>5.1f}%), 평균 {sub['pnl'].mean():+.2f}%")
    
    monthly_cnt = len(trades) / (5 * 12)
    monthly_pnl = trades['pnl'].mean() * monthly_cnt
    
    print(f"""
■ 전체 성과:
  - 총 거래: {len(trades)}건 (월 {monthly_cnt:.1f}회)
  - 평균 PnL: {trades['pnl'].mean():+.2f}%
  - 월 기대 수익: {monthly_pnl:+.2f}%
  - 연 기대 수익: {monthly_pnl * 12:+.1f}%
""")
    
    # 연도별
    trades['year'] = pd.to_datetime(trades['datetime']).dt.year
    print("■ 연도별:")
    for year, grp in trades.groupby('year'):
        wins = grp[(grp['reason'] == 'TP') | ((grp['reason'] == 'TRAIL_SL') & (grp['pnl'] > 0))]
        win_rate = len(wins) / len(grp) * 100
        print(f"  {year}: {len(grp):>3}건, 승률 {win_rate:>5.1f}%, 평균 {grp['pnl'].mean():+.2f}%, 총 {grp['pnl'].sum():+.1f}%")
    
    # 방향별
    print("\n■ 방향별:")
    for d in ['LONG', 'SHORT']:
        sub = trades[trades['direction'] == d]
        if len(sub) > 0:
            wins = sub[(sub['reason'] == 'TP') | ((sub['reason'] == 'TRAIL_SL') & (sub['pnl'] > 0))]
            print(f"  {d}: {len(sub)}건, 승률 {len(wins)/len(sub)*100:.1f}%, 평균 {sub['pnl'].mean():+.2f}%")
    
    # 월 15% 달성 가능성
    print(f"\n■ 월 15% 달성 분석:")
    print(f"  현재 월 수익: {monthly_pnl:.2f}%")
    print(f"  목표 대비: {monthly_pnl/15*100:.1f}%")
    
    if monthly_pnl >= 15:
        print("  → 월 15% 달성 가능!")
    else:
        needed_per_trade = 15 / monthly_cnt
        print(f"  → 부족. 거래당 {needed_per_trade:.1f}% 필요 (현재 {trades['pnl'].mean():.2f}%)")
    
    trades.to_csv('mtf_v3_results.csv', index=False)
    print(f"\n저장: mtf_v3_results.csv")

else:
    print("\n수익나는 전략 없음")
