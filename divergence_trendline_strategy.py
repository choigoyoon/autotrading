import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# 데이터 로드
df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

# MACD 계산
def calc_macd(df, fast=12, slow=26, signal=9):
    exp1 = df['close'].ewm(span=fast).mean()
    exp2 = df['close'].ewm(span=slow).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

df_15m['macd'], df_15m['macd_signal'], df_15m['macd_hist'] = calc_macd(df_15m)
df_4h['macd'], df_4h['macd_signal'], df_4h['macd_hist'] = calc_macd(df_4h)

# 고점/저점 찾기
def find_peaks(df, order=5):
    highs = argrelextrema(df['high'].values, np.greater, order=order)[0]
    lows = argrelextrema(df['low'].values, np.less, order=order)[0]
    return highs, lows

# 하락 추세선 생성 (고점 연결)
def get_down_trendlines(df, high_idx):
    trendlines = []
    for i in range(len(high_idx) - 1):
        idx1, idx2 = high_idx[i], high_idx[i+1]
        h1, h2 = df.iloc[idx1]['high'], df.iloc[idx2]['high']
        if h1 > h2:  # 하락 추세
            slope = (h2 - h1) / (idx2 - idx1)
            trendlines.append({
                'start_idx': idx1,
                'start_price': h1,
                'slope': slope,
                'start_time': df.iloc[idx1]['datetime']
            })
    return trendlines

# 다이버전스 감지 (가격 저점↘ + MACD 저점↗)
def detect_divergence(df, low_idx, lookback=3):
    divergences = []
    for i in range(lookback, len(low_idx)):
        curr_idx = low_idx[i]
        prev_idx = low_idx[i-1]
        
        curr_price = df.iloc[curr_idx]['low']
        prev_price = df.iloc[prev_idx]['low']
        curr_macd = df.iloc[curr_idx]['macd_hist']
        prev_macd = df.iloc[prev_idx]['macd_hist']
        
        # 가격 저점↘ + MACD 저점↗ = 상승 다이버전스
        if curr_price < prev_price and curr_macd > prev_macd:
            divergences.append({
                'idx': curr_idx,
                'time': df.iloc[curr_idx]['datetime'],
                'price': curr_price,
                'macd': curr_macd
            })
    return divergences

# 고점 올림 확인 (HH)
def check_higher_high(df, from_idx, lookback=50):
    """from_idx 이후로 고점이 올라갔는지 확인"""
    if from_idx + lookback >= len(df):
        return False, None
    
    window = df.iloc[from_idx:from_idx+lookback]
    highs, _ = find_peaks(window, order=3)
    
    if len(highs) >= 2:
        h1 = window.iloc[highs[0]]['high']
        h2 = window.iloc[highs[1]]['high']
        if h2 > h1:  # 고점 올림
            return True, from_idx + highs[1]
    return False, None

# FVG 감지
def detect_fvg(df):
    fvgs = []
    for i in range(2, len(df)):
        prev2 = df.iloc[i-2]
        curr = df.iloc[i]
        if prev2['high'] < curr['low']:  # 상승 FVG
            fvgs.append({
                'idx': i,
                'time': curr['datetime'],
                'top': curr['low'],
                'bottom': prev2['high'],
                'body_size': abs(curr['close'] - curr['open']) / curr['open'] * 100
            })
    return fvgs

# 추세선 돌파 확인
def check_trendline_break(df, trendlines, idx):
    """현재 가격이 하락 추세선 위에 있는지"""
    price = df.iloc[idx]['close']
    
    for tl in trendlines:
        if tl['start_idx'] < idx:
            # 현재 시점의 추세선 가격
            tl_price = tl['start_price'] + tl['slope'] * (idx - tl['start_idx'])
            if price > tl_price:
                return True, tl_price
    return False, None

print("=" * 70)
print("🎯 다이버전스 + 추세선 + FVG 전략 백테스트")
print("=" * 70)

# 4H 기준 분석
high_idx_4h, low_idx_4h = find_peaks(df_4h, order=3)
trendlines_4h = get_down_trendlines(df_4h, high_idx_4h)
divergences_4h = detect_divergence(df_4h, low_idx_4h)
fvgs_4h = detect_fvg(df_4h)

print(f"\n[4H 데이터 분석]")
print(f"  하락 추세선: {len(trendlines_4h)}개")
print(f"  다이버전스: {len(divergences_4h)}개")
print(f"  FVG: {len(fvgs_4h)}개")

# 전략 시뮬레이션
def simulate_strategy(df_4h, df_15m, tp_pct=1.5, sl_pct=-1.0):
    trades = []
    
    high_idx, low_idx = find_peaks(df_4h, order=3)
    trendlines = get_down_trendlines(df_4h, high_idx)
    divergences = detect_divergence(df_4h, low_idx)
    fvgs = detect_fvg(df_4h)
    
    # 각 다이버전스 이후 체크
    for div in divergences:
        div_idx = div['idx']
        div_time = div['time']
        
        # 1. 고점 올림 확인
        hh_found, hh_idx = check_higher_high(df_4h, div_idx)
        if not hh_found:
            continue
        
        hh_time = df_4h.iloc[hh_idx]['datetime']
        
        # 2. 그 이후 FVG 찾기
        for fvg in fvgs:
            if fvg['time'] > hh_time:
                fvg_time = fvg['time']
                fvg_top = fvg['top']
                
                # 3. 추세선 돌파 확인
                broke, tl_price = check_trendline_break(df_4h, trendlines, fvg['idx'])
                
                # 4. 진입 (FVG 터치)
                future_15m = df_15m[df_15m['datetime'] > fvg_time].head(200)
                if len(future_15m) < 10:
                    continue
                
                entry_idx = None
                for i, (idx, row) in enumerate(future_15m.iterrows()):
                    if row['low'] <= fvg_top:
                        if i + 1 < len(future_15m):
                            entry_idx = i + 1
                        break
                
                if entry_idx is None:
                    continue
                
                entry_row = future_15m.iloc[entry_idx]
                entry_price = entry_row['open']
                entry_time = entry_row['datetime']
                
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 + sl_pct / 100)
                
                # 결과 판정
                after = future_15m.iloc[entry_idx + 1:]
                result = None
                for _, row in after.iterrows():
                    if row['low'] <= sl_price:
                        result = 'loss'
                        break
                    if row['high'] >= tp_price:
                        result = 'win'
                        break
                
                if result:
                    trades.append({
                        'time': entry_time,
                        'result': result,
                        'pnl': tp_pct - 0.06 if result == 'win' else sl_pct - 0.06,
                        'div_time': div_time,
                        'hh_time': hh_time,
                        'fvg_time': fvg_time,
                        'tl_break': broke
                    })
                break  # 하나의 다이버전스당 하나의 진입
    
    return trades

# TP/SL 조합 테스트
print("\n" + "=" * 70)
print("📊 TP/SL 조합별 결과")
print("=" * 70)

print(f"\n{'TP':>5} {'SL':>5} {'거래':>6} {'승률':>7} {'총PnL':>8} {'월평균':>8}")
print("-" * 50)

best = None
for tp in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for sl in [-0.5, -1.0, -1.5, -2.0]:
        trades = simulate_strategy(df_4h, df_15m, tp, sl)
        if len(trades) < 5:
            continue
        
        wins = sum(1 for t in trades if t['result'] == 'win')
        wr = wins / len(trades) * 100
        total_pnl = sum(t['pnl'] for t in trades)
        
        df_t = pd.DataFrame(trades)
        df_t['month'] = pd.to_datetime(df_t['time']).dt.to_period('M')
        months = len(df_t['month'].unique())
        monthly = total_pnl / months if months > 0 else 0
        
        print(f"{tp:>5.1f} {sl:>5.1f} {len(trades):>6} {wr:>6.1f}% {total_pnl:>7.1f}% {monthly:>7.2f}%")
        
        if best is None or wr > best['wr']:
            best = {'tp': tp, 'sl': sl, 'trades': trades, 'wr': wr, 'pnl': total_pnl, 'monthly': monthly}

# 최고 결과 상세
if best:
    print("\n" + "=" * 70)
    print(f"🏆 최적 전략: TP {best['tp']}% / SL {best['sl']}%")
    print("=" * 70)
    
    trades = best['trades']
    df_t = pd.DataFrame(trades)
    
    # MDD 계산
    cum, peak, mdd = 0, 0, 0
    for t in trades:
        cum += t['pnl']
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    
    print(f"\n  거래 수: {len(trades)}")
    print(f"  승률: {best['wr']:.1f}%")
    print(f"  총 PnL: {best['pnl']:.1f}%")
    print(f"  MDD: {mdd:.1f}%")
    print(f"  월평균: {best['monthly']:.2f}%")
    
    # 추세선 돌파 여부별 성과
    with_break = [t for t in trades if t['tl_break']]
    without_break = [t for t in trades if not t['tl_break']]
    
    if with_break:
        wr_break = sum(1 for t in with_break if t['result'] == 'win') / len(with_break) * 100
        print(f"\n  [추세선 돌파 O]: {len(with_break)}회, 승률 {wr_break:.1f}%")
    if without_break:
        wr_no = sum(1 for t in without_break if t['result'] == 'win') / len(without_break) * 100
        print(f"  [추세선 돌파 X]: {len(without_break)}회, 승률 {wr_no:.1f}%")

# 기존 전략과 비교
print("\n" + "=" * 70)
print("📈 기존 전략 vs 새 전략 비교")
print("=" * 70)

# 기존 (단순 FVG)
old_trades = []
for fvg in fvgs_4h:
    future = df_15m[df_15m['datetime'] > fvg['time']].head(200)
    if len(future) < 10:
        continue
    
    entry_idx = None
    for i, (idx, row) in enumerate(future.iterrows()):
        if row['low'] <= fvg['top']:
            if i + 1 < len(future):
                entry_idx = i + 1
            break
    
    if entry_idx is None:
        continue
    
    ep = future.iloc[entry_idx]['open']
    et = future.iloc[entry_idx]['datetime']
    tp_p, sl_p = ep * 1.01, ep * 0.985
    
    after = future.iloc[entry_idx + 1:]
    for _, row in after.iterrows():
        if row['low'] <= sl_p:
            old_trades.append({'time': et, 'result': 'loss', 'pnl': -1.56})
            break
        if row['high'] >= tp_p:
            old_trades.append({'time': et, 'result': 'win', 'pnl': 0.94})
            break

old_wr = sum(1 for t in old_trades if t['result'] == 'win') / len(old_trades) * 100 if old_trades else 0
old_pnl = sum(t['pnl'] for t in old_trades)

if best:
    print(f"\n{'지표':<15} {'기존(FVG만)':>15} {'새전략(다이버+추세)':>20}")
    print("-" * 55)
    print(f"{'거래 수':<15} {len(old_trades):>15} {len(best['trades']):>20}")
    print(f"{'승률':<15} {old_wr:>14.1f}% {best['wr']:>19.1f}%")
    print(f"{'총 PnL':<15} {old_pnl:>14.1f}% {best['pnl']:>19.1f}%")

