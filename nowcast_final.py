"""
나우캐스트 최종 최적화 - 발견된 양수 수익 설정 심화
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("나우캐스트 최종 최적화")
print("=" * 70)

# 데이터
df = pd.read_csv('btc_15m_ohlcv.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna().reset_index(drop=True)

breakouts = pd.read_csv('nowcast_breakouts.csv').to_dict('records')

# 지표
df['vol_ma20'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['vol_ma20']
df['ma50'] = df['close'].rolling(50).mean()
df['mom_5'] = df['close'].pct_change(5) * 100

delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + gain / loss))

total_days = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days
months = total_days / 30
print(f"데이터: {len(df)}봉 ({months:.1f}개월)")


def backtest(tp, sl, interval, fvg_only, vol_min, trend, rsi_max, mom_min, details=False):
    open_p = df['open'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    vol_ratio = df['vol_ratio'].values
    ma50_arr = df['ma50'].values
    rsi_arr = df['rsi'].values
    mom5_arr = df['mom_5'].values
    
    trades = []
    last_entry = -interval - 1
    
    for brk in breakouts:
        i = brk['idx']
        
        if i - last_entry < interval:
            continue
        if brk['type'] != 'long':
            continue
        
        if fvg_only and not brk['has_fvg']:
            continue
        if vol_min > 0 and vol_ratio[i] < vol_min:
            continue
        if trend and close[i] < ma50_arr[i]:
            continue
        if rsi_max > 0 and rsi_arr[i] > rsi_max:
            continue
        if mom_min is not None and mom5_arr[i] < mom_min:
            continue
        
        entry_idx = i + 1
        if entry_idx >= len(df) - 50:
            continue
        
        entry_price = open_p[entry_idx]
        tp_level = entry_price * (1 + tp / 100)
        sl_level = entry_price * (1 - sl / 100)
        
        result = 'TIMEOUT'
        exit_price = close[min(entry_idx + 49, len(df) - 1)]
        exit_idx = entry_idx + 49
        
        for j in range(entry_idx + 1, min(entry_idx + 50, len(df))):
            if high[j] >= tp_level:
                result, exit_price, exit_idx = 'TP', tp_level, j
                break
            if low[j] <= sl_level:
                result, exit_price, exit_idx = 'SL', sl_level, j
                break
        
        pnl = (exit_price - entry_price) / entry_price * 100
        trades.append({
            'pnl': pnl, 'result': result, 
            'entry_idx': entry_idx, 'hold': exit_idx - entry_idx
        })
        last_entry = i
    
    if not trades:
        return 0, 0, 0, 0, []
    
    pnls = [t['pnl'] for t in trades]
    results = [t['result'] for t in trades]
    n = len(trades)
    wr = sum(1 for p in pnls if p > 0) / n * 100
    avg = np.mean(pnls)
    total = sum(pnls)
    
    if details:
        return n, wr, avg, total, trades
    return n, wr, avg, total, []


# 1. FVG+Vol+RSI 세밀 탐색 (최고 순수익)
print("\n" + "=" * 70)
print("1. FVG + Vol + RSI 세밀 탐색")
print("=" * 70)

results = []
for tp in [4.0, 4.5, 5.0, 5.5, 6.0]:
    for sl in [0.3, 0.5, 0.7, 1.0]:
        for vol in [0.8, 1.0, 1.2]:
            for rsi in [45, 50, 55]:
                for intv in [8, 10, 12]:
                    n, wr, avg, total, _ = backtest(tp, sl, intv, True, vol, False, rsi, None)
                    if n >= 30:
                        net = avg - 0.11
                        results.append({
                            'tp': tp, 'sl': sl, 'vol': vol, 'rsi': rsi, 'intv': intv,
                            'n': n, 'wr': wr, 'net': net, 'total': total
                        })

df1 = pd.DataFrame(results).sort_values('net', ascending=False)
print(df1.head(15).to_string(index=False))


# 2. FVG + Mom 세밀 탐색 (두번째 높은 순수익)
print("\n" + "=" * 70)
print("2. FVG + 모멘텀 세밀 탐색")
print("=" * 70)

results = []
for tp in [4.0, 4.5, 5.0, 5.5, 6.0]:
    for sl in [0.5, 0.7, 1.0]:
        for mom in [0, 1, 2]:
            for intv in [8, 10, 12]:
                n, wr, avg, total, _ = backtest(tp, sl, intv, True, 0, False, 0, mom)
                if n >= 30:
                    net = avg - 0.11
                    results.append({
                        'tp': tp, 'sl': sl, 'mom': mom, 'intv': intv,
                        'n': n, 'wr': wr, 'net': net, 'total': total
                    })

df2 = pd.DataFrame(results).sort_values('net', ascending=False)
print(df2.head(15).to_string(index=False))


# 3. FVG + Vol 세밀 탐색 (안정적)
print("\n" + "=" * 70)
print("3. FVG + Vol 세밀 탐색")
print("=" * 70)

results = []
for tp in [4.0, 4.5, 5.0, 5.5, 6.0]:
    for sl in [0.5, 0.7, 1.0]:
        for vol in [1.0, 1.2, 1.5, 2.0]:
            for intv in [8, 10, 12, 15]:
                n, wr, avg, total, _ = backtest(tp, sl, intv, True, vol, False, 0, None)
                if n >= 50:
                    net = avg - 0.11
                    results.append({
                        'tp': tp, 'sl': sl, 'vol': vol, 'intv': intv,
                        'n': n, 'wr': wr, 'net': net, 'total': total
                    })

df3 = pd.DataFrame(results).sort_values('net', ascending=False)
print(df3.head(15).to_string(index=False))


# 4. 최종 후보 상세 분석
print("\n" + "=" * 70)
print("4. 최종 후보 상세 분석")
print("=" * 70)

candidates = []

# FVG+Vol+RSI 최고
if len(df1) > 0:
    b = df1.iloc[0]
    n, wr, avg, total, trades = backtest(b['tp'], b['sl'], int(b['intv']), True, b['vol'], False, int(b['rsi']), None, details=True)
    tp_cnt = sum(1 for t in trades if t['result'] == 'TP')
    sl_cnt = sum(1 for t in trades if t['result'] == 'SL')
    to_cnt = sum(1 for t in trades if t['result'] == 'TIMEOUT')
    candidates.append({
        'name': f"FVG+Vol>{b['vol']}+RSI<{int(b['rsi'])}",
        'params': f"TP{b['tp']} SL{b['sl']} INT{int(b['intv'])}",
        'trades': n, 'win_rate': wr, 'net_pnl': avg - 0.11,
        'total_pnl': total, 'monthly': total / months,
        'tp': tp_cnt, 'sl': sl_cnt, 'timeout': to_cnt
    })

# FVG+Mom 최고
if len(df2) > 0:
    b = df2.iloc[0]
    n, wr, avg, total, trades = backtest(b['tp'], b['sl'], int(b['intv']), True, 0, False, 0, int(b['mom']), details=True)
    tp_cnt = sum(1 for t in trades if t['result'] == 'TP')
    sl_cnt = sum(1 for t in trades if t['result'] == 'SL')
    to_cnt = sum(1 for t in trades if t['result'] == 'TIMEOUT')
    candidates.append({
        'name': f"FVG+Mom>{int(b['mom'])}",
        'params': f"TP{b['tp']} SL{b['sl']} INT{int(b['intv'])}",
        'trades': n, 'win_rate': wr, 'net_pnl': avg - 0.11,
        'total_pnl': total, 'monthly': total / months,
        'tp': tp_cnt, 'sl': sl_cnt, 'timeout': to_cnt
    })

# FVG+Vol 최고
if len(df3) > 0:
    b = df3.iloc[0]
    n, wr, avg, total, trades = backtest(b['tp'], b['sl'], int(b['intv']), True, b['vol'], False, 0, None, details=True)
    tp_cnt = sum(1 for t in trades if t['result'] == 'TP')
    sl_cnt = sum(1 for t in trades if t['result'] == 'SL')
    to_cnt = sum(1 for t in trades if t['result'] == 'TIMEOUT')
    candidates.append({
        'name': f"FVG+Vol>{b['vol']}",
        'params': f"TP{b['tp']} SL{b['sl']} INT{int(b['intv'])}",
        'trades': n, 'win_rate': wr, 'net_pnl': avg - 0.11,
        'total_pnl': total, 'monthly': total / months,
        'tp': tp_cnt, 'sl': sl_cnt, 'timeout': to_cnt
    })

cand_df = pd.DataFrame(candidates)
print(cand_df.to_string(index=False))


# 5. 최종 추천
print("\n" + "=" * 70)
print("★ 최종 추천 설정 ★")
print("=" * 70)

if len(cand_df) > 0:
    # 순수익 기준 정렬
    cand_df = cand_df.sort_values('net_pnl', ascending=False)
    best = cand_df.iloc[0]
    
    print(f"\n[1순위 - 최고 수익률]")
    print(f"  전략: {best['name']}")
    print(f"  파라미터: {best['params']}")
    print(f"  거래수: {best['trades']}회 (월 {best['trades']/months:.1f}회)")
    print(f"  승률: {best['win_rate']:.1f}%")
    print(f"  순수익/거래: {best['net_pnl']:.4f}%")
    print(f"  총수익: {best['total_pnl']:.1f}%")
    print(f"  월수익: {best['monthly']:.2f}%")
    print(f"  TP/SL/TIMEOUT: {best['tp']}/{best['sl']}/{best['timeout']}")
    
    # 안정성 기준 (거래수 많은 것)
    stable = cand_df.sort_values('trades', ascending=False).iloc[0]
    if stable['name'] != best['name']:
        print(f"\n[2순위 - 안정성]")
        print(f"  전략: {stable['name']}")
        print(f"  파라미터: {stable['params']}")
        print(f"  거래수: {stable['trades']}회 (월 {stable['trades']/months:.1f}회)")
        print(f"  승률: {stable['win_rate']:.1f}%")
        print(f"  순수익/거래: {stable['net_pnl']:.4f}%")
        print(f"  월수익: {stable['monthly']:.2f}%")

# 비교 테이블
print("\n" + "=" * 70)
print("기존 vs 나우캐스트 최적화 비교")
print("=" * 70)

print("""
| 항목          | 기존(미래참조) | 나우캐스트(기본) | 나우캐스트(최적화) |
|---------------|---------------|-----------------|-------------------|
| 승률          | 84-90%        | 62.9%           | {:.1f}%           |
| 순수익/거래   | +0.7%+        | -0.056%         | {:+.4f}%          |
| 월수익        | 14%+          | 2.1%            | {:.2f}%           |
| 실전 가능     | ❌            | ✅              | ✅                |
""".format(best['win_rate'], best['net_pnl'], best['monthly']))

# 레버리지 시뮬레이션
print("\n" + "=" * 70)
print("레버리지 시뮬레이션")
print("=" * 70)

for lev in [1, 2, 3, 5]:
    lev_monthly = best['monthly'] * lev
    lev_annual = ((1 + lev_monthly/100) ** 12 - 1) * 100
    risk_note = "" if lev == 1 else f" (청산 위험: SL {float(best['params'].split('SL')[1].split()[0])*lev:.1f}%)"
    print(f"  {lev}x 레버리지: 월 {lev_monthly:.2f}%, 연 {lev_annual:.1f}%{risk_note}")

# 저장
cand_df.to_csv('nowcast_final_candidates.csv', index=False)
print(f"\n결과 저장: nowcast_final_candidates.csv")

print("\n" + "=" * 70)
print("최적화 완료!")
print("=" * 70)
