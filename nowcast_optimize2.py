"""
나우캐스트 전략 심층 최적화
- FVG+볼륨 조합이 유망 (+0.042% 순수익)
- 더 세밀한 파라미터 탐색
"""

import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("나우캐스트 전략 심층 최적화")
print("=" * 70)

# 데이터 로드
df = pd.read_csv('btc_15m_ohlcv.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna().reset_index(drop=True)

breakouts = pd.read_csv('nowcast_breakouts.csv').to_dict('records')

# 지표 계산
df['vol_ma20'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['vol_ma20']

delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + gain / loss))

df['ma20'] = df['close'].rolling(20).mean()
df['ma50'] = df['close'].rolling(50).mean()
df['mom_5'] = df['close'].pct_change(5) * 100

high_low = df['high'] - df['low']
high_close = abs(df['high'] - df['close'].shift())
low_close = abs(df['low'] - df['close'].shift())
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean()
df['atr_pct'] = df['atr'] / df['close'] * 100

print(f"데이터: {len(df)}봉, 돌파: {len(breakouts)}개")


def backtest(breakouts, df, tp, sl, interval=10, filters=None, max_hold=50):
    open_p = df['open'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    trades = []
    last_entry = -interval - 1
    
    for brk in breakouts:
        i = brk['idx']
        
        if i - last_entry < interval:
            continue
        if brk['type'] != 'long':
            continue
        
        # 필터
        if filters:
            skip = False
            if 'fvg' in filters and filters['fvg'] and not brk['has_fvg']:
                skip = True
            if 'vol_min' in filters and df.iloc[i]['vol_ratio'] < filters['vol_min']:
                skip = True
            if 'vol_max' in filters and df.iloc[i]['vol_ratio'] > filters['vol_max']:
                skip = True
            if 'trend' in filters and filters['trend'] and df.iloc[i]['close'] < df.iloc[i]['ma50']:
                skip = True
            if 'rsi_min' in filters and df.iloc[i]['rsi'] < filters['rsi_min']:
                skip = True
            if 'rsi_max' in filters and df.iloc[i]['rsi'] > filters['rsi_max']:
                skip = True
            if 'mom_min' in filters and df.iloc[i]['mom_5'] < filters['mom_min']:
                skip = True
            if 'atr_min' in filters and df.iloc[i]['atr_pct'] < filters['atr_min']:
                skip = True
            if 'atr_max' in filters and df.iloc[i]['atr_pct'] > filters['atr_max']:
                skip = True
            if skip:
                continue
        
        entry_idx = i + 1
        if entry_idx >= len(df) - max_hold:
            continue
        
        entry_price = open_p[entry_idx]
        tp_level = entry_price * (1 + tp / 100)
        sl_level = entry_price * (1 - sl / 100)
        
        result = 'TIMEOUT'
        exit_price = close[min(entry_idx + max_hold - 1, len(df) - 1)]
        
        for j in range(entry_idx + 1, min(entry_idx + max_hold, len(df))):
            if high[j] >= tp_level:
                result, exit_price = 'TP', tp_level
                break
            if low[j] <= sl_level:
                result, exit_price = 'SL', sl_level
                break
        
        pnl = (exit_price - entry_price) / entry_price * 100
        trades.append({'result': result, 'pnl': pnl, 'idx': i})
        last_entry = i
    
    if not trades:
        return 0, 0, 0, 0, []
    
    pnls = [t['pnl'] for t in trades]
    win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100
    avg_pnl = np.mean(pnls)
    total_pnl = sum(pnls)
    
    return len(trades), win_rate, avg_pnl, total_pnl, trades


# 1. FVG+볼륨 기반 세밀 최적화
print("\n" + "=" * 70)
print("1. FVG+볼륨 세밀 최적화")
print("=" * 70)

tp_range = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
sl_range = [0.5, 0.7, 1.0, 1.5, 2.0]
vol_range = [0.8, 1.0, 1.2, 1.5]
interval_range = [5, 10, 15, 20]

results = []
base_filter = {'fvg': True}

for tp, sl, vol, intv in product(tp_range, sl_range, vol_range, interval_range):
    filters = {**base_filter, 'vol_min': vol}
    n, wr, avg, total, _ = backtest(breakouts, df, tp, sl, interval=intv, filters=filters)
    if n >= 50:
        net = avg - 0.11
        results.append({
            'tp': tp, 'sl': sl, 'vol': vol, 'interval': intv,
            'trades': n, 'win_rate': wr, 'avg_pnl': avg, 'net_pnl': net, 'total_pnl': total
        })

results_df = pd.DataFrame(results).sort_values('net_pnl', ascending=False)
print("\nFVG+볼륨 상위 15개:")
print(results_df.head(15).to_string(index=False))


# 2. 추가 필터 테스트 (FVG+볼륨 기반)
print("\n" + "=" * 70)
print("2. 추가 필터 조합 (FVG+볼륨 기반)")
print("=" * 70)

# 상위 설정 사용
if len(results_df) > 0:
    best = results_df.iloc[0]
    base_tp, base_sl, base_vol, base_intv = best['tp'], best['sl'], best['vol'], best['interval']
else:
    base_tp, base_sl, base_vol, base_intv = 2.5, 1.0, 1.0, 10

print(f"기준: TP={base_tp}%, SL={base_sl}%, Vol>{base_vol}x, Interval={base_intv}")

filter_combos = [
    ('FVG+볼륨', {'fvg': True, 'vol_min': base_vol}),
    ('FVG+볼륨+추세', {'fvg': True, 'vol_min': base_vol, 'trend': True}),
    ('FVG+볼륨+RSI<50', {'fvg': True, 'vol_min': base_vol, 'rsi_max': 50}),
    ('FVG+볼륨+RSI<45', {'fvg': True, 'vol_min': base_vol, 'rsi_max': 45}),
    ('FVG+볼륨+RSI30-50', {'fvg': True, 'vol_min': base_vol, 'rsi_min': 30, 'rsi_max': 50}),
    ('FVG+볼륨+모멘텀>0', {'fvg': True, 'vol_min': base_vol, 'mom_min': 0}),
    ('FVG+볼륨+모멘텀>-1', {'fvg': True, 'vol_min': base_vol, 'mom_min': -1}),
    ('FVG+볼륨+ATR<2', {'fvg': True, 'vol_min': base_vol, 'atr_max': 2.0}),
    ('FVG+볼륨+ATR<1.5', {'fvg': True, 'vol_min': base_vol, 'atr_max': 1.5}),
    ('FVG+볼륨+ATR0.3-1.5', {'fvg': True, 'vol_min': base_vol, 'atr_min': 0.3, 'atr_max': 1.5}),
    ('FVG+볼륨High', {'fvg': True, 'vol_min': 1.5}),
    ('FVG+볼륨VHigh', {'fvg': True, 'vol_min': 2.0}),
    ('FVG+추세+RSI<50', {'fvg': True, 'trend': True, 'rsi_max': 50}),
    ('FVG+모멘텀>0', {'fvg': True, 'mom_min': 0}),
    ('FVG+ATR<1.5', {'fvg': True, 'atr_max': 1.5}),
]

filter_results = []
for name, filters in filter_combos:
    n, wr, avg, total, _ = backtest(breakouts, df, base_tp, base_sl, interval=base_intv, filters=filters)
    if n > 0:
        net = avg - 0.11
        filter_results.append({
            'name': name, 'trades': n, 'win_rate': wr, 
            'avg_pnl': avg, 'net_pnl': net, 'total_pnl': total
        })
        status = "✓" if net > 0 else " "
        print(f"  {status} {name:20s}: 거래={n:4d}, 승률={wr:5.1f}%, 순수익={net:+.4f}%")

filter_df = pd.DataFrame(filter_results).sort_values('net_pnl', ascending=False)


# 3. 최적 필터로 전체 파라미터 재탐색
print("\n" + "=" * 70)
print("3. 최적 필터 전체 파라미터 탐색")
print("=" * 70)

# 양수 순수익 필터만
positive_filters = filter_df[filter_df['net_pnl'] > 0]
if len(positive_filters) > 0:
    print(f"\n양수 순수익 필터: {len(positive_filters)}개")
    
    for _, row in positive_filters.head(3).iterrows():
        fname = row['name']
        # 필터 찾기
        for n, f in filter_combos:
            if n == fname:
                print(f"\n{fname} 상세 탐색:")
                
                sub_results = []
                for tp in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
                    for sl in [0.5, 0.7, 1.0, 1.5, 2.0]:
                        for intv in [5, 10, 15]:
                            nn, wr, avg, total, _ = backtest(breakouts, df, tp, sl, interval=intv, filters=f)
                            if nn >= 30:
                                net = avg - 0.11
                                sub_results.append({
                                    'tp': tp, 'sl': sl, 'intv': intv,
                                    'trades': nn, 'win_rate': wr, 'net_pnl': net, 'total_pnl': total
                                })
                
                sub_df = pd.DataFrame(sub_results).sort_values('net_pnl', ascending=False)
                print(sub_df.head(10).to_string(index=False))
                break
else:
    print("양수 순수익 필터 없음. 기본 FVG+볼륨으로 진행")


# 4. 홀딩 기간 최적화
print("\n" + "=" * 70)
print("4. 홀딩 기간 최적화")
print("=" * 70)

hold_results = []
best_filter = {'fvg': True, 'vol_min': 1.0}

for hold in [20, 30, 40, 50, 60, 80, 100]:
    for tp in [2.0, 2.5, 3.0, 4.0]:
        for sl in [0.7, 1.0, 1.5]:
            n, wr, avg, total, _ = backtest(breakouts, df, tp, sl, interval=10, 
                                            filters=best_filter, max_hold=hold)
            if n >= 50:
                net = avg - 0.11
                hold_results.append({
                    'hold': hold, 'tp': tp, 'sl': sl,
                    'trades': n, 'win_rate': wr, 'net_pnl': net, 'total_pnl': total
                })

hold_df = pd.DataFrame(hold_results).sort_values('net_pnl', ascending=False)
print("\n홀딩 기간별 상위 10개:")
print(hold_df.head(10).to_string(index=False))


# 5. 최종 최적 설정
print("\n" + "=" * 70)
print("5. 최종 최적 설정")
print("=" * 70)

# 모든 결과 합치기
all_results = []

# results_df에서
if len(results_df) > 0:
    for _, r in results_df.head(20).iterrows():
        all_results.append({
            'config': f"FVG+Vol>{r['vol']} TP{r['tp']} SL{r['sl']} INT{r['interval']}",
            'trades': r['trades'], 'win_rate': r['win_rate'],
            'net_pnl': r['net_pnl'], 'total_pnl': r['total_pnl']
        })

# hold_df에서
if len(hold_df) > 0:
    for _, r in hold_df.head(10).iterrows():
        all_results.append({
            'config': f"FVG+Vol>1.0 TP{r['tp']} SL{r['sl']} H{r['hold']}",
            'trades': r['trades'], 'win_rate': r['win_rate'],
            'net_pnl': r['net_pnl'], 'total_pnl': r['total_pnl']
        })

all_df = pd.DataFrame(all_results).sort_values('net_pnl', ascending=False)

print("\n전체 상위 10개:")
print(all_df.head(10).to_string(index=False))

# 최종 선정
if len(all_df) > 0 and all_df.iloc[0]['net_pnl'] > 0:
    best = all_df.iloc[0]
    total_days = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days
    months = total_days / 30
    
    print(f"\n{'='*50}")
    print("★ 최종 최적 설정 ★")
    print(f"{'='*50}")
    print(f"  설정: {best['config']}")
    print(f"  거래: {best['trades']}회 ({months:.0f}개월)")
    print(f"  승률: {best['win_rate']:.1f}%")
    print(f"  순수익/거래: {best['net_pnl']:.4f}%")
    print(f"  총수익: {best['total_pnl']:.1f}%")
    print(f"  월수익: {best['total_pnl']/months:.2f}%")
    print(f"  월거래: {best['trades']/months:.1f}회")
    
    # 연 복리 계산
    monthly_return = best['total_pnl'] / months / 100
    annual_compound = ((1 + monthly_return) ** 12 - 1) * 100
    print(f"  연 복리: {annual_compound:.1f}%")
else:
    print("\n⚠️ 수수료 후 양수 순수익 설정 없음")
    print("전략 재검토 필요")

# 저장
all_df.to_csv('nowcast_final_optimization.csv', index=False)
print(f"\n결과 저장: nowcast_final_optimization.csv")

print("\n" + "=" * 70)
print("최적화 완료!")
print("=" * 70)
