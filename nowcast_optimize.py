"""
나우캐스트 전략 최적화
1. 그리드 서치로 TP/SL 최적 조합
2. 추가 필터 테스트 (볼륨, 모멘텀, 연속신호)
3. MTF 필터
4. 복합 필터 조합
"""

import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("나우캐스트 전략 최적화")
print("=" * 70)

# 데이터 로드
df = pd.read_csv('btc_15m_ohlcv.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna().reset_index(drop=True)

# 돌파 데이터 로드
breakouts = pd.read_csv('nowcast_breakouts.csv').to_dict('records')
print(f"데이터: {len(df)}봉, 돌파: {len(breakouts)}개")

# 추가 지표 계산
print("\n추가 지표 계산 중...")

# 1. 볼륨 이동평균
df['vol_ma20'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['vol_ma20']

# 2. RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# 3. ATR
high_low = df['high'] - df['low']
high_close = abs(df['high'] - df['close'].shift())
low_close = abs(df['low'] - df['close'].shift())
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean()
df['atr_pct'] = df['atr'] / df['close'] * 100

# 4. 이동평균
df['ma20'] = df['close'].rolling(20).mean()
df['ma50'] = df['close'].rolling(50).mean()
df['ma200'] = df['close'].rolling(200).mean()

# 5. 모멘텀
df['mom_5'] = df['close'].pct_change(5) * 100
df['mom_10'] = df['close'].pct_change(10) * 100

# 6. 볼린저 밴드
df['bb_mid'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

print("지표 계산 완료")

# 백테스트 함수
def backtest(breakouts, df, tp, sl, interval=10, filters=None):
    """필터 적용 백테스트"""
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
        
        # 필터 적용
        if filters:
            skip = False
            
            # 볼륨 필터
            if 'vol_min' in filters:
                if df.iloc[i]['vol_ratio'] < filters['vol_min']:
                    skip = True
            
            # RSI 필터
            if 'rsi_min' in filters and not skip:
                if df.iloc[i]['rsi'] < filters['rsi_min']:
                    skip = True
            if 'rsi_max' in filters and not skip:
                if df.iloc[i]['rsi'] > filters['rsi_max']:
                    skip = True
            
            # 추세 필터 (MA)
            if 'trend' in filters and not skip:
                if filters['trend'] == 'up':
                    if df.iloc[i]['close'] < df.iloc[i]['ma50']:
                        skip = True
                elif filters['trend'] == 'strong_up':
                    if df.iloc[i]['ma20'] < df.iloc[i]['ma50']:
                        skip = True
            
            # 모멘텀 필터
            if 'mom_min' in filters and not skip:
                if df.iloc[i]['mom_5'] < filters['mom_min']:
                    skip = True
            
            # ATR 필터
            if 'atr_min' in filters and not skip:
                if df.iloc[i]['atr_pct'] < filters['atr_min']:
                    skip = True
            if 'atr_max' in filters and not skip:
                if df.iloc[i]['atr_pct'] > filters['atr_max']:
                    skip = True
            
            # BB 필터
            if 'bb_max' in filters and not skip:
                if df.iloc[i]['bb_pct'] > filters['bb_max']:
                    skip = True
            
            # FVG 필터
            if 'fvg' in filters and filters['fvg'] and not skip:
                if not brk['has_fvg']:
                    skip = True
            
            if skip:
                continue
        
        entry_idx = i + 1
        if entry_idx >= len(df) - 50:
            continue
        
        entry_price = open_p[entry_idx]
        tp_level = entry_price * (1 + tp / 100)
        sl_level = entry_price * (1 - sl / 100)
        
        result = 'TIMEOUT'
        exit_price = close[min(entry_idx + 49, len(df) - 1)]
        
        for j in range(entry_idx + 1, min(entry_idx + 50, len(df))):
            if high[j] >= tp_level:
                result, exit_price = 'TP', tp_level
                break
            if low[j] <= sl_level:
                result, exit_price = 'SL', sl_level
                break
        
        pnl = (exit_price - entry_price) / entry_price * 100
        trades.append({'result': result, 'pnl': pnl})
        last_entry = i
    
    if not trades:
        return 0, 0, 0, 0
    
    pnls = [t['pnl'] for t in trades]
    win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100
    avg_pnl = np.mean(pnls)
    total_pnl = sum(pnls)
    
    return len(trades), win_rate, avg_pnl, total_pnl


# 1. TP/SL 그리드 서치
print("\n" + "=" * 70)
print("1. TP/SL 그리드 서치")
print("=" * 70)

tp_range = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
sl_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

grid_results = []
for tp, sl in product(tp_range, sl_range):
    n, wr, avg, total = backtest(breakouts, df, tp, sl)
    net = avg - 0.11
    grid_results.append({
        'tp': tp, 'sl': sl, 'trades': n, 
        'win_rate': wr, 'avg_pnl': avg, 'net_pnl': net, 'total_pnl': total
    })

grid_df = pd.DataFrame(grid_results)
grid_df = grid_df.sort_values('net_pnl', ascending=False)

print("\n상위 10개 (순수익 기준):")
print(grid_df.head(10).to_string(index=False))

best_tp = grid_df.iloc[0]['tp']
best_sl = grid_df.iloc[0]['sl']
print(f"\n최적: TP={best_tp}%, SL={best_sl}%")


# 2. 필터 테스트
print("\n" + "=" * 70)
print("2. 필터 테스트")
print("=" * 70)

filter_tests = [
    ('기본 (필터없음)', {}),
    ('볼륨 > 1.0x', {'vol_min': 1.0}),
    ('볼륨 > 1.5x', {'vol_min': 1.5}),
    ('볼륨 > 2.0x', {'vol_min': 2.0}),
    ('RSI 30-70', {'rsi_min': 30, 'rsi_max': 70}),
    ('RSI 40-60', {'rsi_min': 40, 'rsi_max': 60}),
    ('RSI < 50 (과매도)', {'rsi_max': 50}),
    ('추세: 상승 (>MA50)', {'trend': 'up'}),
    ('추세: 강상승 (MA20>MA50)', {'trend': 'strong_up'}),
    ('모멘텀 > 0%', {'mom_min': 0}),
    ('모멘텀 > 1%', {'mom_min': 1}),
    ('ATR 0.5-2%', {'atr_min': 0.5, 'atr_max': 2.0}),
    ('ATR 1-3%', {'atr_min': 1.0, 'atr_max': 3.0}),
    ('BB < 0.5 (하단)', {'bb_max': 0.5}),
    ('BB < 0.3 (바닥)', {'bb_max': 0.3}),
    ('FVG 필수', {'fvg': True}),
]

filter_results = []
for name, filters in filter_tests:
    n, wr, avg, total = backtest(breakouts, df, best_tp, best_sl, filters=filters)
    net = avg - 0.11 if n > 0 else 0
    filter_results.append({
        'filter': name, 'trades': n,
        'win_rate': wr, 'avg_pnl': avg, 'net_pnl': net
    })
    print(f"  {name:25s}: 거래={n:4d}, 승률={wr:5.1f}%, 순수익={net:+.3f}%")

filter_df = pd.DataFrame(filter_results)


# 3. 복합 필터 조합
print("\n" + "=" * 70)
print("3. 복합 필터 조합")
print("=" * 70)

combo_tests = [
    ('볼륨+추세', {'vol_min': 1.0, 'trend': 'up'}),
    ('볼륨+RSI', {'vol_min': 1.0, 'rsi_max': 50}),
    ('볼륨+모멘텀', {'vol_min': 1.0, 'mom_min': 0}),
    ('볼륨+ATR', {'vol_min': 1.0, 'atr_min': 0.5, 'atr_max': 2.0}),
    ('볼륨+BB', {'vol_min': 1.0, 'bb_max': 0.5}),
    ('추세+RSI', {'trend': 'up', 'rsi_max': 50}),
    ('추세+모멘텀', {'trend': 'up', 'mom_min': 0}),
    ('FVG+볼륨', {'fvg': True, 'vol_min': 1.0}),
    ('FVG+추세', {'fvg': True, 'trend': 'up'}),
    ('볼륨+추세+RSI', {'vol_min': 1.0, 'trend': 'up', 'rsi_max': 50}),
    ('볼륨+추세+모멘텀', {'vol_min': 1.0, 'trend': 'up', 'mom_min': 0}),
    ('볼륨+BB+RSI', {'vol_min': 1.0, 'bb_max': 0.5, 'rsi_max': 50}),
    ('FVG+볼륨+추세', {'fvg': True, 'vol_min': 1.0, 'trend': 'up'}),
    ('올인원', {'vol_min': 1.2, 'trend': 'up', 'rsi_max': 55, 'mom_min': -1}),
]

combo_results = []
for name, filters in combo_tests:
    n, wr, avg, total = backtest(breakouts, df, best_tp, best_sl, filters=filters)
    net = avg - 0.11 if n > 0 else 0
    combo_results.append({
        'combo': name, 'trades': n,
        'win_rate': wr, 'avg_pnl': avg, 'net_pnl': net, 'total_pnl': total
    })
    print(f"  {name:20s}: 거래={n:4d}, 승률={wr:5.1f}%, 순수익={net:+.3f}%")

combo_df = pd.DataFrame(combo_results)
combo_df = combo_df.sort_values('net_pnl', ascending=False)

print("\n복합 필터 상위 5개:")
print(combo_df.head(5).to_string(index=False))


# 4. 최적 필터로 TP/SL 재최적화
print("\n" + "=" * 70)
print("4. 최적 필터로 TP/SL 재최적화")
print("=" * 70)

# 상위 필터 선정
best_filters = [
    ('볼륨+추세', {'vol_min': 1.0, 'trend': 'up'}),
    ('볼륨+BB', {'vol_min': 1.0, 'bb_max': 0.5}),
    ('FVG+볼륨+추세', {'fvg': True, 'vol_min': 1.0, 'trend': 'up'}),
]

final_results = []
for fname, filters in best_filters:
    print(f"\n{fname} 필터:")
    for tp, sl in product([0.5, 0.7, 1.0, 1.5, 2.0], [1.5, 2.0, 3.0, 4.0, 5.0]):
        n, wr, avg, total = backtest(breakouts, df, tp, sl, filters=filters)
        if n >= 100:  # 최소 거래수
            net = avg - 0.11
            final_results.append({
                'filter': fname, 'tp': tp, 'sl': sl, 'trades': n,
                'win_rate': wr, 'avg_pnl': avg, 'net_pnl': net, 'total_pnl': total
            })

final_df = pd.DataFrame(final_results)
final_df = final_df.sort_values('net_pnl', ascending=False)

print("\n최종 상위 10개:")
print(final_df.head(10).to_string(index=False))


# 5. 최종 최적 설정
print("\n" + "=" * 70)
print("5. 최종 최적 설정")
print("=" * 70)

if len(final_df) > 0:
    best = final_df.iloc[0]
    
    # 해당 필터 찾기
    best_filter_name = best['filter']
    best_filter = None
    for fname, f in best_filters:
        if fname == best_filter_name:
            best_filter = f
            break
    
    total_days = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days
    months = total_days / 30
    
    print(f"\n최적 파라미터:")
    print(f"  필터: {best['filter']}")
    print(f"  TP: {best['tp']}%")
    print(f"  SL: {best['sl']}%")
    print(f"\n성과:")
    print(f"  총 거래: {best['trades']}회")
    print(f"  승률: {best['win_rate']:.1f}%")
    print(f"  평균 수익: {best['avg_pnl']:.3f}%")
    print(f"  수수료 후: {best['net_pnl']:.3f}%")
    print(f"  총 수익: {best['total_pnl']:.1f}%")
    print(f"\n월간 추정:")
    print(f"  월 거래: {best['trades']/months:.1f}회")
    print(f"  월 수익: {best['total_pnl']/months:.1f}%")
    
    # 기존 대비
    print(f"\n기존 대비 개선:")
    print(f"  기존 (필터없음): 승률 62.9%, 순수익 -0.056%")
    print(f"  최적 (필터적용): 승률 {best['win_rate']:.1f}%, 순수익 {best['net_pnl']:.3f}%")

# 저장
final_df.to_csv('nowcast_optimization_final.csv', index=False)
print(f"\n결과 저장: nowcast_optimization_final.csv")

print("\n" + "=" * 70)
print("최적화 완료!")
print("=" * 70)
