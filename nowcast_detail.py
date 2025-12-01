"""
나우캐스트 상세 분석
- 매매 횟수 분석
- 승률 개선 방안
- MDD 계산
- 진입 후 가격 이동 분석
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("나우캐스트 상세 분석")
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

total_days = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days
months = total_days / 30
years = total_days / 365

print(f"데이터: {len(df)}봉")
print(f"기간: {df['datetime'].iloc[0].date()} ~ {df['datetime'].iloc[-1].date()}")
print(f"      ({years:.1f}년 / {months:.0f}개월 / {total_days}일)")


def detailed_backtest(tp, sl, interval, fvg_only, vol_min):
    """상세 백테스트 - 진입 후 이동 추적"""
    open_p = df['open'].values
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    vol_ratio = df['vol_ratio'].values
    datetimes = df['datetime'].values
    
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
        
        entry_idx = i + 1
        if entry_idx >= len(df) - 100:
            continue
        
        entry_price = open_p[entry_idx]
        entry_time = datetimes[entry_idx]
        tp_level = entry_price * (1 + tp / 100)
        sl_level = entry_price * (1 - sl / 100)
        
        # 진입 후 이동 추적
        max_profit = 0
        max_drawdown = 0
        bar_moves = []
        
        result = 'TIMEOUT'
        exit_price = close[entry_idx + 99]
        exit_idx = entry_idx + 99
        
        for j in range(entry_idx + 1, min(entry_idx + 100, len(df))):
            # 현재 바에서의 수익률
            curr_high_pct = (high[j] - entry_price) / entry_price * 100
            curr_low_pct = (low[j] - entry_price) / entry_price * 100
            curr_close_pct = (close[j] - entry_price) / entry_price * 100
            
            max_profit = max(max_profit, curr_high_pct)
            max_drawdown = min(max_drawdown, curr_low_pct)
            
            bar_moves.append({
                'bar': j - entry_idx,
                'high_pct': curr_high_pct,
                'low_pct': curr_low_pct,
                'close_pct': curr_close_pct
            })
            
            # TP/SL 체크
            if high[j] >= tp_level:
                result, exit_price, exit_idx = 'TP', tp_level, j
                break
            if low[j] <= sl_level:
                result, exit_price, exit_idx = 'SL', sl_level, j
                break
        
        pnl = (exit_price - entry_price) / entry_price * 100
        hold_bars = exit_idx - entry_idx
        
        trades.append({
            'entry_idx': entry_idx,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'result': result,
            'pnl': pnl,
            'hold_bars': hold_bars,
            'max_profit': max_profit,
            'max_drawdown': max_drawdown,
            'bar_moves': bar_moves
        })
        last_entry = i
    
    return trades


# 1. 안정형 전략 (FVG + Vol > 2.0)
print("\n" + "=" * 70)
print("1. 안정형 전략 상세 분석 (FVG + Vol > 2.0)")
print("=" * 70)

trades1 = detailed_backtest(tp=5.5, sl=0.7, interval=12, fvg_only=True, vol_min=2.0)

print(f"\n[매매 횟수]")
print(f"  총 거래: {len(trades1)}회")
print(f"  월평균: {len(trades1)/months:.1f}회")
print(f"  일평균: {len(trades1)/total_days:.2f}회")

# 연도별 분석
trades_df1 = pd.DataFrame(trades1)
trades_df1['year'] = pd.to_datetime(trades_df1['entry_time']).dt.year
yearly = trades_df1.groupby('year').agg({
    'pnl': ['count', 'sum', 'mean'],
    'result': lambda x: (x == 'TP').sum()
}).round(3)
yearly.columns = ['거래수', '총수익', '평균수익', 'TP횟수']
print(f"\n[연도별 매매]")
print(yearly.to_string())

# 승률 분석
print(f"\n[승률 분석]")
results = [t['result'] for t in trades1]
pnls = [t['pnl'] for t in trades1]
tp_cnt = results.count('TP')
sl_cnt = results.count('SL')
to_cnt = results.count('TIMEOUT')
win_cnt = sum(1 for p in pnls if p > 0)

print(f"  TP: {tp_cnt}회 ({tp_cnt/len(trades1)*100:.1f}%)")
print(f"  SL: {sl_cnt}회 ({sl_cnt/len(trades1)*100:.1f}%)")
print(f"  TIMEOUT: {to_cnt}회 ({to_cnt/len(trades1)*100:.1f}%)")
print(f"  실제 승률: {win_cnt/len(trades1)*100:.1f}% (PnL > 0)")

# MDD 계산
print(f"\n[MDD (Maximum Drawdown)]")
cumsum = np.cumsum(pnls)
running_max = np.maximum.accumulate(cumsum)
drawdown = cumsum - running_max
mdd = drawdown.min()
mdd_idx = np.argmin(drawdown)

print(f"  MDD: {mdd:.2f}%")
print(f"  MDD 발생 시점: 거래 #{mdd_idx+1}")
if mdd_idx < len(trades1):
    print(f"  MDD 발생 일자: {trades1[mdd_idx]['entry_time']}")

# 누적 수익 곡선 주요 지점
print(f"\n  누적 수익 곡선:")
checkpoints = [0, len(cumsum)//4, len(cumsum)//2, 3*len(cumsum)//4, len(cumsum)-1]
for cp in checkpoints:
    print(f"    거래 #{cp+1}: 누적 {cumsum[cp]:.2f}%")

# 진입 후 이동 분석
print(f"\n[진입 후 가격 이동 분석]")

# 평균 이동
all_moves = []
for t in trades1:
    for m in t['bar_moves'][:50]:  # 최대 50봉까지
        all_moves.append(m)

moves_df = pd.DataFrame(all_moves)
bar_stats = moves_df.groupby('bar').agg({
    'high_pct': 'mean',
    'low_pct': 'mean',
    'close_pct': 'mean'
}).round(3)

print(f"\n  진입 후 평균 이동 (바별):")
print(f"  {'바':>4} {'고가':>8} {'저가':>8} {'종가':>8}")
for bar in [1, 2, 3, 5, 10, 20, 30, 50]:
    if bar in bar_stats.index:
        row = bar_stats.loc[bar]
        print(f"  {bar:4d} {row['high_pct']:+8.3f}% {row['low_pct']:+8.3f}% {row['close_pct']:+8.3f}%")

# 최대 유리/불리 분석
max_profits = [t['max_profit'] for t in trades1]
max_drawdowns = [t['max_drawdown'] for t in trades1]

print(f"\n  진입 후 최대 유리/불리:")
print(f"    평균 최대 유리: +{np.mean(max_profits):.2f}%")
print(f"    평균 최대 불리: {np.mean(max_drawdowns):.2f}%")
print(f"    최대 유리 (MAX): +{max(max_profits):.2f}%")
print(f"    최대 불리 (MIN): {min(max_drawdowns):.2f}%")

# TP 도달 전 최대 역행
tp_trades = [t for t in trades1 if t['result'] == 'TP']
sl_trades = [t for t in trades1 if t['result'] == 'SL']

if tp_trades:
    tp_max_adverse = [t['max_drawdown'] for t in tp_trades]
    print(f"\n  TP 거래의 최대 역행:")
    print(f"    평균: {np.mean(tp_max_adverse):.3f}%")
    print(f"    최대: {min(tp_max_adverse):.3f}%")

if sl_trades:
    sl_max_favor = [t['max_profit'] for t in sl_trades]
    print(f"\n  SL 거래의 최대 유리 (놓친 이익):")
    print(f"    평균: +{np.mean(sl_max_favor):.3f}%")
    print(f"    최대: +{max(sl_max_favor):.3f}%")

# 홀딩 기간 분석
hold_bars = [t['hold_bars'] for t in trades1]
print(f"\n[홀딩 기간]")
print(f"  평균: {np.mean(hold_bars):.1f}봉 ({np.mean(hold_bars)*15/60:.1f}시간)")
print(f"  TP 평균: {np.mean([t['hold_bars'] for t in tp_trades]):.1f}봉" if tp_trades else "  TP 없음")
print(f"  SL 평균: {np.mean([t['hold_bars'] for t in sl_trades]):.1f}봉" if sl_trades else "  SL 없음")


# 2. 승률 개선 방안 탐색
print("\n" + "=" * 70)
print("2. 승률 개선 방안")
print("=" * 70)

# 현재 승률이 낮은 이유 분석
print("\n[현재 문제점]")
print(f"  • 승률 33.6%로 낮음")
print(f"  • SL 0.7%가 너무 타이트 → 조기 손절")
print(f"  • TP 5.5%가 너무 높음 → 도달 어려움")

# SL 확대 테스트
print("\n[SL 확대 테스트]")
for sl in [0.7, 1.0, 1.5, 2.0, 2.5, 3.0]:
    trades = detailed_backtest(tp=5.5, sl=sl, interval=12, fvg_only=True, vol_min=2.0)
    if trades:
        pnls = [t['pnl'] for t in trades]
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        avg = np.mean(pnls) - 0.11
        print(f"  SL {sl}%: 승률 {wr:.1f}%, 순수익 {avg:+.3f}%")

# TP 조정 테스트
print("\n[TP 조정 테스트 (SL 0.7% 고정)]")
for tp in [2.0, 3.0, 4.0, 5.0, 5.5, 6.0]:
    trades = detailed_backtest(tp=tp, sl=0.7, interval=12, fvg_only=True, vol_min=2.0)
    if trades:
        pnls = [t['pnl'] for t in trades]
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        avg = np.mean(pnls) - 0.11
        print(f"  TP {tp}%: 승률 {wr:.1f}%, 순수익 {avg:+.3f}%")

# 균형잡힌 설정
print("\n[균형잡힌 TP/SL 조합]")
best_balanced = []
for tp in [2.0, 2.5, 3.0, 3.5, 4.0]:
    for sl in [1.0, 1.5, 2.0]:
        trades = detailed_backtest(tp=tp, sl=sl, interval=12, fvg_only=True, vol_min=2.0)
        if trades:
            pnls = [t['pnl'] for t in trades]
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            avg = np.mean(pnls) - 0.11
            total = sum(pnls)
            best_balanced.append({
                'tp': tp, 'sl': sl, 'n': len(trades),
                'wr': wr, 'net': avg, 'total': total
            })

bal_df = pd.DataFrame(best_balanced).sort_values('net', ascending=False)
print(bal_df.head(10).to_string(index=False))


# 3. 최적 균형 설정 상세 분석
print("\n" + "=" * 70)
print("3. 최적 균형 설정 상세")
print("=" * 70)

if len(bal_df) > 0:
    best = bal_df.iloc[0]
    trades2 = detailed_backtest(tp=best['tp'], sl=best['sl'], interval=12, fvg_only=True, vol_min=2.0)
    
    print(f"\n설정: TP {best['tp']}%, SL {best['sl']}%")
    print(f"거래수: {len(trades2)}회")
    print(f"승률: {best['wr']:.1f}%")
    print(f"순수익: {best['net']:.4f}%/거래")
    
    # MDD
    pnls2 = [t['pnl'] for t in trades2]
    cumsum2 = np.cumsum(pnls2)
    running_max2 = np.maximum.accumulate(cumsum2)
    drawdown2 = cumsum2 - running_max2
    mdd2 = drawdown2.min()
    
    print(f"MDD: {mdd2:.2f}%")
    print(f"총수익: {sum(pnls2):.1f}%")
    print(f"월수익: {sum(pnls2)/months:.2f}%")


# 4. 진입 타이밍 개선 (추가 필터)
print("\n" + "=" * 70)
print("4. 진입 타이밍 개선 (추가 필터)")
print("=" * 70)

# RSI, 모멘텀 등 추가
df['rsi'] = 100 - (100 / (1 + df['close'].diff().where(lambda x: x > 0, 0).rolling(14).mean() / 
                          (-df['close'].diff().where(lambda x: x < 0, 0)).rolling(14).mean()))
df['mom_5'] = df['close'].pct_change(5) * 100
df['ma20'] = df['close'].rolling(20).mean()
df['ma50'] = df['close'].rolling(50).mean()

# 추세 방향 확인
def backtest_with_filters(tp, sl, interval, fvg_only, vol_min, trend=False, rsi_max=0, mom_min=None):
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
        
        for j in range(entry_idx + 1, min(entry_idx + 50, len(df))):
            if high[j] >= tp_level:
                exit_price = tp_level
                result = 'TP'
                break
            if low[j] <= sl_level:
                exit_price = sl_level
                result = 'SL'
                break
        
        pnl = (exit_price - entry_price) / entry_price * 100
        trades.append({'pnl': pnl, 'result': result})
        last_entry = i
    
    if not trades:
        return 0, 0, 0
    
    pnls = [t['pnl'] for t in trades]
    wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
    net = np.mean(pnls) - 0.11
    return len(trades), wr, net


print("\n[추가 필터 효과 (TP 3.0%, SL 1.5%)]")
configs = [
    ('기본 (FVG+Vol>2)', False, 0, None),
    ('+ 추세 (>MA50)', True, 0, None),
    ('+ RSI < 50', False, 50, None),
    ('+ RSI < 45', False, 45, None),
    ('+ 모멘텀 > 0', False, 0, 0),
    ('+ 모멘텀 > -1', False, 0, -1),
    ('+ 추세 + RSI<50', True, 50, None),
    ('+ 추세 + Mom>0', True, 0, 0),
]

for name, trend, rsi, mom in configs:
    n, wr, net = backtest_with_filters(3.0, 1.5, 12, True, 2.0, trend, rsi, mom)
    if n > 0:
        status = "✓" if net > 0 else " "
        print(f"  {status} {name:25s}: 거래={n:3d}, 승률={wr:5.1f}%, 순수익={net:+.4f}%")


# 5. 최종 추천
print("\n" + "=" * 70)
print("★ 최종 추천 설정 ★")
print("=" * 70)

print("""
[고수익형] - 낮은 승률, 높은 RR
  필터: FVG + Vol > 2.0
  TP: 5.5%, SL: 0.7%
  승률: 33.6%, 순수익: +0.138%/거래
  MDD: {:.1f}%
  특징: 큰 추세를 잡으려고 시도, 손절 빠름

[균형형] - 적정 승률, 적정 수익
  필터: FVG + Vol > 2.0
  TP: 3.0%, SL: 1.5%
  더 높은 승률, 안정적인 거래
  
[대응 전략]
  1. 진입 후 +2% 도달 시 → 손절을 본절로 이동
  2. 진입 후 +3% 도달 시 → 50% 익절, 나머지 트레일링
  3. 진입 후 30봉 경과 시 → 타임아웃 고려
""".format(mdd))

# 결과 저장
trades_df1.to_csv('nowcast_trades_detail.csv', index=False)
print(f"\n거래 내역 저장: nowcast_trades_detail.csv")

print("\n" + "=" * 70)
print("분석 완료!")
print("=" * 70)
