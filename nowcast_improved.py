"""
나우캐스트 개선 전략
- 트레일링 스탑
- 분할 익절
- 본절 이동
- 시간 기반 청산
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("나우캐스트 개선 전략")
print("=" * 70)

# 데이터
df = pd.read_csv('btc_15m_ohlcv.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna().reset_index(drop=True)

breakouts = pd.read_csv('nowcast_breakouts.csv').to_dict('records')

df['vol_ma20'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['vol_ma20']

total_days = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days
months = total_days / 30

print(f"데이터: {len(df)}봉 ({months:.0f}개월)")


def advanced_backtest(tp, sl, interval, fvg_only, vol_min, 
                      trailing=False, trail_start=0, trail_pct=0,
                      partial=False, partial_at=0, partial_pct=50,
                      breakeven=False, be_at=0,
                      time_exit=0):
    """
    고급 백테스트
    - trailing: 트레일링 스탑 사용
    - trail_start: 트레일링 시작 수익% (예: 2% 수익 후 시작)
    - trail_pct: 트레일링 폭% (예: 고점 대비 1% 하락 시 청산)
    - partial: 분할 익절 사용
    - partial_at: 분할 익절 시점%
    - partial_pct: 분할 익절 비율%
    - breakeven: 본절 이동
    - be_at: 본절 이동 시점%
    - time_exit: 시간 청산 (봉 수, 0이면 미사용)
    """
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
        
        # 상태 변수
        position_size = 1.0  # 100%
        realized_pnl = 0
        max_high = entry_price
        trailing_active = False
        breakeven_active = False
        
        result = 'TIMEOUT'
        exit_price = None
        exit_idx = None
        max_bars = time_exit if time_exit > 0 else 100
        
        for j in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
            curr_high = high[j]
            curr_low = low[j]
            curr_close = close[j]
            
            curr_high_pct = (curr_high - entry_price) / entry_price * 100
            curr_low_pct = (curr_low - entry_price) / entry_price * 100
            
            # 최고점 업데이트
            if curr_high > max_high:
                max_high = curr_high
            
            # 본절 이동 체크
            if breakeven and not breakeven_active and curr_high_pct >= be_at:
                sl_level = entry_price * 1.001  # 약간의 이익 확보
                breakeven_active = True
            
            # 트레일링 스탑 활성화 체크
            if trailing and not trailing_active and curr_high_pct >= trail_start:
                trailing_active = True
            
            # 트레일링 스탑 레벨 업데이트
            if trailing_active:
                trail_sl = max_high * (1 - trail_pct / 100)
                if trail_sl > sl_level:
                    sl_level = trail_sl
            
            # 분할 익절 체크
            if partial and position_size > 0.5 and curr_high_pct >= partial_at:
                partial_exit = entry_price * (1 + partial_at / 100)
                realized_pnl += partial_at * (partial_pct / 100)
                position_size -= partial_pct / 100
            
            # TP 체크
            if curr_high >= tp_level and position_size > 0:
                result = 'TP'
                exit_price = tp_level
                exit_idx = j
                realized_pnl += tp * position_size
                break
            
            # SL 체크
            if curr_low <= sl_level:
                sl_pct = (sl_level - entry_price) / entry_price * 100
                result = 'SL' if sl_pct < 0 else 'BE'  # 본절 또는 손절
                exit_price = sl_level
                exit_idx = j
                realized_pnl += sl_pct * position_size
                break
        
        # 타임아웃
        if exit_price is None:
            exit_idx = min(entry_idx + max_bars - 1, len(df) - 1)
            exit_price = close[exit_idx]
            exit_pct = (exit_price - entry_price) / entry_price * 100
            realized_pnl += exit_pct * position_size
        
        trades.append({
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'result': result,
            'pnl': realized_pnl,
            'hold_bars': exit_idx - entry_idx if exit_idx else 0
        })
        last_entry = i
    
    return trades


def calc_stats(trades, fee=0.11):
    if not trades:
        return None
    
    pnls = [t['pnl'] for t in trades]
    results = [t['result'] for t in trades]
    
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = cumsum - running_max
    mdd = drawdown.min()
    
    win_cnt = sum(1 for p in pnls if p > 0)
    
    return {
        'n': len(trades),
        'win_rate': win_cnt / len(trades) * 100,
        'avg_pnl': np.mean(pnls),
        'net_pnl': np.mean(pnls) - fee,
        'total_pnl': sum(pnls),
        'mdd': mdd,
        'tp': results.count('TP'),
        'sl': results.count('SL'),
        'be': results.count('BE'),
        'timeout': results.count('TIMEOUT')
    }


# 1. 기본 전략 (비교 기준)
print("\n" + "=" * 70)
print("1. 기본 전략 (비교 기준)")
print("=" * 70)

base_trades = advanced_backtest(
    tp=5.5, sl=0.7, interval=12, fvg_only=True, vol_min=2.0
)
base_stats = calc_stats(base_trades)

print(f"  거래: {base_stats['n']}회")
print(f"  승률: {base_stats['win_rate']:.1f}%")
print(f"  순수익: {base_stats['net_pnl']:.4f}%/거래")
print(f"  총수익: {base_stats['total_pnl']:.1f}%")
print(f"  MDD: {base_stats['mdd']:.1f}%")
print(f"  TP/SL/TO: {base_stats['tp']}/{base_stats['sl']}/{base_stats['timeout']}")


# 2. 트레일링 스탑 전략
print("\n" + "=" * 70)
print("2. 트레일링 스탑 전략")
print("=" * 70)

print("\n[트레일링 시작점 테스트]")
for start in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for trail in [0.5, 1.0, 1.5]:
        trades = advanced_backtest(
            tp=10.0, sl=0.7, interval=12, fvg_only=True, vol_min=2.0,
            trailing=True, trail_start=start, trail_pct=trail
        )
        stats = calc_stats(trades)
        if stats and stats['net_pnl'] > 0:
            print(f"  ✓ 시작 {start}%, 폭 {trail}%: 승률={stats['win_rate']:.1f}%, "
                  f"순수익={stats['net_pnl']:.4f}%, MDD={stats['mdd']:.1f}%")


# 3. 본절 이동 전략
print("\n" + "=" * 70)
print("3. 본절 이동 전략")
print("=" * 70)

print("\n[본절 이동 시점 테스트]")
for be_at in [0.5, 1.0, 1.5, 2.0]:
    for sl in [0.7, 1.0, 1.5]:
        trades = advanced_backtest(
            tp=5.5, sl=sl, interval=12, fvg_only=True, vol_min=2.0,
            breakeven=True, be_at=be_at
        )
        stats = calc_stats(trades)
        if stats:
            status = "✓" if stats['net_pnl'] > 0 else " "
            print(f"  {status} BE@{be_at}%, SL={sl}%: 승률={stats['win_rate']:.1f}%, "
                  f"순수익={stats['net_pnl']:.4f}%, BE={stats['be']}회")


# 4. 분할 익절 전략
print("\n" + "=" * 70)
print("4. 분할 익절 전략")
print("=" * 70)

print("\n[분할 익절 테스트]")
for partial_at in [2.0, 3.0, 4.0]:
    for partial_pct in [30, 50, 70]:
        trades = advanced_backtest(
            tp=6.0, sl=0.7, interval=12, fvg_only=True, vol_min=2.0,
            partial=True, partial_at=partial_at, partial_pct=partial_pct
        )
        stats = calc_stats(trades)
        if stats:
            status = "✓" if stats['net_pnl'] > 0 else " "
            print(f"  {status} {partial_pct}%@{partial_at}%: 승률={stats['win_rate']:.1f}%, "
                  f"순수익={stats['net_pnl']:.4f}%")


# 5. 시간 기반 청산
print("\n" + "=" * 70)
print("5. 시간 기반 청산")
print("=" * 70)

print("\n[시간 청산 테스트]")
for time_bars in [20, 30, 40, 50, 60]:
    trades = advanced_backtest(
        tp=5.5, sl=0.7, interval=12, fvg_only=True, vol_min=2.0,
        time_exit=time_bars
    )
    stats = calc_stats(trades)
    if stats:
        status = "✓" if stats['net_pnl'] > 0 else " "
        hours = time_bars * 15 / 60
        print(f"  {status} {time_bars}봉({hours:.0f}h): 승률={stats['win_rate']:.1f}%, "
              f"순수익={stats['net_pnl']:.4f}%, TO={stats['timeout']}회")


# 6. 복합 전략
print("\n" + "=" * 70)
print("6. 복합 전략 (최적 조합)")
print("=" * 70)

combos = [
    # (name, tp, sl, trailing, trail_start, trail_pct, breakeven, be_at, partial, partial_at, partial_pct, time_exit)
    ('기본', 5.5, 0.7, False, 0, 0, False, 0, False, 0, 0, 0),
    ('넓은SL', 5.5, 1.5, False, 0, 0, False, 0, False, 0, 0, 0),
    ('트레일링', 10.0, 0.7, True, 2.0, 1.0, False, 0, False, 0, 0, 0),
    ('본절이동', 5.5, 0.7, False, 0, 0, True, 1.5, False, 0, 0, 0),
    ('분할익절', 6.0, 0.7, False, 0, 0, False, 0, True, 3.0, 50, 0),
    ('트레일+본절', 10.0, 0.7, True, 2.0, 1.0, True, 1.0, False, 0, 0, 0),
    ('분할+본절', 6.0, 0.7, False, 0, 0, True, 1.5, True, 3.0, 50, 0),
    ('트레일+분할', 10.0, 0.7, True, 2.0, 1.0, False, 0, True, 2.0, 50, 0),
    ('올인원', 10.0, 1.0, True, 2.0, 1.0, True, 1.0, True, 2.0, 50, 0),
    ('시간제한+트레일', 10.0, 0.7, True, 2.0, 1.0, False, 0, False, 0, 0, 50),
]

results = []
for name, tp, sl, trailing, t_start, t_pct, be, be_at, partial, p_at, p_pct, time_ex in combos:
    trades = advanced_backtest(
        tp=tp, sl=sl, interval=12, fvg_only=True, vol_min=2.0,
        trailing=trailing, trail_start=t_start, trail_pct=t_pct,
        breakeven=be, be_at=be_at,
        partial=partial, partial_at=p_at, partial_pct=p_pct,
        time_exit=time_ex
    )
    stats = calc_stats(trades)
    if stats:
        results.append({
            'name': name,
            'trades': stats['n'],
            'win_rate': stats['win_rate'],
            'net_pnl': stats['net_pnl'],
            'total_pnl': stats['total_pnl'],
            'mdd': stats['mdd'],
            'monthly': stats['total_pnl'] / months
        })

results_df = pd.DataFrame(results).sort_values('net_pnl', ascending=False)
print("\n[복합 전략 비교]")
print(results_df.to_string(index=False))


# 7. 최종 최적화
print("\n" + "=" * 70)
print("7. 최적 전략 세부 튜닝")
print("=" * 70)

# 가장 좋은 조합 찾기
best_results = []

for tp in [8.0, 10.0, 12.0]:
    for sl in [0.7, 1.0, 1.5]:
        for t_start in [1.5, 2.0, 2.5]:
            for t_pct in [0.7, 1.0, 1.5]:
                for be_at in [0.7, 1.0, 1.5]:
                    trades = advanced_backtest(
                        tp=tp, sl=sl, interval=12, fvg_only=True, vol_min=2.0,
                        trailing=True, trail_start=t_start, trail_pct=t_pct,
                        breakeven=True, be_at=be_at
                    )
                    stats = calc_stats(trades)
                    if stats and stats['net_pnl'] > 0.1:
                        best_results.append({
                            'tp': tp, 'sl': sl, 
                            't_start': t_start, 't_pct': t_pct, 
                            'be_at': be_at,
                            'win_rate': stats['win_rate'],
                            'net_pnl': stats['net_pnl'],
                            'total_pnl': stats['total_pnl'],
                            'mdd': stats['mdd']
                        })

if best_results:
    best_df = pd.DataFrame(best_results).sort_values('net_pnl', ascending=False)
    print("\n[상위 10개 설정]")
    print(best_df.head(10).to_string(index=False))
    
    # 최고 설정 상세
    best = best_df.iloc[0]
    print(f"\n★ 최적 설정 ★")
    print(f"  TP: {best['tp']}%")
    print(f"  SL: {best['sl']}%")
    print(f"  트레일링: {best['t_start']}% 시작, {best['t_pct']}% 폭")
    print(f"  본절: {best['be_at']}%에서 이동")
    print(f"\n  승률: {best['win_rate']:.1f}%")
    print(f"  순수익: {best['net_pnl']:.4f}%/거래")
    print(f"  총수익: {best['total_pnl']:.1f}%")
    print(f"  월수익: {best['total_pnl']/months:.2f}%")
    print(f"  MDD: {best['mdd']:.1f}%")
else:
    print("순수익 0.1% 이상 설정 없음")


# 8. 최종 비교
print("\n" + "=" * 70)
print("8. 최종 비교 요약")
print("=" * 70)

print("""
| 전략        | 승률  | 순수익/거래 | 월수익 | MDD    |
|-------------|-------|------------|--------|--------|
| 기존(미래참조)| 84%   | +0.70%     | 14%    | -5%    |
| 나우캐스트기본| 27.5% | +0.13%     | 1.0%   | -14.6% |
""")

if best_results:
    best = best_df.iloc[0]
    print(f"| 최적화      | {best['win_rate']:.1f}% | {best['net_pnl']:+.4f}%    | {best['total_pnl']/months:.1f}%   | {best['mdd']:.1f}%  |")

print("\n" + "=" * 70)
print("분석 완료!")
print("=" * 70)
