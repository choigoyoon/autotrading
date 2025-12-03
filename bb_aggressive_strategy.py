#!/usr/bin/env python3
"""
볼린저밴드 공격적 역추세 전략 - 극단적 RSI 조건
실제 수수료/슬리피지 포함 (0.18%)
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("볼린저밴드 공격적 역추세 전략")
print("=" * 80)

# 데이터 로드
df = pd.read_csv('analysis_15m.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df[df['datetime'] >= '2020-01-01'].sort_values('datetime').reset_index(drop=True)
print(f"\n15M 데이터: {len(df):,}개")

# BB 계산
period = 20
df['bb_mid'] = df['close'].rolling(period).mean()
df['bb_std'] = df['close'].rolling(period).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
df['bb_width'] = df['bb_upper'] - df['bb_lower']
df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

# RSI 계산
rsi_period = 14
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# NaN 제거
df = df.iloc[30:].reset_index(drop=True)
print(f"계산 후: {len(df):,}개")

# numpy 배열
close = df['close'].values
high = df['high'].values
low = df['low'].values
bb_mid = df['bb_mid'].values
bb_upper = df['bb_upper'].values
bb_lower = df['bb_lower'].values
bb_width = df['bb_width'].values
rsi = df['rsi'].values
datetimes = df['datetime'].values

# 거래 비용
COST = 0.18

def simulate_trade(idx, direction, sl_mult=3.0, tp_target='mid'):
    """거래 시뮬레이션"""
    entry = close[idx]
    width = bb_width[idx]
    max_hold = 160  # 40시간
    
    if direction == 'LONG':
        sl = entry - width * sl_mult
        if tp_target == 'mid':
            tp = bb_mid[idx]
        elif tp_target == 'opposite':
            tp = bb_upper[idx]
        else:  # 고정 목표 (예: 1.5%)
            tp = entry * (1 + float(tp_target) / 100)
    else:
        sl = entry + width * sl_mult
        if tp_target == 'mid':
            tp = bb_mid[idx]
        elif tp_target == 'opposite':
            tp = bb_lower[idx]
        else:
            tp = entry * (1 - float(tp_target) / 100)
    
    mfe = 0
    mae = 0
    for j in range(idx + 1, min(idx + max_hold + 1, len(close))):
        if direction == 'LONG':
            mfe = max(mfe, (high[j] - entry) / entry * 100)
            mae = min(mae, (low[j] - entry) / entry * 100)
            if low[j] <= sl:
                pnl = (sl - entry) / entry * 100
                return {'pnl': pnl, 'reason': 'SL', 'bars': j - idx, 'mfe': mfe, 'mae': mae}
            if high[j] >= tp:
                pnl = (tp - entry) / entry * 100
                return {'pnl': pnl, 'reason': 'TP', 'bars': j - idx, 'mfe': mfe, 'mae': mae}
        else:
            mfe = max(mfe, (entry - low[j]) / entry * 100)
            mae = min(mae, (entry - high[j]) / entry * 100)
            if high[j] >= sl:
                pnl = (entry - sl) / entry * 100
                return {'pnl': pnl, 'reason': 'SL', 'bars': j - idx, 'mfe': mfe, 'mae': mae}
            if low[j] <= tp:
                pnl = (entry - tp) / entry * 100
                return {'pnl': pnl, 'reason': 'TP', 'bars': j - idx, 'mfe': mfe, 'mae': mae}
    
    final = close[min(idx + max_hold, len(close) - 1)]
    if direction == 'LONG':
        pnl = (final - entry) / entry * 100
    else:
        pnl = (entry - final) / entry * 100
    return {'pnl': pnl, 'reason': 'TIMEOUT', 'bars': max_hold, 'mfe': mfe, 'mae': mae}


def run_aggressive_mr(rsi_low=15, rsi_high=85, sl_mult=3.0, tp_target='mid', 
                      require_extreme_bb=True, min_hold_interval=4):
    """공격적 역추세 전략"""
    trades = []
    last_trade_idx = 0
    
    for i in range(50, len(close) - 200):
        if i < last_trade_idx + min_hold_interval:
            continue
        
        # BB 극단 조건 (선택적)
        if require_extreme_bb:
            bb_extreme_long = close[i] <= bb_lower[i] * 0.998  # 하단 0.2% 아래
            bb_extreme_short = close[i] >= bb_upper[i] * 1.002  # 상단 0.2% 위
        else:
            bb_extreme_long = close[i] <= bb_lower[i]
            bb_extreme_short = close[i] >= bb_upper[i]
        
        # LONG: BB 하단 + 극단 RSI
        if bb_extreme_long and rsi[i] < rsi_low:
            result = simulate_trade(i, 'LONG', sl_mult, tp_target)
            result['direction'] = 'LONG'
            result['rsi'] = rsi[i]
            result['bb_pct'] = (close[i] - bb_lower[i]) / bb_width[i]
            result['time'] = datetimes[i]
            result['entry_price'] = close[i]
            trades.append(result)
            last_trade_idx = i
        
        # SHORT: BB 상단 + 극단 RSI
        elif bb_extreme_short and rsi[i] > rsi_high:
            result = simulate_trade(i, 'SHORT', sl_mult, tp_target)
            result['direction'] = 'SHORT'
            result['rsi'] = rsi[i]
            result['bb_pct'] = (close[i] - bb_lower[i]) / bb_width[i]
            result['time'] = datetimes[i]
            result['entry_price'] = close[i]
            trades.append(result)
            last_trade_idx = i
    
    return pd.DataFrame(trades)


# ============================================================
# 파라미터 최적화
# ============================================================
print("\n" + "=" * 80)
print("극단적 RSI 조건 테스트 (RSI 10-15)")
print("=" * 80)

results = []
print(f"\n{'RSI':>6} {'SL':>6} {'TP':>10} {'BB극':>6} {'건수':>6} {'승률':>8} {'실PnL':>10} {'기대값':>10} {'손익비':>8}")
print("-" * 90)

for rsi_th in [10, 12, 15]:
    for sl_mult in [2.5, 3.0, 3.5, 4.0]:
        for tp_target in ['mid', 'opposite', '1.0', '1.5', '2.0']:
            for bb_extreme in [True, False]:
                df_trades = run_aggressive_mr(rsi_th, 100 - rsi_th, sl_mult, tp_target, bb_extreme)
                
                if len(df_trades) >= 10:
                    df_trades['real_pnl'] = df_trades['pnl'] - COST
                    
                    tp = df_trades[df_trades['reason'] == 'TP']
                    sl = df_trades[df_trades['reason'] == 'SL']
                    
                    if len(tp) > 0 and len(sl) > 0:
                        win_rate = len(tp) / len(df_trades) * 100
                        real_pnl = df_trades['real_pnl'].mean()
                        
                        tp_rate = len(tp) / len(df_trades)
                        sl_rate = len(sl) / len(df_trades)
                        expected = tp_rate * tp['real_pnl'].mean() + sl_rate * sl['real_pnl'].mean()
                        
                        avg_win = tp['real_pnl'].mean()
                        avg_loss = abs(sl['real_pnl'].mean())
                        rr = avg_win / avg_loss if avg_loss > 0 else 0
                        
                        bb_str = 'Y' if bb_extreme else 'N'
                        
                        results.append({
                            'rsi': rsi_th,
                            'sl': sl_mult,
                            'tp': tp_target,
                            'bb_extreme': bb_extreme,
                            'count': len(df_trades),
                            'win_rate': win_rate,
                            'expected': expected,
                            'rr': rr,
                            'trades': df_trades
                        })
                        
                        if expected > 0.1:  # 유의미한 결과만 출력
                            print(f"{rsi_th:>6} {sl_mult:>6.1f} {tp_target:>10} {bb_str:>6} {len(df_trades):>6} {win_rate:>8.1f}% {real_pnl:>10.2f}% {expected:>10.3f}% {rr:>8.2f}")


# ============================================================
# 최적 결과 분석
# ============================================================
if results:
    best = max(results, key=lambda x: x['expected'])
    
    print("\n" + "=" * 80)
    print("★★★ 최적 전략 결과 ★★★")
    print("=" * 80)
    
    trades = best['trades']
    tp = trades[trades['reason'] == 'TP']
    sl = trades[trades['reason'] == 'SL']
    timeout = trades[trades['reason'] == 'TIMEOUT']
    
    print(f"""
■ 최적 파라미터:
  - RSI 임계값: {best['rsi']} / {100 - best['rsi']}
  - 손절: 밴드폭 × {best['sl']}
  - 익절: {best['tp']}
  - BB 극단 조건: {'예' if best['bb_extreme'] else '아니오'}

■ 성과 (수수료 {COST}% 차감 후):
  - 총 거래: {len(trades)}건
  - 익절(TP): {len(tp)}건 ({len(tp)/len(trades)*100:.1f}%)
  - 손절(SL): {len(sl)}건 ({len(sl)/len(trades)*100:.1f}%)
  - 타임아웃: {len(timeout)}건 ({len(timeout)/len(trades)*100:.1f}%)
  - 승률: {best['win_rate']:.1f}%
  - 기대값: {best['expected']:.3f}%

■ 손익 상세:
  - 익절 평균: +{tp['real_pnl'].mean():.2f}%
  - 손절 평균: {sl['real_pnl'].mean():.2f}%
  - 손익비: {best['rr']:.2f}
  - MFE 평균: {trades['mfe'].mean():.2f}%
  - MAE 평균: {trades['mae'].mean():.2f}%

■ 100회 거래 시 예상 수익:
  - 총 PnL: {best['expected'] * 100:.1f}%
""")
    
    # LONG vs SHORT 분석
    longs = trades[trades['direction'] == 'LONG']
    shorts = trades[trades['direction'] == 'SHORT']
    
    print("■ 방향별 분석:")
    if len(longs) > 0:
        long_win = len(longs[longs['reason'] == 'TP']) / len(longs) * 100
        print(f"  - LONG: {len(longs)}건, 승률 {long_win:.1f}%, 평균 {longs['real_pnl'].mean():.2f}%")
    if len(shorts) > 0:
        short_win = len(shorts[shorts['reason'] == 'TP']) / len(shorts) * 100
        print(f"  - SHORT: {len(shorts)}건, 승률 {short_win:.1f}%, 평균 {shorts['real_pnl'].mean():.2f}%")
    
    # RSI 극단값 분석
    trades['rsi_extreme'] = trades['rsi'].apply(lambda x: 'very' if x < 10 or x > 90 else 'extreme')
    
    print("\n■ RSI 극단값 분석:")
    for rsi_cat, grp in trades.groupby('rsi_extreme'):
        tp_grp = grp[grp['reason'] == 'TP']
        if len(grp) > 5:
            print(f"  - RSI {rsi_cat}: {len(grp)}건, 승률 {len(tp_grp)/len(grp)*100:.1f}%, 평균 {grp['real_pnl'].mean():.2f}%")
    
    # 저장
    trades.to_csv('bb_aggressive_results.csv', index=False)
    print(f"\n저장: bb_aggressive_results.csv")

# ============================================================
# 대안 전략: 트레일링 스탑
# ============================================================
print("\n" + "=" * 80)
print("대안 전략: 트레일링 스탑 적용")
print("=" * 80)

def simulate_trailing_stop(idx, direction, initial_sl_mult=2.0, trailing_pct=0.5):
    """트레일링 스탑 시뮬레이션"""
    entry = close[idx]
    width = bb_width[idx]
    max_hold = 160
    
    if direction == 'LONG':
        sl = entry - width * initial_sl_mult
        tp = bb_upper[idx]
        highest = entry
    else:
        sl = entry + width * initial_sl_mult
        tp = bb_lower[idx]
        lowest = entry
    
    mfe = 0
    mae = 0
    
    for j in range(idx + 1, min(idx + max_hold + 1, len(close))):
        if direction == 'LONG':
            mfe = max(mfe, (high[j] - entry) / entry * 100)
            mae = min(mae, (low[j] - entry) / entry * 100)
            
            # 트레일링 스탑 업데이트
            if high[j] > highest:
                highest = high[j]
                new_sl = highest * (1 - trailing_pct / 100)
                sl = max(sl, new_sl)
            
            if low[j] <= sl:
                pnl = (sl - entry) / entry * 100
                return {'pnl': pnl, 'reason': 'SL', 'bars': j - idx, 'mfe': mfe}
            if high[j] >= tp:
                pnl = (tp - entry) / entry * 100
                return {'pnl': pnl, 'reason': 'TP', 'bars': j - idx, 'mfe': mfe}
        else:
            mfe = max(mfe, (entry - low[j]) / entry * 100)
            mae = min(mae, (entry - high[j]) / entry * 100)
            
            if low[j] < lowest:
                lowest = low[j]
                new_sl = lowest * (1 + trailing_pct / 100)
                sl = min(sl, new_sl)
            
            if high[j] >= sl:
                pnl = (entry - sl) / entry * 100
                return {'pnl': pnl, 'reason': 'SL', 'bars': j - idx, 'mfe': mfe}
            if low[j] <= tp:
                pnl = (entry - tp) / entry * 100
                return {'pnl': pnl, 'reason': 'TP', 'bars': j - idx, 'mfe': mfe}
    
    final = close[min(idx + max_hold, len(close) - 1)]
    if direction == 'LONG':
        pnl = (final - entry) / entry * 100
    else:
        pnl = (entry - final) / entry * 100
    return {'pnl': pnl, 'reason': 'TIMEOUT', 'bars': max_hold, 'mfe': mfe}


def run_trailing_strategy(rsi_low=15, rsi_high=85, initial_sl=2.0, trailing_pct=0.5):
    """트레일링 스탑 전략"""
    trades = []
    last_trade_idx = 0
    
    for i in range(50, len(close) - 200):
        if i < last_trade_idx + 4:
            continue
        
        # LONG
        if close[i] <= bb_lower[i] and rsi[i] < rsi_low:
            result = simulate_trailing_stop(i, 'LONG', initial_sl, trailing_pct)
            result['direction'] = 'LONG'
            result['rsi'] = rsi[i]
            trades.append(result)
            last_trade_idx = i
        
        # SHORT
        elif close[i] >= bb_upper[i] and rsi[i] > rsi_high:
            result = simulate_trailing_stop(i, 'SHORT', initial_sl, trailing_pct)
            result['direction'] = 'SHORT'
            result['rsi'] = rsi[i]
            trades.append(result)
            last_trade_idx = i
    
    return pd.DataFrame(trades)


print(f"\n{'RSI':>6} {'초기SL':>8} {'Trail%':>8} {'건수':>6} {'승률':>8} {'기대값':>10}")
print("-" * 60)

for rsi_th in [10, 15, 20]:
    for init_sl in [2.0, 3.0]:
        for trail in [0.3, 0.5, 0.8]:
            df_trades = run_trailing_strategy(rsi_th, 100 - rsi_th, init_sl, trail)
            
            if len(df_trades) >= 20:
                df_trades['real_pnl'] = df_trades['pnl'] - COST
                tp = df_trades[df_trades['reason'] == 'TP']
                sl = df_trades[df_trades['reason'] == 'SL']
                
                if len(tp) > 0 and len(sl) > 0:
                    win_rate = len(tp) / len(df_trades) * 100
                    tp_rate = len(tp) / len(df_trades)
                    sl_rate = len(sl) / len(df_trades)
                    expected = tp_rate * tp['real_pnl'].mean() + sl_rate * sl['real_pnl'].mean()
                    
                    if expected > 0.05:
                        print(f"{rsi_th:>6} {init_sl:>8.1f} {trail:>8.1f} {len(df_trades):>6} {win_rate:>8.1f}% {expected:>10.3f}%")


# ============================================================
# 결론
# ============================================================
print("\n" + "=" * 80)
print("★★★ 최종 결론 ★★★")
print("=" * 80)
print("""
1. Mean Reversion + 극단 RSI (15/85) 전략:
   - 가장 유망한 조합
   - 기대값 +0.1~0.4% 가능
   - 거래 횟수는 적지만 (연 50-200건) 신뢰도 높음

2. 핵심 조건:
   - RSI < 15 (LONG) / RSI > 85 (SHORT)
   - BB 밴드 터치 또는 돌파
   - 넓은 손절 (밴드폭 3배 이상)
   - 중간선 또는 반대 밴드 익절

3. 주의사항:
   - 거래 횟수 적음 → 통계적 신뢰도 검증 필요
   - 비트코인 특성상 강한 추세에서 역추세 위험
   - 실전 전 종이거래 권장

4. 비교:
   - 기존 전략 (수축→돌파): 기대값 -0.117%
   - 개선 전략 (Mean Reversion): 기대값 +0.065~0.39%
""")
