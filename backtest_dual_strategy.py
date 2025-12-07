import pandas as pd
import numpy as np

"""
듀얼 전략: HL 반등 + H3 돌파

전략 1: HL 반등 (바닥 매수)
- 저점에서 Higher Low 발생
- 변곡점 캔들에서 진입

전략 2: H3 돌파 (돌파 매수)
- 하락 추세선 (H1 > H2 > H3) 형성
- H3 돌파 + 3캔들 확인 후 진입
"""

# 데이터 로드
candles = pd.read_csv('btc_15m_ohlcv.csv')
candles['datetime'] = pd.to_datetime(candles['datetime'])
candles['ema_200'] = candles['close'].ewm(span=200, adjust=False).mean()

print("="*100)
print("📈 듀얼 전략 백테스트: HL 반등 + H3 돌파")
print("="*100)

# 파라미터
SWING_WINDOW = 10
CONFIRM_CANDLES = 3
TP_PCT = 2.0
SL_PCT = 2.0

# 상태 변수
trades_hl = []  # HL 전략 거래
trades_h3 = []  # H3 전략 거래

position_hl = None
position_h3 = None

# HL 전략용
recent_lows = []
searching_inflection = False
hl_event = None

# H3 전략용
confirmed_highs = []
breakout_pending = None

print("\n시뮬레이션 시작...")

for i in range(SWING_WINDOW * 2 + 200, len(candles)):
    candle = candles.iloc[i]
    prev_candles = candles.iloc[max(0, i-20):i]
    ema = candles.iloc[i]['ema_200']
    
    # ========================================
    # 전략 1: HL 반등
    # ========================================
    
    # 저점 감지
    if len(prev_candles) >= 10:
        recent_low = prev_candles['low'].tail(10).min()
        if candle['low'] <= recent_low * 1.002:
            recent_lows.append({'price': candle['low'], 'index': i})
            if len(recent_lows) > 5:
                recent_lows.pop(0)
    
    # HL 감지
    if len(recent_lows) >= 2 and not searching_inflection and position_hl is None:
        current_low = recent_lows[-1]['price']
        previous_low = recent_lows[-2]['price']
        if current_low > previous_low and candle['close'] > current_low * 1.003:
            hl_strength = ((current_low - previous_low) / previous_low) * 100
            if hl_strength >= 0.5:
                hl_event = {'price': current_low, 'index': recent_lows[-1]['index'], 'strength': hl_strength}
                searching_inflection = True
    
    # HL 청산
    if position_hl is not None:
        # SL
        if candle['low'] <= position_hl['sl']:
            pnl = ((position_hl['sl'] - position_hl['entry']) / position_hl['entry']) * 100
            trades_hl.append({'time': candle['datetime'], 'pnl': pnl, 'reason': 'SL', 'type': 'HL'})
            position_hl = None
            searching_inflection = False
        # TP
        elif candle['high'] >= position_hl['tp']:
            pnl = ((position_hl['tp'] - position_hl['entry']) / position_hl['entry']) * 100
            trades_hl.append({'time': candle['datetime'], 'pnl': pnl, 'reason': 'TP', 'type': 'HL'})
            position_hl = None
            searching_inflection = False
        # EMA 이탈
        elif position_hl.get('above_ema') and candle['close'] < ema:
            pnl = ((candle['close'] - position_hl['entry']) / position_hl['entry']) * 100
            trades_hl.append({'time': candle['datetime'], 'pnl': pnl, 'reason': 'SL_EMA', 'type': 'HL'})
            position_hl = None
            searching_inflection = False
    
    # HL 진입 (변곡점 캔들)
    if position_hl is None and searching_inflection and hl_event:
        if i - hl_event['index'] > 20:
            searching_inflection = False
            hl_event = None
        elif candle['close'] > candle['open']:
            body = candle['close'] - candle['open']
            body_pct = (body / candle['open']) * 100
            rng = candle['high'] - candle['low']
            if rng > 0 and body_pct >= 0.3 and (body / rng) * 100 >= 60:
                entry = candle['close']
                hl_str = hl_event['strength']
                tp_pct = 1.5 if hl_str >= 2 else (1.0 if hl_str >= 1 else 0.7)
                
                position_hl = {
                    'entry': entry,
                    'tp': entry * (1 + tp_pct / 100),
                    'sl': hl_event['price'] * 0.99,
                    'above_ema': entry > ema
                }
                searching_inflection = False
    
    # ========================================
    # 전략 2: H3 돌파
    # ========================================
    
    # Swing High 확정 (10캔들 딜레이)
    check_idx = i - SWING_WINDOW
    start_idx = max(0, check_idx - SWING_WINDOW)
    end_idx = check_idx + SWING_WINDOW + 1
    
    if end_idx <= len(candles):
        window_highs = candles.iloc[start_idx:end_idx]['high'].values
        check_high = candles.iloc[check_idx]['high']
        
        if check_high == window_highs.max():
            confirmed_highs.append({'index': check_idx, 'price': check_high})
            if len(confirmed_highs) > 10:
                confirmed_highs.pop(0)
    
    # H3 청산
    if position_h3 is not None:
        # SL
        if candle['low'] <= position_h3['sl']:
            pnl = -SL_PCT - 0.11
            trades_h3.append({'time': candle['datetime'], 'pnl': pnl, 'reason': 'SL', 'type': 'H3'})
            position_h3 = None
        # TP
        elif candle['high'] >= position_h3['tp']:
            pnl = TP_PCT - 0.11
            trades_h3.append({'time': candle['datetime'], 'pnl': pnl, 'reason': 'TP', 'type': 'H3'})
            position_h3 = None
    
    # H3 돌파 대기 중
    if breakout_pending and position_h3 is None:
        breakout_pending['count'] += 1
        if breakout_pending['count'] >= CONFIRM_CANDLES:
            if candle['close'] > breakout_pending['h3_price']:
                entry = candle['close']
                position_h3 = {
                    'entry': entry,
                    'tp': entry * (1 + TP_PCT / 100),
                    'sl': entry * (1 - SL_PCT / 100)
                }
            breakout_pending = None
    
    # H3 돌파 감지
    if position_h3 is None and breakout_pending is None and len(confirmed_highs) >= 3:
        h1 = confirmed_highs[-3]
        h2 = confirmed_highs[-2]
        h3 = confirmed_highs[-1]
        
        # Lower Highs (하락 추세선)
        if h1['price'] > h2['price'] > h3['price']:
            # 돌파
            if candle['close'] > h3['price']:
                prev_close = candles.iloc[i-1]['close']
                if prev_close <= h3['price']:
                    breakout_pending = {'h3_price': h3['price'], 'count': 0}

print("\n백테스트 완료!")

# 결과 분석
trades_hl_df = pd.DataFrame(trades_hl)
trades_h3_df = pd.DataFrame(trades_h3)
all_trades = pd.concat([trades_hl_df, trades_h3_df], ignore_index=True)

print("\n" + "="*100)
print("📊 전략별 결과")
print("="*100)

for name, df in [("HL 반등", trades_hl_df), ("H3 돌파", trades_h3_df)]:
    if len(df) > 0:
        wins = len(df[df['reason'] == 'TP'])
        total = len(df)
        win_rate = wins / total * 100 if total > 0 else 0
        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()
        
        print(f"\n{name} 전략:")
        print(f"  거래 수: {total}건")
        print(f"  승률: {win_rate:.1f}%")
        print(f"  총 PNL: {total_pnl:.2f}%")
        print(f"  평균 PNL: {avg_pnl:.3f}%")

# 통합 결과
print("\n" + "="*100)
print("📈 통합 결과 (HL + H3)")
print("="*100)

if len(all_trades) > 0:
    wins = len(all_trades[all_trades['reason'] == 'TP'])
    total = len(all_trades)
    win_rate = wins / total * 100
    total_pnl = all_trades['pnl'].sum()
    avg_pnl = all_trades['pnl'].mean()
    
    print(f"\n총 거래: {total}건")
    print(f"  - HL 반등: {len(trades_hl_df)}건")
    print(f"  - H3 돌파: {len(trades_h3_df)}건")
    print(f"승률: {win_rate:.1f}%")
    print(f"총 PNL: {total_pnl:.2f}%")
    print(f"평균 PNL: {avg_pnl:.3f}%")
    
    # 연도별
    all_trades['year'] = pd.to_datetime(all_trades['time']).dt.year
    print("\n연도별:")
    for year in sorted(all_trades['year'].unique()):
        yt = all_trades[all_trades['year'] == year]
        yr_win = len(yt[yt['reason'] == 'TP']) / len(yt) * 100
        print(f"  {year}: {len(yt)}건, PNL {yt['pnl'].sum():+.2f}%, 승률 {yr_win:.1f}%")

# 기존 전략과 비교
print("\n" + "="*100)
print("📊 기존 HL 전략 vs 듀얼 전략")
print("="*100)

old_pnl = 169.40
old_trades = 690
old_winrate = 59.7

print(f"\n{'항목':<15} {'기존 HL':<20} {'듀얼 (HL+H3)':<20} {'차이':<15}")
print("-"*70)
print(f"{'거래 수':<15} {old_trades:<20} {len(all_trades):<20} {len(all_trades)-old_trades:+d}")
print(f"{'총 PNL':<15} {old_pnl:<20.2f} {total_pnl:<20.2f} {total_pnl-old_pnl:+.2f}%p")
print(f"{'승률':<15} {old_winrate:<20.1f} {win_rate:<20.1f} {win_rate-old_winrate:+.1f}%p")
