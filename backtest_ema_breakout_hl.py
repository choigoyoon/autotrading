import pandas as pd
import numpy as np

"""
EMA 돌파 + HL + 변곡점 캔들 전략

핵심 아이디어:
1. EMA 200을 아래에서 위로 돌파하는 순간 포착
2. 돌파 후 HL(Higher Low) 발생 확인
3. 변곡점 캔들(강한 양봉)에서 진입

이렇게 하면:
- 가짜 반등 필터링 (EMA 돌파 확인)
- 추세 전환 확정 (HL 발생)
- 최적 진입 시점 (변곡점 캔들)
"""

# 데이터 로드
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])

print("="*100)
print("📈 EMA 돌파 + HL + 변곡점 캔들 전략 백테스트")
print("="*100)

# EMA 200 계산 (15분봉 기준)
EMA_PERIOD = 200
candles_df['ema_200'] = candles_df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
candles_df['above_ema'] = candles_df['close'] > candles_df['ema_200']

# EMA 돌파 감지 (아래 → 위)
candles_df['ema_breakout'] = (candles_df['above_ema'] == True) & (candles_df['above_ema'].shift(1) == False)

print(f"✅ EMA 200 계산 완료")
print(f"   EMA 돌파 횟수: {candles_df['ema_breakout'].sum()}회")

# 백테스트 시작
trades = []
position = None

recent_lows = []
searching_inflection = False
hl_event = None

# EMA 돌파 상태 추적
ema_breakout_time = None
ema_breakout_active = False
BREAKOUT_WINDOW = 96  # 돌파 후 24시간(96개 15분봉) 이내 진입

# 통계
total_hl_events = 0
hl_after_breakout = 0
inflection_after_breakout = 0

print("\n시뮬레이션 시작...")

for i in range(EMA_PERIOD + 100, len(candles_df)):
    candle = candles_df.iloc[i]
    prev_candles = candles_df.iloc[max(0, i-20):i]
    
    # === EMA 돌파 감지 ===
    if candles_df.iloc[i]['ema_breakout']:
        ema_breakout_time = i
        ema_breakout_active = True
    
    # 돌파 후 24시간 지나면 비활성화
    if ema_breakout_active and ema_breakout_time:
        if i - ema_breakout_time > BREAKOUT_WINDOW:
            ema_breakout_active = False
    
    # === 저점 감지 ===
    if len(prev_candles) >= 10:
        recent_low = prev_candles['low'].tail(10).min()
        
        if candle['low'] <= recent_low * 1.002:
            recent_lows.append({
                'price': candle['low'],
                'time': candle['datetime'],
                'index': i
            })
            if len(recent_lows) > 5:
                recent_lows.pop(0)
    
    # === HL 감지 ===
    if len(recent_lows) >= 2 and not searching_inflection:
        current_low = recent_lows[-1]['price']
        previous_low = recent_lows[-2]['price']
        
        if current_low > previous_low:
            if candle['close'] > current_low * 1.003:
                hl_strength = ((current_low - previous_low) / previous_low) * 100
                
                if hl_strength >= 0.5:
                    total_hl_events += 1
                    
                    # EMA 돌파 후 HL인 경우만 진입 준비
                    if ema_breakout_active:
                        hl_after_breakout += 1
                        hl_event = {
                            'hl_time': recent_lows[-1]['time'],
                            'hl_price': current_low,
                            'hl_strength': hl_strength,
                            'hl_index': recent_lows[-1]['index'],
                            'breakout_idx': ema_breakout_time
                        }
                        searching_inflection = True
    
    # === 청산 체크 ===
    if position is not None:
        if candle['low'] <= position['sl_price']:
            exit_price = position['sl_price']
            pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
            trades.append({
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': candle['datetime'],
                'exit_price': exit_price,
                'exit_reason': 'SL',
                'pnl_pct': pnl_pct,
                'hl_strength': position['hl_strength'],
                'breakout_to_entry': position['breakout_to_entry']
            })
            position = None
            searching_inflection = False
            continue
        
        if candle['high'] >= position['tp2_price']:
            exit_price = position['tp2_price']
            pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
            trades.append({
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': candle['datetime'],
                'exit_price': exit_price,
                'exit_reason': 'TP2',
                'pnl_pct': pnl_pct,
                'hl_strength': position['hl_strength'],
                'breakout_to_entry': position['breakout_to_entry']
            })
            position = None
            searching_inflection = False
            continue
        
        if candle['high'] >= position['tp1_price']:
            exit_price = position['tp1_price']
            pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
            trades.append({
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': candle['datetime'],
                'exit_price': exit_price,
                'exit_reason': 'TP1',
                'pnl_pct': pnl_pct,
                'hl_strength': position['hl_strength'],
                'breakout_to_entry': position['breakout_to_entry']
            })
            position = None
            searching_inflection = False
            continue
    
    # === 변곡점 캔들 찾기 ===
    if position is None and searching_inflection and hl_event is not None:
        if i - hl_event['hl_index'] > 20:
            searching_inflection = False
            hl_event = None
            continue
        
        # 변곡점 조건
        if candle['close'] <= candle['open']:
            continue
        
        body_size = candle['close'] - candle['open']
        body_pct = (body_size / candle['open']) * 100
        if body_pct < 0.3:
            continue
        
        total_range = candle['high'] - candle['low']
        if total_range == 0:
            continue
        body_to_range = (body_size / total_range) * 100
        if body_to_range < 60:
            continue
        
        if i >= 5:
            prev_volume_avg = candles_df.iloc[i-5:i]['volume'].mean()
            volume_ratio = candle['volume'] / prev_volume_avg if prev_volume_avg > 0 else 1
            if volume_ratio < 1.0:
                continue
        
        inflection_after_breakout += 1
        
        # 진입!
        entry_price = candle['close']
        hl_strength = hl_event['hl_strength']
        
        if hl_strength >= 5:
            tp1_pct, tp2_pct = 2.0, 4.0
        elif hl_strength >= 2:
            tp1_pct, tp2_pct = 1.5, 3.0
        elif hl_strength >= 1:
            tp1_pct, tp2_pct = 1.0, 2.0
        else:
            tp1_pct, tp2_pct = 0.7, 1.5
        
        position = {
            'entry_time': candle['datetime'],
            'entry_price': entry_price,
            'tp1_price': entry_price * (1 + tp1_pct / 100),
            'tp2_price': entry_price * (1 + tp2_pct / 100),
            'sl_price': hl_event['hl_price'] * 0.99,
            'hl_strength': hl_strength,
            'breakout_to_entry': i - hl_event['breakout_idx']
        }
        searching_inflection = False

print(f"\n백테스트 완료!")

# 결과 분석
trades_df = pd.DataFrame(trades)

print(f"\n📊 EMA 돌파 + HL 통계:")
print(f"  전체 HL 이벤트: {total_hl_events}개")
print(f"  EMA 돌파 후 HL: {hl_after_breakout}개 ({hl_after_breakout/total_hl_events*100 if total_hl_events > 0 else 0:.1f}%)")
print(f"  최종 진입: {len(trades_df)}개")

if len(trades_df) > 0:
    tp_trades = trades_df[trades_df['exit_reason'].str.contains('TP')]
    sl_trades = trades_df[trades_df['exit_reason'] == 'SL']
    
    win_rate = len(tp_trades) / len(trades_df) * 100
    total_pnl = trades_df['pnl_pct'].sum()
    avg_pnl = trades_df['pnl_pct'].mean()
    
    print("\n" + "="*100)
    print("📈 EMA 돌파 + HL 전략 결과")
    print("="*100)
    print(f"  총 거래: {len(trades_df)}건")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  SL 비율: {len(sl_trades)/len(trades_df)*100:.1f}%")
    print(f"  총 PNL: {total_pnl:.2f}%")
    print(f"  평균 PNL: {avg_pnl:.2f}%")
    print(f"  최대 이익: {trades_df['pnl_pct'].max():.2f}%")
    print(f"  최대 손실: {trades_df['pnl_pct'].min():.2f}%")
    
    # 청산 사유별
    print("\n청산 사유별:")
    for reason in trades_df['exit_reason'].unique():
        rt = trades_df[trades_df['exit_reason'] == reason]
        print(f"  {reason}: {len(rt)}건 ({len(rt)/len(trades_df)*100:.1f}%), 평균 {rt['pnl_pct'].mean():.2f}%")
    
    # 연도별
    trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
    print("\n연도별:")
    for year in sorted(trades_df['year'].unique()):
        yt = trades_df[trades_df['year'] == year]
        yr_win = len(yt[yt['exit_reason'].str.contains('TP')]) / len(yt) * 100
        print(f"  {year}: {len(yt)}건, PNL {yt['pnl_pct'].sum():+.2f}%, 승률 {yr_win:.1f}%")
    
    # 기존 전략과 비교
    print("\n" + "="*100)
    print("📊 전략 비교")
    print("="*100)
    
    old_df = pd.read_csv('backtest_inflection_no_lookahead_results.csv')
    old_pnl = old_df['pnl_pct'].sum()
    old_win_rate = len(old_df[old_df['exit_reason'].str.contains('TP')]) / len(old_df) * 100
    old_avg = old_df['pnl_pct'].mean()
    
    print(f"\n{'지표':<15} {'기존 HL전략':<20} {'EMA돌파+HL':<20} {'차이':<15}")
    print("-"*70)
    print(f"{'거래 수':<15} {len(old_df):<20} {len(trades_df):<20} {len(trades_df)-len(old_df):+d}")
    print(f"{'총 PNL':<15} {old_pnl:<20.2f} {total_pnl:<20.2f} {total_pnl-old_pnl:+.2f}%p")
    print(f"{'평균 PNL':<15} {old_avg:<20.2f} {avg_pnl:<20.2f} {avg_pnl-old_avg:+.2f}%p")
    print(f"{'승률':<15} {old_win_rate:<20.1f} {win_rate:<20.1f} {win_rate-old_win_rate:+.1f}%p")
    
    trades_df.to_csv('backtest_ema_breakout_hl_results.csv', index=False)
    print(f"\n✅ 결과 저장 완료")
else:
    print("\n⚠️ 거래가 없습니다. 조건을 완화해야 합니다.")
