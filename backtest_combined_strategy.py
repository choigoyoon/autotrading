import pandas as pd
import numpy as np

"""
통합 전략: 추세선 돌파 + HL 변곡점 + EMA 필터

핵심 로직:
1. EMA 200 위에 있어야 함 (상승 추세)
2. H3 추세선 돌파 발생 (하락 추세 끝)
3. HL (Higher Low) 발생 (반전 확인)
4. 변곡점 캔들에서 진입 (정확한 타이밍)

이렇게 하면 세 가지 확인을 거치므로 신뢰도가 높아짐!
"""

# 데이터 로드
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])

print("="*100)
print("📈 통합 전략: 추세선 돌파 + HL + EMA 필터")
print("="*100)

# EMA 200 계산
EMA_PERIOD = 200
candles_df['ema_200'] = candles_df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
candles_df['above_ema'] = candles_df['close'] > candles_df['ema_200']

# 파라미터
WINDOW = 10
BREAKOUT_VALID_WINDOW = 48  # 돌파 후 12시간(48캔들) 이내 유효

trades = []
position = None

# Swing High/Low 리스트
confirmed_highs = []
recent_lows = []

# 추세선 돌파 상태
trendline_broken = False
trendline_break_idx = None
h3_price = None

# 통계
total_breakouts = 0
breakouts_above_ema = 0
hl_after_breakout = 0
final_entries = 0

print("\n시뮬레이션 시작...")

for i in range(WINDOW * 2 + EMA_PERIOD, len(candles_df)):
    candle = candles_df.iloc[i]
    prev_candles = candles_df.iloc[max(0, i-20):i]
    
    # === 1. Swing High 확정 ===
    check_idx = i - WINDOW
    start_idx = max(0, check_idx - WINDOW)
    end_idx = check_idx + WINDOW + 1
    
    if end_idx <= len(candles_df):
        window_highs = candles_df.iloc[start_idx:end_idx]['high'].values
        check_high = candles_df.iloc[check_idx]['high']
        
        if check_high == window_highs.max():
            confirmed_highs.append({
                'index': check_idx,
                'price': check_high,
                'time': candles_df.iloc[check_idx]['datetime']
            })
            if len(confirmed_highs) > 10:
                confirmed_highs.pop(0)
    
    # === 2. 추세선 돌파 감지 ===
    if not trendline_broken and len(confirmed_highs) >= 3:
        h1 = confirmed_highs[-3]
        h2 = confirmed_highs[-2]
        h3 = confirmed_highs[-1]
        
        # Lower Highs 조건
        if h1['price'] > h2['price'] > h3['price']:
            # 돌파 확인
            if candle['close'] > h3['price']:
                prev_close = candles_df.iloc[i-1]['close']
                if prev_close <= h3['price']:
                    total_breakouts += 1
                    
                    # EMA 200 위에서 돌파?
                    if candles_df.iloc[i]['above_ema']:
                        breakouts_above_ema += 1
                        trendline_broken = True
                        trendline_break_idx = i
                        h3_price = h3['price']
    
    # 돌파 유효 시간 체크
    if trendline_broken and trendline_break_idx:
        if i - trendline_break_idx > BREAKOUT_VALID_WINDOW:
            trendline_broken = False
            trendline_break_idx = None
            h3_price = None
    
    # === 3. 저점 감지 ===
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
    
    # === 4. 청산 체크 ===
    if position is not None:
        # SL
        if candle['low'] <= position['sl_price']:
            pnl_pct = ((position['sl_price'] - position['entry_price']) / position['entry_price']) * 100 - 0.11
            trades.append({
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': candle['datetime'],
                'exit_price': position['sl_price'],
                'exit_reason': 'SL',
                'pnl_pct': pnl_pct,
                'hl_strength': position['hl_strength']
            })
            position = None
            continue
        
        # TP2
        if candle['high'] >= position['tp2_price']:
            pnl_pct = ((position['tp2_price'] - position['entry_price']) / position['entry_price']) * 100 - 0.11
            trades.append({
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': candle['datetime'],
                'exit_price': position['tp2_price'],
                'exit_reason': 'TP2',
                'pnl_pct': pnl_pct,
                'hl_strength': position['hl_strength']
            })
            position = None
            continue
        
        # TP1
        if candle['high'] >= position['tp1_price']:
            pnl_pct = ((position['tp1_price'] - position['entry_price']) / position['entry_price']) * 100 - 0.11
            trades.append({
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': candle['datetime'],
                'exit_price': position['tp1_price'],
                'exit_reason': 'TP1',
                'pnl_pct': pnl_pct,
                'hl_strength': position['hl_strength']
            })
            position = None
            continue
    
    # === 5. HL + 변곡점 진입 (추세선 돌파 후에만) ===
    if position is None and trendline_broken:
        # HL 감지
        if len(recent_lows) >= 2:
            current_low = recent_lows[-1]['price']
            previous_low = recent_lows[-2]['price']
            
            if current_low > previous_low:
                if candle['close'] > current_low * 1.003:
                    hl_strength = ((current_low - previous_low) / previous_low) * 100
                    
                    if hl_strength >= 0.5:
                        hl_after_breakout += 1
                        
                        # 변곡점 캔들 조건
                        if candle['close'] > candle['open']:
                            body_size = candle['close'] - candle['open']
                            body_pct = (body_size / candle['open']) * 100
                            total_range = candle['high'] - candle['low']
                            
                            if total_range > 0 and body_pct >= 0.3:
                                body_to_range = (body_size / total_range) * 100
                                
                                if body_to_range >= 60:
                                    # 거래량 확인
                                    if i >= 5:
                                        prev_vol = candles_df.iloc[i-5:i]['volume'].mean()
                                        vol_ratio = candle['volume'] / prev_vol if prev_vol > 0 else 1
                                        
                                        if vol_ratio >= 1.0:
                                            final_entries += 1
                                            
                                            # 진입!
                                            entry_price = candle['close']
                                            
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
                                                'sl_price': current_low * 0.99,
                                                'hl_strength': hl_strength
                                            }
                                            
                                            # 돌파 상태 리셋
                                            trendline_broken = False
                                            trendline_break_idx = None

print(f"\n백테스트 완료!")

# 결과 분석
trades_df = pd.DataFrame(trades)

print(f"\n📊 통합 전략 통계:")
print(f"  추세선 돌파: {total_breakouts}회")
print(f"  EMA 위 돌파: {breakouts_above_ema}회 ({breakouts_above_ema/total_breakouts*100 if total_breakouts > 0 else 0:.1f}%)")
print(f"  돌파 후 HL: {hl_after_breakout}회")
print(f"  최종 진입: {len(trades_df)}건")

if len(trades_df) > 0:
    tp_trades = trades_df[trades_df['exit_reason'].str.contains('TP')]
    sl_trades = trades_df[trades_df['exit_reason'] == 'SL']
    
    win_rate = len(tp_trades) / len(trades_df) * 100
    total_pnl = trades_df['pnl_pct'].sum()
    avg_pnl = trades_df['pnl_pct'].mean()
    
    print("\n" + "="*100)
    print("📈 통합 전략 결과")
    print("="*100)
    print(f"  총 거래: {len(trades_df)}건")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  SL 비율: {len(sl_trades)/len(trades_df)*100:.1f}%")
    print(f"  총 PNL: {total_pnl:.2f}%")
    print(f"  평균 PNL: {avg_pnl:.2f}%")
    
    # 청산 사유별
    print("\n청산 사유별:")
    for reason in trades_df['exit_reason'].unique():
        rt = trades_df[trades_df['exit_reason'] == reason]
        print(f"  {reason}: {len(rt)}건, 평균 {rt['pnl_pct'].mean():.2f}%")
    
    # 연도별
    trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
    print("\n연도별:")
    for year in sorted(trades_df['year'].unique()):
        yt = trades_df[trades_df['year'] == year]
        yr_win = len(yt[yt['exit_reason'].str.contains('TP')]) / len(yt) * 100 if len(yt) > 0 else 0
        print(f"  {year}: {len(yt)}건, PNL {yt['pnl_pct'].sum():+.2f}%, 승률 {yr_win:.1f}%")
    
    # 기존 전략과 비교
    print("\n" + "="*100)
    print("📊 전략 비교")
    print("="*100)
    
    old_df = pd.read_csv('backtest_inflection_no_lookahead_results.csv')
    old_pnl = old_df['pnl_pct'].sum()
    old_win_rate = len(old_df[old_df['exit_reason'].str.contains('TP')]) / len(old_df) * 100
    old_avg = old_df['pnl_pct'].mean()
    
    print(f"\n{'지표':<15} {'기존 HL전략':<20} {'통합 전략':<20} {'차이':<15}")
    print("-"*70)
    print(f"{'거래 수':<15} {len(old_df):<20} {len(trades_df):<20} {len(trades_df)-len(old_df):+d}")
    print(f"{'총 PNL':<15} {old_pnl:<20.2f} {total_pnl:<20.2f} {total_pnl-old_pnl:+.2f}%p")
    print(f"{'평균 PNL':<15} {old_avg:<20.2f} {avg_pnl:<20.2f} {avg_pnl-old_avg:+.2f}%p")
    print(f"{'승률':<15} {old_win_rate:<20.1f} {win_rate:<20.1f} {win_rate-old_win_rate:+.1f}%p")
    
    trades_df.to_csv('backtest_combined_strategy_results.csv', index=False)
    print(f"\n✅ 결과 저장 완료")
else:
    print("\n⚠️ 거래가 없습니다.")
