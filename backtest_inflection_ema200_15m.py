import pandas as pd
import numpy as np

# 데이터 로드
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])

print("="*100)
print("변곡점 캔들 진입 백테스트 + EMA 200 필터 (15분봉 기준)")
print("="*100)

# === 15분봉 기준 EMA 200 계산 ===
# 200개 15분봉 = 50시간 = 약 2일
EMA_PERIOD = 200
print(f"\n15분봉 EMA 200 계산 중 ({EMA_PERIOD}개 봉 = 50시간)...")

candles_df['ema_200'] = candles_df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
candles_df['above_ema200'] = candles_df['close'] > candles_df['ema_200']

print(f"✅ EMA 200 계산 완료")
print(f"   현재 가격: {candles_df.iloc[-1]['close']:,.2f}")
print(f"   EMA 200: {candles_df.iloc[-1]['ema_200']:,.2f}")

# 실시간 시뮬레이션
trades = []
position = None

recent_lows = []
searching_inflection = False
hl_event = None

ema_filter_count = 0
total_inflection_candidates = 0

print("\n실시간 시뮬레이션 시작 (15분봉 EMA 200 필터 적용)...")

for i in range(EMA_PERIOD + 100, len(candles_df)):
    candle = candles_df.iloc[i]
    prev_candles = candles_df.iloc[max(0, i-20):i]
    
    # Step 1: 저점 감지
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
    
    # Step 2: HL 감지
    if len(recent_lows) >= 2 and not searching_inflection:
        current_low = recent_lows[-1]['price']
        previous_low = recent_lows[-2]['price']
        
        if current_low > previous_low:
            if candle['close'] > current_low * 1.003:
                hl_strength = ((current_low - previous_low) / previous_low) * 100
                
                if hl_strength >= 0.5:
                    hl_event = {
                        'hl_time': recent_lows[-1]['time'],
                        'hl_price': current_low,
                        'hl_strength': hl_strength,
                        'hl_index': recent_lows[-1]['index']
                    }
                    searching_inflection = True
    
    # Step 3: 청산 체크
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
                'above_ema': position['above_ema'],
                'ema_200': position['ema_200']
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
                'above_ema': position['above_ema'],
                'ema_200': position['ema_200']
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
                'above_ema': position['above_ema'],
                'ema_200': position['ema_200']
            })
            position = None
            searching_inflection = False
            continue
    
    # Step 4: 변곡점 캔들 찾기
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
        
        total_inflection_candidates += 1
        
        # EMA 200 필터 (15분봉 기준)
        above_ema = candles_df.iloc[i]['above_ema200']
        ema_value = candles_df.iloc[i]['ema_200']
        
        if not above_ema:
            ema_filter_count += 1
            continue
        
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
            'above_ema': above_ema,
            'ema_200': ema_value
        }
        searching_inflection = False

print(f"\n백테스트 완료!")

# 결과 분석
trades_df = pd.DataFrame(trades)

print(f"\n📊 EMA 200 필터링 통계 (15분봉 기준):")
print(f"  변곡점 캔들 후보: {total_inflection_candidates}개")
print(f"  EMA 200 아래 (차단): {ema_filter_count}개 ({ema_filter_count/total_inflection_candidates*100 if total_inflection_candidates > 0 else 0:.1f}%)")
print(f"  EMA 200 위 (진입): {len(trades_df)}개 ({len(trades_df)/total_inflection_candidates*100 if total_inflection_candidates > 0 else 0:.1f}%)")

if len(trades_df) > 0:
    tp_trades = trades_df[trades_df['exit_reason'].str.contains('TP')]
    sl_trades = trades_df[trades_df['exit_reason'] == 'SL']
    
    win_rate = len(tp_trades) / len(trades_df) * 100
    total_pnl = trades_df['pnl_pct'].sum()
    avg_pnl = trades_df['pnl_pct'].mean()
    
    print("\n" + "="*100)
    print("📈 EMA 200 위에서만 진입 결과 (15분봉 기준)")
    print("="*100)
    print(f"  총 거래: {len(trades_df)}건")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  SL 비율: {len(sl_trades)/len(trades_df)*100:.1f}%")
    print(f"  총 PNL: {total_pnl:.2f}%")
    print(f"  평균 PNL: {avg_pnl:.2f}%")
    print(f"  최대 이익: {trades_df['pnl_pct'].max():.2f}%")
    print(f"  최대 손실: {trades_df['pnl_pct'].min():.2f}%")
    
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
    old_sl_rate = len(old_df[old_df['exit_reason'] == 'SL']) / len(old_df) * 100
    old_avg = old_df['pnl_pct'].mean()
    
    print(f"\n{'지표':<15} {'기존 전략':<20} {'EMA200 필터':<20} {'차이':<15}")
    print("-"*70)
    print(f"{'거래 수':<15} {len(old_df):<20} {len(trades_df):<20} {len(trades_df)-len(old_df):+d}")
    print(f"{'총 PNL':<15} {old_pnl:<20.2f} {total_pnl:<20.2f} {total_pnl-old_pnl:+.2f}%p")
    print(f"{'평균 PNL':<15} {old_avg:<20.2f} {avg_pnl:<20.2f} {avg_pnl-old_avg:+.2f}%p")
    print(f"{'승률':<15} {old_win_rate:<20.1f} {win_rate:<20.1f} {win_rate-old_win_rate:+.1f}%p")
    print(f"{'SL 비율':<15} {old_sl_rate:<20.1f} {len(sl_trades)/len(trades_df)*100:<20.1f} {len(sl_trades)/len(trades_df)*100-old_sl_rate:+.1f}%p")
    
    trades_df.to_csv('backtest_ema200_15m_results.csv', index=False)
    print(f"\n✅ 결과 저장: backtest_ema200_15m_results.csv")
