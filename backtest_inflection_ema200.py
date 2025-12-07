import pandas as pd
import numpy as np

# 데이터 로드
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])

print("="*100)
print("변곡점 캔들 진입 백테스트 + EMA 200 필터")
print("="*100)

# === EMA 200 계산 (15분봉 기준) ===
# 일봉 기준 EMA 200 = 15분봉 기준 200 * 24 * 4 = 19,200개
# 또는 15분봉 자체 EMA 200 사용

# 일봉 기준으로 계산 (더 의미있음)
EMA_PERIOD = 200 * 24 * 4  # 19,200개 15분봉 = 200일
print(f"\nEMA 200 계산 중 (일봉 기준: {EMA_PERIOD}개 봉)...")

candles_df['ema_200d'] = candles_df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
candles_df['above_ema200'] = candles_df['close'] > candles_df['ema_200d']

print(f"✅ EMA 200 계산 완료")

# 실시간 시뮬레이션
trades = []
position = None

recent_lows = []
searching_inflection = False
hl_event = None

ema_filter_count = 0
total_inflection_candidates = 0

print("\n실시간 시뮬레이션 시작 (EMA 200 필터 적용)...")

for i in range(100, len(candles_df)):
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
                'above_ema': position['above_ema']
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
                'above_ema': position['above_ema']
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
                'above_ema': position['above_ema']
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
        
        # EMA 200 필터
        above_ema = candles_df.iloc[i]['above_ema200']
        
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
            'above_ema': above_ema
        }
        searching_inflection = False

print(f"\n백테스트 완료!")

# 결과 분석
trades_df = pd.DataFrame(trades)

print(f"\n📊 EMA 200 필터링 통계:")
print(f"  변곡점 캔들 후보: {total_inflection_candidates}개")
print(f"  EMA 200 아래 (차단): {ema_filter_count}개 ({ema_filter_count/total_inflection_candidates*100 if total_inflection_candidates > 0 else 0:.1f}%)")
print(f"  EMA 200 위 (진입): {len(trades_df)}개")

if len(trades_df) > 0:
    tp_trades = trades_df[trades_df['exit_reason'].str.contains('TP')]
    sl_trades = trades_df[trades_df['exit_reason'] == 'SL']
    
    win_rate = len(tp_trades) / len(trades_df) * 100
    total_pnl = trades_df['pnl_pct'].sum()
    avg_pnl = trades_df['pnl_pct'].mean()
    
    print("\n" + "="*100)
    print("📈 EMA 200 위에서만 진입 결과")
    print("="*100)
    print(f"  총 거래: {len(trades_df)}건")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  SL 비율: {len(sl_trades)/len(trades_df)*100:.1f}%")
    print(f"  총 PNL: {total_pnl:.2f}%")
    print(f"  평균 PNL: {avg_pnl:.2f}%")
    
    # 연도별
    trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
    print("\n연도별:")
    for year in sorted(trades_df['year'].unique()):
        yt = trades_df[trades_df['year'] == year]
        print(f"  {year}: {len(yt)}건, PNL {yt['pnl_pct'].sum():.2f}%")
    
    # 기존 전략과 비교
    print("\n" + "="*100)
    print("📊 전략 비교")
    print("="*100)
    
    old_df = pd.read_csv('backtest_inflection_no_lookahead_results.csv')
    old_pnl = old_df['pnl_pct'].sum()
    old_win_rate = len(old_df[old_df['exit_reason'].str.contains('TP')]) / len(old_df) * 100
    
    print(f"\n기존 (필터 없음): {len(old_df)}건, PNL {old_pnl:.2f}%, 승률 {old_win_rate:.1f}%")
    print(f"EMA 200 필터:    {len(trades_df)}건, PNL {total_pnl:.2f}%, 승률 {win_rate:.1f}%")
    print(f"\n차이: 거래 {len(trades_df)-len(old_df):+d}건, PNL {total_pnl-old_pnl:+.2f}%p, 승률 {win_rate-old_win_rate:+.1f}%p")
    
    trades_df.to_csv('backtest_ema200_results.csv', index=False)
