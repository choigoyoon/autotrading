import pandas as pd
import numpy as np

# 데이터 로드
trades_df = pd.read_csv('backtest_inflection_no_lookahead_results.csv')
candles_15m = pd.read_csv('btc_15m_ohlcv.csv')
candles_1h = pd.read_csv('btc_1h_ohlcv.csv')
candles_4h = pd.read_csv('btc_4h_ohlcv.csv')

# 타임스탬프 변환
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
candles_15m['datetime'] = pd.to_datetime(candles_15m['datetime'])
candles_1h['datetime'] = pd.to_datetime(candles_1h['datetime'])
candles_4h['datetime'] = pd.to_datetime(candles_4h['datetime'])

print("="*120)
print("개별 거래 상세 분석 (진입 전/후 상황, MTF 분석)")
print("="*120)

# 승리/손실 케이스 선택
win_trades = trades_df[trades_df['exit_reason'].str.contains('TP')].head(10)
loss_trades = trades_df[trades_df['exit_reason'] == 'SL'].head(10)

def analyze_trade_context(trade, candles_15m, candles_1h, candles_4h):
    """개별 거래의 진입 전/후 상황 및 MTF 분석"""
    
    entry_time = trade['entry_time']
    entry_price = trade['entry_price']
    exit_time = trade['exit_time']
    
    # 진입 캔들 찾기
    entry_candle_idx = candles_15m[candles_15m['datetime'] == entry_time].index
    if len(entry_candle_idx) == 0:
        return None
    entry_candle_idx = entry_candle_idx[0]
    
    # 진입 전 20개 캔들 (5시간)
    before_candles = candles_15m.iloc[max(0, entry_candle_idx-20):entry_candle_idx]
    
    # 진입 후 청산까지 캔들
    exit_candle_idx = candles_15m[candles_15m['datetime'] <= exit_time].index
    if len(exit_candle_idx) == 0:
        exit_candle_idx = entry_candle_idx + 20
    else:
        exit_candle_idx = exit_candle_idx[-1]
    
    after_candles = candles_15m.iloc[entry_candle_idx:exit_candle_idx+1]
    
    # === 진입 전 상황 분석 ===
    
    # 1. 가격 추세 (진입 전 20개 캔들)
    price_change_before = ((before_candles.iloc[-1]['close'] - before_candles.iloc[0]['close']) / before_candles.iloc[0]['close']) * 100
    
    # 2. 최근 하락폭
    recent_high = before_candles['high'].max()
    recent_low = before_candles['low'].min()
    drawdown = ((recent_low - recent_high) / recent_high) * 100
    
    # 3. 연속 하락/상승 캔들
    consecutive_red = 0
    consecutive_green = 0
    for i in range(len(before_candles)-1, -1, -1):
        candle = before_candles.iloc[i]
        if candle['close'] < candle['open']:
            consecutive_red += 1
        else:
            break
    
    for i in range(len(before_candles)-1, -1, -1):
        candle = before_candles.iloc[i]
        if candle['close'] > candle['open']:
            consecutive_green += 1
        else:
            break
    
    # 4. 볼륨 추세
    avg_volume_before = before_candles['volume'].mean()
    recent_volume = before_candles.iloc[-5:]['volume'].mean()
    volume_trend = ((recent_volume - avg_volume_before) / avg_volume_before) * 100
    
    # === MTF (Multi-TimeFrame) 분석 ===
    
    # 1H 타임프레임
    h1_candle = candles_1h[candles_1h['datetime'] <= entry_time].tail(1)
    if len(h1_candle) > 0:
        h1_candle = h1_candle.iloc[0]
        h1_prev_candles = candles_1h[candles_1h['datetime'] < entry_time].tail(10)
        
        # 1H 추세
        if len(h1_prev_candles) >= 5:
            h1_ma5 = h1_prev_candles['close'].tail(5).mean()
            h1_trend = "상승" if h1_candle['close'] > h1_ma5 else "하락"
            h1_trend_strength = ((h1_candle['close'] - h1_ma5) / h1_ma5) * 100
        else:
            h1_trend = "알수없음"
            h1_trend_strength = 0
        
        # 1H 캔들 방향
        h1_candle_direction = "양봉" if h1_candle['close'] > h1_candle['open'] else "음봉"
        h1_body_pct = ((h1_candle['close'] - h1_candle['open']) / h1_candle['open']) * 100
    else:
        h1_trend = "데이터없음"
        h1_trend_strength = 0
        h1_candle_direction = "알수없음"
        h1_body_pct = 0
    
    # 4H 타임프레임
    h4_candle = candles_4h[candles_4h['datetime'] <= entry_time].tail(1)
    if len(h4_candle) > 0:
        h4_candle = h4_candle.iloc[0]
        h4_prev_candles = candles_4h[candles_4h['datetime'] < entry_time].tail(10)
        
        # 4H 추세
        if len(h4_prev_candles) >= 5:
            h4_ma5 = h4_prev_candles['close'].tail(5).mean()
            h4_trend = "상승" if h4_candle['close'] > h4_ma5 else "하락"
            h4_trend_strength = ((h4_candle['close'] - h4_ma5) / h4_ma5) * 100
        else:
            h4_trend = "알수없음"
            h4_trend_strength = 0
        
        # 4H 캔들 방향
        h4_candle_direction = "양봉" if h4_candle['close'] > h4_candle['open'] else "음봉"
        h4_body_pct = ((h4_candle['close'] - h4_candle['open']) / h4_candle['open']) * 100
    else:
        h4_trend = "데이터없음"
        h4_trend_strength = 0
        h4_candle_direction = "알수없음"
        h4_body_pct = 0
    
    # === 진입 후 상황 분석 ===
    
    # 최고/최저 도달
    max_price_after = after_candles['high'].max()
    min_price_after = after_candles['low'].min()
    
    max_profit = ((max_price_after - entry_price) / entry_price) * 100
    max_loss = ((min_price_after - entry_price) / entry_price) * 100
    
    # 진입 후 즉시 하락 여부
    first_5_candles = after_candles.head(5)
    immediate_drop = ((first_5_candles['low'].min() - entry_price) / entry_price) * 100
    
    return {
        'trade_info': {
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_time': exit_time,
            'exit_price': trade['exit_price'],
            'exit_reason': trade['exit_reason'],
            'pnl_pct': trade['pnl_pct'],
            'hl_strength': trade['hl_strength']
        },
        'before_entry': {
            'price_change': price_change_before,
            'drawdown': drawdown,
            'consecutive_red': consecutive_red,
            'consecutive_green': consecutive_green,
            'volume_trend': volume_trend
        },
        'mtf': {
            '1h_trend': h1_trend,
            '1h_trend_strength': h1_trend_strength,
            '1h_candle_direction': h1_candle_direction,
            '1h_body_pct': h1_body_pct,
            '4h_trend': h4_trend,
            '4h_trend_strength': h4_trend_strength,
            '4h_candle_direction': h4_candle_direction,
            '4h_body_pct': h4_body_pct
        },
        'after_entry': {
            'max_profit': max_profit,
            'max_loss': max_loss,
            'immediate_drop': immediate_drop
        }
    }

# 승리 케이스 분석
print("\n" + "="*120)
print("✅ 승리 케이스 10건 분석")
print("="*120)

win_analysis = []
for idx, trade in win_trades.iterrows():
    analysis = analyze_trade_context(trade, candles_15m, candles_1h, candles_4h)
    if analysis:
        win_analysis.append(analysis)

for i, analysis in enumerate(win_analysis, 1):
    print(f"\n{'='*120}")
    print(f"승리 케이스 #{i}")
    print(f"{'='*120}")
    
    info = analysis['trade_info']
    before = analysis['before_entry']
    mtf = analysis['mtf']
    after = analysis['after_entry']
    
    print(f"\n📊 거래 정보:")
    print(f"  진입: {info['entry_time']} @ ${info['entry_price']:.2f}")
    print(f"  청산: {info['exit_time']} @ ${info['exit_price']:.2f}")
    print(f"  결과: {info['exit_reason']} | PNL: {info['pnl_pct']:.2f}%")
    print(f"  HL 강도: {info['hl_strength']:.2f}%")
    
    print(f"\n🔍 진입 전 상황 (15분봉 기준):")
    print(f"  가격 변화 (최근 5시간): {before['price_change']:+.2f}%")
    print(f"  최대 하락폭: {before['drawdown']:.2f}%")
    print(f"  연속 빨강 캔들: {before['consecutive_red']}개")
    print(f"  연속 초록 캔들: {before['consecutive_green']}개")
    print(f"  거래량 추세: {before['volume_trend']:+.1f}%")
    
    print(f"\n🎯 MTF 분석:")
    print(f"  1시간봉:")
    print(f"    추세: {mtf['1h_trend']} ({mtf['1h_trend_strength']:+.2f}%)")
    print(f"    현재 캔들: {mtf['1h_candle_direction']} ({mtf['1h_body_pct']:+.2f}%)")
    print(f"  4시간봉:")
    print(f"    추세: {mtf['4h_trend']} ({mtf['4h_trend_strength']:+.2f}%)")
    print(f"    현재 캔들: {mtf['4h_candle_direction']} ({mtf['4h_body_pct']:+.2f}%)")
    
    print(f"\n📈 진입 후 움직임:")
    print(f"  최대 수익: {after['max_profit']:+.2f}%")
    print(f"  최대 손실: {after['max_loss']:+.2f}%")
    print(f"  진입 직후 하락: {after['immediate_drop']:+.2f}%")

# 손실 케이스 분석
print("\n\n" + "="*120)
print("❌ 손실 케이스 10건 분석")
print("="*120)

loss_analysis = []
for idx, trade in loss_trades.iterrows():
    analysis = analyze_trade_context(trade, candles_15m, candles_1h, candles_4h)
    if analysis:
        loss_analysis.append(analysis)

for i, analysis in enumerate(loss_analysis, 1):
    print(f"\n{'='*120}")
    print(f"손실 케이스 #{i}")
    print(f"{'='*120}")
    
    info = analysis['trade_info']
    before = analysis['before_entry']
    mtf = analysis['mtf']
    after = analysis['after_entry']
    
    print(f"\n📊 거래 정보:")
    print(f"  진입: {info['entry_time']} @ ${info['entry_price']:.2f}")
    print(f"  청산: {info['exit_time']} @ ${info['exit_price']:.2f}")
    print(f"  결과: {info['exit_reason']} | PNL: {info['pnl_pct']:.2f}%")
    print(f"  HL 강도: {info['hl_strength']:.2f}%")
    
    print(f"\n🔍 진입 전 상황 (15분봉 기준):")
    print(f"  가격 변화 (최근 5시간): {before['price_change']:+.2f}%")
    print(f"  최대 하락폭: {before['drawdown']:.2f}%")
    print(f"  연속 빨강 캔들: {before['consecutive_red']}개")
    print(f"  연속 초록 캔들: {before['consecutive_green']}개")
    print(f"  거래량 추세: {before['volume_trend']:+.1f}%")
    
    print(f"\n🎯 MTF 분석:")
    print(f"  1시간봉:")
    print(f"    추세: {mtf['1h_trend']} ({mtf['1h_trend_strength']:+.2f}%)")
    print(f"    현재 캔들: {mtf['1h_candle_direction']} ({mtf['1h_body_pct']:+.2f}%)")
    print(f"  4시간봉:")
    print(f"    추세: {mtf['4h_trend']} ({mtf['4h_trend_strength']:+.2f}%)")
    print(f"    현재 캔들: {mtf['4h_candle_direction']} ({mtf['4h_body_pct']:+.2f}%)")
    
    print(f"\n📈 진입 후 움직임:")
    print(f"  최대 수익: {after['max_profit']:+.2f}%")
    print(f"  최대 손실: {after['max_loss']:+.2f}%")
    print(f"  진입 직후 하락: {after['immediate_drop']:+.2f}%")

# 승리 vs 손실 패턴 비교
print("\n\n" + "="*120)
print("🔬 승리 vs 손실 패턴 비교")
print("="*120)

# 평균 계산
win_df = pd.DataFrame([a['before_entry'] for a in win_analysis])
win_mtf_df = pd.DataFrame([a['mtf'] for a in win_analysis])
win_after_df = pd.DataFrame([a['after_entry'] for a in win_analysis])

loss_df = pd.DataFrame([a['before_entry'] for a in loss_analysis])
loss_mtf_df = pd.DataFrame([a['mtf'] for a in loss_analysis])
loss_after_df = pd.DataFrame([a['after_entry'] for a in loss_analysis])

print("\n진입 전 상황:")
print(f"  가격 변화: 승리 {win_df['price_change'].mean():+.2f}% vs 손실 {loss_df['price_change'].mean():+.2f}%")
print(f"  최대 하락폭: 승리 {win_df['drawdown'].mean():.2f}% vs 손실 {loss_df['drawdown'].mean():.2f}%")
print(f"  거래량 추세: 승리 {win_df['volume_trend'].mean():+.1f}% vs 손실 {loss_df['volume_trend'].mean():+.1f}%")

print("\nMTF 추세:")
print(f"  1H 추세 강도: 승리 {win_mtf_df['1h_trend_strength'].mean():+.2f}% vs 손실 {loss_mtf_df['1h_trend_strength'].mean():+.2f}%")
print(f"  4H 추세 강도: 승리 {win_mtf_df['4h_trend_strength'].mean():+.2f}% vs 손실 {loss_mtf_df['4h_trend_strength'].mean():+.2f}%")

# 1H/4H 추세 방향 통계
print("\n1H 추세 방향:")
print(f"  승리: {win_mtf_df['1h_trend'].value_counts().to_dict()}")
print(f"  손실: {loss_mtf_df['1h_trend'].value_counts().to_dict()}")

print("\n4H 추세 방향:")
print(f"  승리: {win_mtf_df['4h_trend'].value_counts().to_dict()}")
print(f"  손실: {loss_mtf_df['4h_trend'].value_counts().to_dict()}")

print("\n진입 후 즉시 하락:")
print(f"  승리: {win_after_df['immediate_drop'].mean():+.2f}%")
print(f"  손실: {loss_after_df['immediate_drop'].mean():+.2f}%")

