import pandas as pd
import numpy as np
from datetime import timedelta

# 데이터 로드
trades_df = pd.read_csv('backtest_HL_strategy_results.csv')
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
l_values_df = pd.read_csv('all_L_values.csv')

# 타임스탬프 변환
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])
l_values_df['datetime'] = pd.to_datetime(l_values_df['datetime'])

print("="*100)
print("진입 캔들 특징 분석 (힘의 변곡점 확정 지표)")
print("="*100)

# 진입 캔들의 특징 분석
entry_candle_analysis = []

for idx, trade in trades_df.iterrows():
    if idx >= 300:  # 300개 샘플링
        break
    
    entry_time = trade['entry_time']
    
    # 진입 캔들 찾기
    entry_candle_idx = candles_df[candles_df['datetime'] == entry_time].index
    if len(entry_candle_idx) == 0:
        continue
    
    entry_candle_idx = entry_candle_idx[0]
    
    # 진입 캔들 이전 데이터 (10개 캔들)
    if entry_candle_idx < 10:
        continue
    
    prev_candles = candles_df.iloc[entry_candle_idx-10:entry_candle_idx]
    entry_candle = candles_df.iloc[entry_candle_idx]
    
    # HL 시점 찾기 (진입 전 가장 최근 L)
    hl_events = l_values_df[l_values_df['datetime'] <= entry_time].tail(1)
    if len(hl_events) == 0:
        continue
    
    hl_event = hl_events.iloc[0]
    hl_time = hl_event['datetime']
    hl_price = hl_event['L_value']
    
    # HL 이후 진입까지의 캔들 수
    hl_candle_idx = candles_df[candles_df['datetime'] == hl_time].index
    if len(hl_candle_idx) == 0:
        continue
    hl_candle_idx = hl_candle_idx[0]
    
    candles_since_hl = entry_candle_idx - hl_candle_idx
    
    # 진입 캔들의 특징 계산
    
    # 1. 캔들 바디 크기
    body_size = abs(entry_candle['close'] - entry_candle['open'])
    body_pct = (body_size / entry_candle['open']) * 100
    
    # 2. 캔들 방향
    is_green = entry_candle['close'] > entry_candle['open']
    
    # 3. 위꼬리/아래꼬리 비율
    if is_green:
        upper_wick = entry_candle['high'] - entry_candle['close']
        lower_wick = entry_candle['open'] - entry_candle['low']
    else:
        upper_wick = entry_candle['high'] - entry_candle['open']
        lower_wick = entry_candle['close'] - entry_candle['low']
    
    total_range = entry_candle['high'] - entry_candle['low']
    upper_wick_pct = (upper_wick / total_range * 100) if total_range > 0 else 0
    lower_wick_pct = (lower_wick / total_range * 100) if total_range > 0 else 0
    body_to_range = (body_size / total_range * 100) if total_range > 0 else 0
    
    # 4. 거래량 변화
    avg_volume_prev = prev_candles['volume'].mean()
    volume_ratio = entry_candle['volume'] / avg_volume_prev if avg_volume_prev > 0 else 1
    
    # 5. HL 대비 가격 위치
    price_from_hl = ((entry_candle['close'] - hl_price) / hl_price) * 100
    
    # 6. 이전 캔들 대비 변화
    prev_candle = candles_df.iloc[entry_candle_idx - 1]
    price_change = ((entry_candle['close'] - prev_candle['close']) / prev_candle['close']) * 100
    
    # 7. 최근 n개 캔들의 연속 상승/하락
    consecutive_green = 0
    consecutive_red = 0
    for i in range(1, min(6, entry_candle_idx)):
        candle = candles_df.iloc[entry_candle_idx - i]
        if candle['close'] > candle['open']:
            consecutive_green += 1
            break
        else:
            consecutive_red += 1
    
    # 8. HL 이후 최저가 대비 현재가
    candles_after_hl = candles_df.iloc[hl_candle_idx:entry_candle_idx+1]
    lowest_since_hl = candles_after_hl['low'].min()
    recovery_from_low = ((entry_candle['close'] - lowest_since_hl) / lowest_since_hl) * 100
    
    # 9. HL 이후 가격 모멘텀 (선형 회귀 기울기)
    if len(candles_after_hl) >= 2:
        prices = candles_after_hl['close'].values
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        momentum = (slope / prices[0]) * 100  # 정규화
    else:
        momentum = 0
    
    entry_candle_analysis.append({
        'trade_idx': idx,
        'entry_time': entry_time,
        'exit_reason': trade['exit_reason'],
        'pnl_pct': trade['pnl_pct'],
        'hl_strength': trade['HL_strength'],
        'candles_since_hl': candles_since_hl,
        
        # 진입 캔들 특징
        'is_green': is_green,
        'body_pct': body_pct,
        'body_to_range': body_to_range,
        'upper_wick_pct': upper_wick_pct,
        'lower_wick_pct': lower_wick_pct,
        'volume_ratio': volume_ratio,
        'price_change': price_change,
        'price_from_hl': price_from_hl,
        'recovery_from_low': recovery_from_low,
        'momentum': momentum,
        'consecutive_red_before': consecutive_red,
        'consecutive_green_before': consecutive_green,
    })

entry_df = pd.DataFrame(entry_candle_analysis)

# 승리 vs 패배 비교
win_trades = entry_df[entry_df['exit_reason'].str.contains('TP')]
loss_trades = entry_df[entry_df['exit_reason'] == 'SL']

print(f"\n분석 거래 수: {len(entry_df)}")
print(f"  승리: {len(win_trades)} ({len(win_trades)/len(entry_df)*100:.1f}%)")
print(f"  손절: {len(loss_trades)} ({len(loss_trades)/len(entry_df)*100:.1f}%)")

print("\n" + "="*100)
print("1. 진입 캔들 색깔 (힘의 방향)")
print("="*100)
print("\n승리 거래:")
print(f"  초록 캔들: {win_trades['is_green'].sum()} ({win_trades['is_green'].sum()/len(win_trades)*100:.1f}%)")
print(f"  빨강 캔들: {(~win_trades['is_green']).sum()} ({(~win_trades['is_green']).sum()/len(win_trades)*100:.1f}%)")

print("\n손절 거래:")
print(f"  초록 캔들: {loss_trades['is_green'].sum()} ({loss_trades['is_green'].sum()/len(loss_trades)*100:.1f}%)")
print(f"  빨강 캔들: {(~loss_trades['is_green']).sum()} ({(~loss_trades['is_green']).sum()/len(loss_trades)*100:.1f}%)")

print("\n" + "="*100)
print("2. 캔들 바디 크기 (힘의 강도)")
print("="*100)
print(f"\n승리 거래 - 평균 바디 크기: {win_trades['body_pct'].mean():.3f}%")
print(f"손절 거래 - 평균 바디 크기: {loss_trades['body_pct'].mean():.3f}%")
print(f"  차이: {win_trades['body_pct'].mean() - loss_trades['body_pct'].mean():.3f}%p")

print(f"\n승리 거래 - 바디/전체 비율: {win_trades['body_to_range'].mean():.1f}%")
print(f"손절 거래 - 바디/전체 비율: {loss_trades['body_to_range'].mean():.1f}%")
print(f"  차이: {win_trades['body_to_range'].mean() - loss_trades['body_to_range'].mean():.1f}%p")

print("\n" + "="*100)
print("3. 거래량 (힘의 확인)")
print("="*100)
print(f"\n승리 거래 - 평균 거래량 비율: {win_trades['volume_ratio'].mean():.2f}x")
print(f"손절 거래 - 평균 거래량 비율: {loss_trades['volume_ratio'].mean():.2f}x")
print(f"  차이: {win_trades['volume_ratio'].mean() - loss_trades['volume_ratio'].mean():.2f}x")

print("\n" + "="*100)
print("4. 가격 변화 (모멘텀)")
print("="*100)
print(f"\n승리 거래 - 평균 가격 변화: {win_trades['price_change'].mean():.3f}%")
print(f"손절 거래 - 평균 가격 변화: {loss_trades['price_change'].mean():.3f}%")
print(f"  차이: {win_trades['price_change'].mean() - loss_trades['price_change'].mean():.3f}%p")

print(f"\n승리 거래 - HL 대비 가격: {win_trades['price_from_hl'].mean():.2f}%")
print(f"손절 거래 - HL 대비 가격: {loss_trades['price_from_hl'].mean():.2f}%")

print("\n" + "="*100)
print("5. HL 이후 회복력")
print("="*100)
print(f"\n승리 거래 - 최저가 대비 회복: {win_trades['recovery_from_low'].mean():.2f}%")
print(f"손절 거래 - 최저가 대비 회복: {loss_trades['recovery_from_low'].mean():.2f}%")
print(f"  차이: {win_trades['recovery_from_low'].mean() - loss_trades['recovery_from_low'].mean():.2f}%p")

print(f"\n승리 거래 - 모멘텀: {win_trades['momentum'].mean():.4f}")
print(f"손절 거래 - 모멘텀: {loss_trades['momentum'].mean():.4f}")

print("\n" + "="*100)
print("6. HL 이후 대기 시간")
print("="*100)
print(f"\n승리 거래 - 평균 대기 캔들: {win_trades['candles_since_hl'].mean():.1f}개 ({win_trades['candles_since_hl'].mean()*15/60:.1f}시간)")
print(f"손절 거래 - 평균 대기 캔들: {loss_trades['candles_since_hl'].mean():.1f}개 ({loss_trades['candles_since_hl'].mean()*15/60:.1f}시간)")

# 저장
entry_df.to_csv('entry_candle_indicators.csv', index=False)
print(f"\n✅ 분석 결과 저장: entry_candle_indicators.csv")

print("\n" + "="*100)
print("핵심 진입 지표 (변곡점 확정 신호)")
print("="*100)

# 가장 차이가 큰 지표 찾기
print("\n🎯 승리 거래의 특징:")
print(f"  1. 초록 캔들 확률: {win_trades['is_green'].sum()/len(win_trades)*100:.1f}%")
print(f"  2. 바디 크기: {win_trades['body_pct'].mean():.3f}% (강한 캔들)")
print(f"  3. 거래량 증가: {win_trades['volume_ratio'].mean():.2f}배")
print(f"  4. 이전 캔들 대비: +{win_trades['price_change'].mean():.3f}%")
print(f"  5. HL 대비 위치: +{win_trades['price_from_hl'].mean():.2f}%")
print(f"  6. 최저가 대비 회복: +{win_trades['recovery_from_low'].mean():.2f}%")
print(f"  7. 모멘텀: {win_trades['momentum'].mean():.4f}")

print("\n⚠️ 손절 거래의 특징:")
print(f"  1. 초록 캔들 확률: {loss_trades['is_green'].sum()/len(loss_trades)*100:.1f}%")
print(f"  2. 바디 크기: {loss_trades['body_pct'].mean():.3f}% (약한 캔들)")
print(f"  3. 거래량 증가: {loss_trades['volume_ratio'].mean():.2f}배")
print(f"  4. 이전 캔들 대비: +{loss_trades['price_change'].mean():.3f}%")
print(f"  5. HL 대비 위치: +{loss_trades['price_from_hl'].mean():.2f}%")
print(f"  6. 최저가 대비 회복: +{loss_trades['recovery_from_low'].mean():.2f}%")
print(f"  7. 모멘텀: {loss_trades['momentum'].mean():.4f}")

print("\n" + "="*100)
print("🔥 가장 큰 차이를 보이는 지표 (변곡점 확정 신호)")
print("="*100)

# 차이 계산
differences = {
    '초록 캔들 확률': (win_trades['is_green'].sum()/len(win_trades) - loss_trades['is_green'].sum()/len(loss_trades)) * 100,
    '바디 크기': win_trades['body_pct'].mean() - loss_trades['body_pct'].mean(),
    '바디/전체 비율': win_trades['body_to_range'].mean() - loss_trades['body_to_range'].mean(),
    '거래량 비율': win_trades['volume_ratio'].mean() - loss_trades['volume_ratio'].mean(),
    '가격 변화': win_trades['price_change'].mean() - loss_trades['price_change'].mean(),
    'HL 대비 가격': win_trades['price_from_hl'].mean() - loss_trades['price_from_hl'].mean(),
    '최저가 대비 회복': win_trades['recovery_from_low'].mean() - loss_trades['recovery_from_low'].mean(),
    '모멘텀': win_trades['momentum'].mean() - loss_trades['momentum'].mean(),
}

sorted_diff = sorted(differences.items(), key=lambda x: abs(x[1]), reverse=True)

for i, (indicator, diff) in enumerate(sorted_diff, 1):
    print(f"{i}. {indicator}: {diff:+.3f}")

