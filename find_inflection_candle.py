import pandas as pd
import numpy as np

# 데이터 로드
trades_df = pd.read_csv('backtest_HL_strategy_results.csv')
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
l_values_df = pd.read_csv('all_L_values.csv')

# 타임스탬프 변환
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])
l_values_df['datetime'] = pd.to_datetime(l_values_df['datetime'])

print("="*100)
print("힘의 변곡점 캔들 찾기 (HL 이후 ~ 진입 전)")
print("="*100)

inflection_analysis = []

for idx, trade in trades_df.iterrows():
    if idx >= 200:
        break
    
    entry_time = trade['entry_time']
    
    # 진입 캔들 찾기
    entry_candle_idx = candles_df[candles_df['datetime'] == entry_time].index
    if len(entry_candle_idx) == 0:
        continue
    entry_candle_idx = entry_candle_idx[0]
    
    if entry_candle_idx < 20:
        continue
    
    # HL 시점 찾기
    hl_events = l_values_df[l_values_df['datetime'] <= entry_time].tail(1)
    if len(hl_events) == 0:
        continue
    
    hl_event = hl_events.iloc[0]
    hl_time = hl_event['datetime']
    hl_price = hl_event['L_value']
    
    # HL 캔들 찾기
    hl_candle_idx = candles_df[candles_df['datetime'] == hl_time].index
    if len(hl_candle_idx) == 0:
        continue
    hl_candle_idx = hl_candle_idx[0]
    
    # HL 이후 ~ 진입 전 캔들들
    candles_between = candles_df.iloc[hl_candle_idx:entry_candle_idx+1]
    
    if len(candles_between) < 2:
        continue
    
    # 변곡점 캔들 찾기: HL 이후 첫 번째 강한 양봉
    inflection_candle = None
    inflection_idx = None
    
    for i in range(len(candles_between)):
        candle = candles_between.iloc[i]
        
        # 양봉 조건
        if candle['close'] <= candle['open']:
            continue
        
        # 바디 크기
        body_size = candle['close'] - candle['open']
        body_pct = (body_size / candle['open']) * 100
        
        # 전체 범위
        total_range = candle['high'] - candle['low']
        body_to_range = (body_size / total_range * 100) if total_range > 0 else 0
        
        # 강한 양봉 조건: 바디가 전체의 60% 이상, 바디 크기 0.3% 이상
        if body_to_range >= 60 and body_pct >= 0.3:
            inflection_candle = candle
            inflection_idx = i
            break
    
    # 변곡점 캔들이 없으면 스킵
    if inflection_candle is None:
        continue
    
    # 변곡점 캔들의 특징
    inflection_body_pct = (inflection_candle['close'] - inflection_candle['open']) / inflection_candle['open'] * 100
    inflection_total_range = inflection_candle['high'] - inflection_candle['low']
    inflection_body_to_range = ((inflection_candle['close'] - inflection_candle['open']) / inflection_total_range * 100) if inflection_total_range > 0 else 0
    
    # 변곡점 캔들의 거래량
    if inflection_idx >= 5:
        prev_volume_avg = candles_between.iloc[max(0, inflection_idx-5):inflection_idx]['volume'].mean()
    else:
        prev_volume_avg = candles_df.iloc[max(0, hl_candle_idx + inflection_idx - 5):hl_candle_idx + inflection_idx]['volume'].mean()
    
    inflection_volume_ratio = inflection_candle['volume'] / prev_volume_avg if prev_volume_avg > 0 else 1
    
    # 변곡점 캔들 이후의 가격 움직임
    candles_after_inflection = candles_between.iloc[inflection_idx+1:]
    if len(candles_after_inflection) > 0:
        max_price_after = candles_after_inflection['high'].max()
        min_price_after = candles_after_inflection['low'].min()
        
        upside_from_inflection = ((max_price_after - inflection_candle['close']) / inflection_candle['close']) * 100
        downside_from_inflection = ((min_price_after - inflection_candle['close']) / inflection_candle['close']) * 100
    else:
        upside_from_inflection = 0
        downside_from_inflection = 0
    
    # HL 대비 변곡점 캔들 위치
    inflection_price_from_hl = ((inflection_candle['close'] - hl_price) / hl_price) * 100
    
    # 진입 캔들과 변곡점 캔들 간 거리
    candles_from_inflection_to_entry = (entry_candle_idx - (hl_candle_idx + inflection_idx))
    
    inflection_analysis.append({
        'trade_idx': idx,
        'hl_time': hl_time,
        'hl_price': hl_price,
        'inflection_time': inflection_candle['datetime'],
        'inflection_price': inflection_candle['close'],
        'entry_time': entry_time,
        'exit_reason': trade['exit_reason'],
        'pnl_pct': trade['pnl_pct'],
        
        # 변곡점 캔들 특징
        'inflection_body_pct': inflection_body_pct,
        'inflection_body_to_range': inflection_body_to_range,
        'inflection_volume_ratio': inflection_volume_ratio,
        'inflection_price_from_hl': inflection_price_from_hl,
        
        # 변곡점 이후 움직임
        'upside_from_inflection': upside_from_inflection,
        'downside_from_inflection': downside_from_inflection,
        
        # 타이밍
        'candles_hl_to_inflection': inflection_idx,
        'candles_inflection_to_entry': candles_from_inflection_to_entry,
    })

inflection_df = pd.DataFrame(inflection_analysis)

# 승리 vs 손절 비교
win_trades = inflection_df[inflection_df['exit_reason'].str.contains('TP')]
loss_trades = inflection_df[inflection_df['exit_reason'] == 'SL']

print(f"\n분석 거래 수: {len(inflection_df)}")
print(f"  승리: {len(win_trades)} ({len(win_trades)/len(inflection_df)*100:.1f}%)")
print(f"  손절: {len(loss_trades)} ({len(loss_trades)/len(inflection_df)*100:.1f}%)")

print("\n" + "="*100)
print("변곡점 캔들의 특징")
print("="*100)

print("\n🎯 승리 거래:")
print(f"  바디 크기: {win_trades['inflection_body_pct'].mean():.3f}%")
print(f"  바디/전체 비율: {win_trades['inflection_body_to_range'].mean():.1f}%")
print(f"  거래량 비율: {win_trades['inflection_volume_ratio'].mean():.2f}x")
print(f"  HL 대비 위치: {win_trades['inflection_price_from_hl'].mean():.2f}%")
print(f"  변곡점 이후 최대 상승: {win_trades['upside_from_inflection'].mean():.2f}%")
print(f"  변곡점 이후 최대 하락: {win_trades['downside_from_inflection'].mean():.2f}%")

print("\n⚠️ 손절 거래:")
print(f"  바디 크기: {loss_trades['inflection_body_pct'].mean():.3f}%")
print(f"  바디/전체 비율: {loss_trades['inflection_body_to_range'].mean():.1f}%")
print(f"  거래량 비율: {loss_trades['inflection_volume_ratio'].mean():.2f}x")
print(f"  HL 대비 위치: {loss_trades['inflection_price_from_hl'].mean():.2f}%")
print(f"  변곡점 이후 최대 상승: {loss_trades['upside_from_inflection'].mean():.2f}%")
print(f"  변곡점 이후 최대 하락: {loss_trades['downside_from_inflection'].mean():.2f}%")

print("\n" + "="*100)
print("타이밍 분석")
print("="*100)

print("\n🎯 승리 거래:")
print(f"  HL → 변곡점: {win_trades['candles_hl_to_inflection'].mean():.1f}개 캔들 ({win_trades['candles_hl_to_inflection'].mean()*15/60:.1f}시간)")
print(f"  변곡점 → 진입: {win_trades['candles_inflection_to_entry'].mean():.1f}개 캔들 ({win_trades['candles_inflection_to_entry'].mean()*15/60:.1f}시간)")

print("\n⚠️ 손절 거래:")
print(f"  HL → 변곡점: {loss_trades['candles_hl_to_inflection'].mean():.1f}개 캔들 ({loss_trades['candles_hl_to_inflection'].mean()*15/60:.1f}시간)")
print(f"  변곡점 → 진입: {loss_trades['candles_inflection_to_entry'].mean():.1f}개 캔들 ({loss_trades['candles_inflection_to_entry'].mean()*15/60:.1f}시간)")

# 저장
inflection_df.to_csv('inflection_candle_analysis.csv', index=False)
print(f"\n✅ 분석 결과 저장: inflection_candle_analysis.csv")

print("\n" + "="*100)
print("💡 핵심 발견: 변곡점 캔들에서 진입해야 하는가?")
print("="*100)

# 변곡점에서 진입했을 때의 예상 수익률
print("\n만약 변곡점 캔들에서 진입했다면:")
print(f"  승리 거래 - 최대 수익: {win_trades['upside_from_inflection'].mean():.2f}%")
print(f"  승리 거래 - 최대 손실: {win_trades['downside_from_inflection'].mean():.2f}%")
print(f"  손절 거래 - 최대 수익: {loss_trades['upside_from_inflection'].mean():.2f}%")
print(f"  손절 거래 - 최대 손실: {loss_trades['downside_from_inflection'].mean():.2f}%")

print("\n결론:")
if win_trades['upside_from_inflection'].mean() > abs(win_trades['downside_from_inflection'].mean()):
    print("  ✅ 변곡점 캔들에서 진입하는 것이 유리함!")
    print(f"     평균 리스크-리워드: {win_trades['upside_from_inflection'].mean() / abs(win_trades['downside_from_inflection'].mean()):.2f}:1")
else:
    print("  ⚠️ 변곡점 캔들 이후 추가 확인 필요")

