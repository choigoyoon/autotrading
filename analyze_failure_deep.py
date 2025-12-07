import pandas as pd
import numpy as np

"""
손실 거래 실패 사유 정밀 분석
- 진입 후 왜 손절됐는지?
- 어떤 상황에서 SL이 터졌는지?
"""

# 데이터 로드
trades_df = pd.read_csv('backtest_ema_sl_results.csv')
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])
trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

print("="*100)
print("🔍 손실 거래 실패 사유 정밀 분석")
print("="*100)

# SL 거래만 추출
sl_trades = trades_df[trades_df['exit_reason'].str.contains('SL')].copy()
print(f"\n총 거래: {len(trades_df)}건")
print(f"손절(SL) 거래: {len(sl_trades)}건 ({len(sl_trades)/len(trades_df)*100:.1f}%)")

# 캔들 데이터에 인덱스 설정
candles_df.set_index('datetime', inplace=True)

# 각 SL 거래의 진입~청산 구간 분석
failure_reasons = []

for idx, trade in sl_trades.iterrows():
    entry_time = trade['entry_time']
    exit_time = trade['exit_time']
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']
    pnl = trade['pnl_pct']
    
    try:
        # 진입~청산 구간 캔들
        trade_candles = candles_df[(candles_df.index >= entry_time) & (candles_df.index <= exit_time)]
        
        if len(trade_candles) < 2:
            continue
        
        # 진입 전 상황 (직전 20봉)
        entry_idx = candles_df.index.get_indexer([entry_time], method='nearest')[0]
        pre_entry = candles_df.iloc[max(0, entry_idx-20):entry_idx]
        
        # 진입 시점 캔들
        entry_candle = candles_df.iloc[entry_idx]
        
        # === 실패 사유 분석 ===
        reasons = []
        
        # 1. 진입 직후 급락 (첫 3봉 내 1% 이상 하락)
        first_3 = trade_candles.head(3)
        if len(first_3) >= 1:
            min_after_entry = first_3['low'].min()
            drop_after_entry = ((min_after_entry - entry_price) / entry_price) * 100
            if drop_after_entry < -1:
                reasons.append('진입직후급락')
        
        # 2. 추세 역행 진입 (진입 전 하락 추세)
        if len(pre_entry) >= 10:
            pre_trend = (pre_entry['close'].iloc[-1] - pre_entry['close'].iloc[0]) / pre_entry['close'].iloc[0] * 100
            if pre_trend < -1:
                reasons.append('하락추세역행진입')
        
        # 3. 가짜 반등 (HL 후 재하락)
        # 진입가 대비 최고점 도달 후 다시 하락
        max_price = trade_candles['high'].max()
        max_profit = ((max_price - entry_price) / entry_price) * 100
        if max_profit > 0.5 and pnl < -0.5:
            reasons.append('가짜반등(익절실패)')
        
        # 4. 고점 추격 진입
        if len(pre_entry) >= 20:
            recent_high = pre_entry['high'].max()
            if entry_price > recent_high * 0.99:  # 고점 1% 이내
                reasons.append('고점추격')
        
        # 5. 변동성 급증 구간
        pre_atr = pre_entry['high'].rolling(5).max() - pre_entry['low'].rolling(5).min()
        if len(pre_atr) > 0:
            avg_range = pre_atr.mean()
            trade_range = trade_candles['high'].max() - trade_candles['low'].min()
            if trade_range > avg_range * 2:
                reasons.append('변동성급증')
        
        # 6. 장기 하락 구간 (24시간 기준 하락 추세)
        if entry_idx >= 96:
            day_ago = candles_df.iloc[entry_idx - 96]['close']
            day_change = ((entry_price - day_ago) / day_ago) * 100
            if day_change < -5:
                reasons.append('장기하락구간')
        
        # 7. 저항선 근처 진입
        if len(pre_entry) >= 50:
            resistance = pre_entry['high'].rolling(20).max().iloc[-1]
            if entry_price > resistance * 0.995:
                reasons.append('저항선근처')
        
        # 8. 지지선 붕괴
        if len(pre_entry) >= 20:
            support = pre_entry['low'].rolling(10).min().iloc[-1]
            if trade_candles['low'].min() < support:
                reasons.append('지지선붕괴')
        
        # 9. 연속 하락 캔들 중 진입
        if len(pre_entry) >= 5:
            red_count = (pre_entry['close'].tail(5) < pre_entry['open'].tail(5)).sum()
            if red_count >= 4:
                reasons.append('연속하락중진입')
        
        # 10. 거래량 감소 중 진입 (반등 신뢰도 낮음)
        if len(pre_entry) >= 10:
            recent_vol = pre_entry['volume'].tail(5).mean()
            prev_vol = pre_entry['volume'].head(5).mean()
            if recent_vol < prev_vol * 0.5:
                reasons.append('거래량감소중')
        
        # 사유 없으면 '기타'
        if not reasons:
            reasons.append('기타')
        
        failure_reasons.append({
            'entry_time': entry_time,
            'pnl': pnl,
            'reasons': '|'.join(reasons),
            'reason_count': len(reasons),
            'drop_after_entry': drop_after_entry if 'drop_after_entry' in dir() else 0,
            'max_profit': max_profit if 'max_profit' in dir() else 0,
            'hold_candles': len(trade_candles)
        })
        
    except Exception as e:
        continue

reasons_df = pd.DataFrame(failure_reasons)
print(f"\n분석된 SL 거래: {len(reasons_df)}건")

# 실패 사유별 통계
print("\n" + "="*100)
print("📊 실패 사유별 통계")
print("="*100)

# 각 사유 카운트
reason_counts = {}
for _, row in reasons_df.iterrows():
    for reason in row['reasons'].split('|'):
        if reason not in reason_counts:
            reason_counts[reason] = {'count': 0, 'pnl_sum': 0}
        reason_counts[reason]['count'] += 1
        reason_counts[reason]['pnl_sum'] += row['pnl']

reason_stats = []
for reason, stats in reason_counts.items():
    reason_stats.append({
        'reason': reason,
        'count': stats['count'],
        'pct': stats['count'] / len(reasons_df) * 100,
        'avg_loss': stats['pnl_sum'] / stats['count']
    })

reason_stats_df = pd.DataFrame(reason_stats).sort_values('count', ascending=False)

print(f"\n{'실패 사유':<20} {'건수':>8} {'비율':>8} {'평균손실':>10}")
print("-"*50)
for _, r in reason_stats_df.iterrows():
    print(f"{r['reason']:<20} {r['count']:>8} {r['pct']:>7.1f}% {r['avg_loss']:>9.2f}%")

# 복합 사유 분석
print("\n" + "="*100)
print("📊 복합 실패 사유 (2개 이상)")
print("="*100)

multi_reasons = reasons_df[reasons_df['reason_count'] >= 2]
print(f"\n복합 사유 거래: {len(multi_reasons)}건 ({len(multi_reasons)/len(reasons_df)*100:.1f}%)")

combo_counts = multi_reasons['reasons'].value_counts().head(10)
print("\n주요 복합 사유 조합:")
for combo, count in combo_counts.items():
    avg_loss = multi_reasons[multi_reasons['reasons'] == combo]['pnl'].mean()
    print(f"  {combo}: {count}건, 평균 {avg_loss:.2f}%")

# 최악의 손실 거래 분석
print("\n" + "="*100)
print("🔴 최악의 손실 거래 TOP 10")
print("="*100)

worst = reasons_df.nsmallest(10, 'pnl')
for _, w in worst.iterrows():
    print(f"\n{w['entry_time']}: {w['pnl']:.2f}%")
    print(f"  사유: {w['reasons']}")
    print(f"  진입 후 최대 하락: {w['drop_after_entry']:.2f}%, 최대 이익: {w['max_profit']:.2f}%")

# 핵심 인사이트
print("\n" + "="*100)
print("💡 핵심 인사이트")
print("="*100)

top_reasons = reason_stats_df.head(5)
print("\n가장 빈번한 실패 사유:")
for i, (_, r) in enumerate(top_reasons.iterrows(), 1):
    print(f"  {i}. {r['reason']}: {r['count']}건 ({r['pct']:.1f}%)")

# 실패 사유별 필터 효과 예측
print("\n" + "="*100)
print("📈 실패 사유 회피 시 예상 효과")
print("="*100)

# 전체 거래 로드
all_trades = pd.read_csv('trade_loss_reasons_analysis.csv')
total = len(all_trades)
total_pnl = all_trades['pnl'].sum()
win_rate = (1 - all_trades['is_loss'].sum() / total) * 100

for _, r in reason_stats_df.head(5).iterrows():
    avoided_loss = -r['avg_loss'] * r['count']  # 회피 시 막을 수 있는 손실
    new_pnl = total_pnl + avoided_loss
    # 이 사유의 손실 거래를 제외하면
    new_trades = total - r['count']
    new_losses = all_trades['is_loss'].sum() - r['count']
    new_win_rate = (1 - new_losses / new_trades) * 100 if new_trades > 0 else 0
    
    print(f"\n'{r['reason']}' 회피 시:")
    print(f"  거래: {total} → {new_trades}건 (-{r['count']})")
    print(f"  승률: {win_rate:.1f}% → {new_win_rate:.1f}%")
    print(f"  예상 PNL 개선: +{avoided_loss:.1f}%")

# 저장
reasons_df.to_csv('sl_failure_reasons.csv', index=False)
print(f"\n✅ 결과 저장: sl_failure_reasons.csv")
