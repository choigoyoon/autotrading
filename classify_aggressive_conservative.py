import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 60)
print("📊 공격적 vs 보수적 진입점 분류")
print("=" * 60)

# 데이터 로드
signals = pd.read_csv('valid_signals.csv')
analysis = pd.read_csv('analysis_15m.csv')
analysis['datetime'] = pd.to_datetime(analysis['datetime'])

print(f"\n✅ 총 신호: {len(signals)} 개")

# === 1. 각 신호의 진입 강도 지표 계산 ===
signal_features = []

for idx, signal in signals.iterrows():
    breakout_time = pd.to_datetime(signal['breakout_time'])
    breakout_price = signal['breakout_price']
    gap_pct = signal['gap_pct']
    
    # 진입 시점 캔들 찾기
    entry_candle = analysis[analysis['datetime'] == breakout_time].iloc[0] if len(analysis[analysis['datetime'] == breakout_time]) > 0 else None
    
    if entry_candle is None:
        continue
    
    # 1. Gap 크기 (돌파 강도)
    gap = gap_pct
    
    # 2. 진입 캔들 Body 비율 (강한 돌파인지)
    body_ratio = abs(entry_candle['close'] - entry_candle['open']) / (entry_candle['high'] - entry_candle['low']) if (entry_candle['high'] - entry_candle['low']) > 0 else 0
    
    # 3. 거래량 비율 (직전 20봉 평균 대비)
    prev_candles = analysis[
        (analysis['datetime'] < breakout_time) &
        (analysis['datetime'] >= breakout_time - timedelta(hours=20))
    ]
    avg_volume = prev_candles['volume'].mean() if len(prev_candles) > 0 else entry_candle['volume']
    volume_ratio = entry_candle['volume'] / avg_volume if avg_volume > 0 else 1.0
    
    # 4. 4시간 추세 (진입 전 96봉 = 24시간 추세)
    trend_candles = analysis[
        (analysis['datetime'] < breakout_time) &
        (analysis['datetime'] >= breakout_time - timedelta(hours=24))
    ]
    if len(trend_candles) >= 2:
        trend_pct = ((trend_candles.iloc[-1]['close'] - trend_candles.iloc[0]['close']) / trend_candles.iloc[0]['close']) * 100
    else:
        trend_pct = 0.0
    
    # 5. 강세 캔들 비율 (직전 20봉)
    bullish_candles = prev_candles[prev_candles['close'] > prev_candles['open']]
    bullish_ratio = len(bullish_candles) / len(prev_candles) * 100 if len(prev_candles) > 0 else 50.0
    
    signal_features.append({
        'index': idx,
        'breakout_time': signal['breakout_time'],
        'breakout_price': breakout_price,
        'gap_pct': gap,
        'body_ratio': body_ratio,
        'volume_ratio': volume_ratio,
        'trend_4h': trend_pct,
        'bullish_ratio': bullish_ratio
    })

features_df = pd.DataFrame(signal_features)

print(f"\n✅ 특성 계산 완료: {len(features_df)} 개")

# === 2. 공격적 vs 보수적 기준 정의 ===
print(f"\n{'=' * 60}")
print("📏 진입 강도 지표 분포")
print(f"{'=' * 60}")

print(f"\nGap (돌파 강도):")
print(f"  평균: {features_df['gap_pct'].mean():.3f}%")
print(f"  중앙값: {features_df['gap_pct'].median():.3f}%")
print(f"  75%: {features_df['gap_pct'].quantile(0.75):.3f}%")

print(f"\nBody Ratio (캔들 강도):")
print(f"  평균: {features_df['body_ratio'].mean():.3f}")
print(f"  중앙값: {features_df['body_ratio'].median():.3f}")
print(f"  75%: {features_df['body_ratio'].quantile(0.75):.3f}")

print(f"\nVolume Ratio (거래량 강도):")
print(f"  평균: {features_df['volume_ratio'].mean():.2f}x")
print(f"  중앙값: {features_df['volume_ratio'].median():.2f}x")
print(f"  75%: {features_df['volume_ratio'].quantile(0.75):.2f}x")

print(f"\n4h Trend (선행 추세):")
print(f"  평균: {features_df['trend_4h'].mean():+.2f}%")
print(f"  중앙값: {features_df['trend_4h'].median():+.2f}%")
print(f"  75%: {features_df['trend_4h'].quantile(0.75):+.2f}%")

print(f"\nBullish Ratio (강세 비율):")
print(f"  평균: {features_df['bullish_ratio'].mean():.1f}%")
print(f"  중앙값: {features_df['bullish_ratio'].median():.1f}%")
print(f"  75%: {features_df['bullish_ratio'].quantile(0.75):.1f}%")

# === 3. 분류 기준 ===
print(f"\n{'=' * 60}")
print("🎯 분류 기준 (3가지 시나리오)")
print(f"{'=' * 60}")

# 시나리오 1: 엄격한 기준 (상위 25%)
aggressive_1 = features_df[
    (features_df['gap_pct'] >= features_df['gap_pct'].quantile(0.75)) &
    (features_df['volume_ratio'] >= features_df['volume_ratio'].quantile(0.75)) &
    (features_df['trend_4h'] >= features_df['trend_4h'].quantile(0.75))
]

# 시나리오 2: 중간 기준 (상위 50%)
aggressive_2 = features_df[
    (features_df['gap_pct'] >= features_df['gap_pct'].median()) &
    (features_df['volume_ratio'] >= features_df['volume_ratio'].median()) &
    (features_df['trend_4h'] >= 0)
]

# 시나리오 3: 유연한 기준 (2개 이상 조건 만족)
conditions = [
    features_df['gap_pct'] >= features_df['gap_pct'].quantile(0.75),
    features_df['volume_ratio'] >= features_df['volume_ratio'].quantile(0.75),
    features_df['trend_4h'] >= features_df['trend_4h'].quantile(0.75),
    features_df['body_ratio'] >= features_df['body_ratio'].quantile(0.75),
    features_df['bullish_ratio'] >= features_df['bullish_ratio'].quantile(0.75)
]
condition_count = sum([cond.astype(int) for cond in conditions])
aggressive_3 = features_df[condition_count >= 3]

print(f"\n시나리오 1 (엄격): Gap/Volume/Trend 모두 상위 25%")
print(f"  공격적: {len(aggressive_1)} ({len(aggressive_1)/len(features_df)*100:.1f}%)")
print(f"  보수적: {len(features_df) - len(aggressive_1)} ({(len(features_df)-len(aggressive_1))/len(features_df)*100:.1f}%)")

print(f"\n시나리오 2 (중간): Gap/Volume 중앙값 이상 + Trend 양수")
print(f"  공격적: {len(aggressive_2)} ({len(aggressive_2)/len(features_df)*100:.1f}%)")
print(f"  보수적: {len(features_df) - len(aggressive_2)} ({(len(features_df)-len(aggressive_2))/len(features_df)*100:.1f}%)")

print(f"\n시나리오 3 (유연): 5가지 조건 중 3개 이상 상위 25%")
print(f"  공격적: {len(aggressive_3)} ({len(aggressive_3)/len(features_df)*100:.1f}%)")
print(f"  보수적: {len(features_df) - len(aggressive_3)} ({(len(features_df)-len(aggressive_3))/len(features_df)*100:.1f}%)")

# === 4. 각 시나리오별 백테스트 결과 미리보기 ===
print(f"\n{'=' * 60}")
print("🔍 실제 성과 비교 (백테스트 결과 기반)")
print(f"{'=' * 60}")

# 백테스트 결과 로드
backtest = pd.read_csv('final_backtest_results.csv')
backtest['entry_time'] = pd.to_datetime(backtest['entry_time'])

# 시나리오별 매칭
for scenario_num, aggressive_indices in enumerate([aggressive_1, aggressive_2, aggressive_3], 1):
    aggressive_times = set(aggressive_indices['breakout_time'].values)
    
    aggressive_trades = backtest[backtest['entry_time'].astype(str).isin(aggressive_times)]
    conservative_trades = backtest[~backtest['entry_time'].astype(str).isin(aggressive_times)]
    
    print(f"\n시나리오 {scenario_num}:")
    
    if len(aggressive_trades) > 0:
        agg_win = (aggressive_trades['profit_pct'] > 0).sum() / len(aggressive_trades) * 100
        agg_avg = aggressive_trades['profit_pct'].mean()
        agg_tp = (aggressive_trades['exit_reason'] == 'TP').sum() / len(aggressive_trades) * 100
        print(f"  공격적 ({len(aggressive_trades)}개): 승률 {agg_win:.1f}% | 평균 {agg_avg:+.2f}% | TP 도달 {agg_tp:.1f}%")
    
    if len(conservative_trades) > 0:
        con_win = (conservative_trades['profit_pct'] > 0).sum() / len(conservative_trades) * 100
        con_avg = conservative_trades['profit_pct'].mean()
        con_tp = (conservative_trades['exit_reason'] == 'TP').sum() / len(conservative_trades) * 100
        print(f"  보수적 ({len(conservative_trades)}개): 승률 {con_win:.1f}% | 평균 {con_avg:+.2f}% | TP 도달 {con_tp:.1f}%")

# === 5. 최적 시나리오 선택 및 저장 ===
print(f"\n{'=' * 60}")
print("💡 권장 시나리오")
print(f"{'=' * 60}")

# 시나리오 2를 기본으로 선택 (균형적)
features_df['entry_style'] = 'CONSERVATIVE'
aggressive_times = set(aggressive_2['breakout_time'].values)
features_df.loc[features_df['breakout_time'].isin(aggressive_times), 'entry_style'] = 'AGGRESSIVE'

# 원본 신호에 병합
signals_classified = signals.merge(
    features_df[['breakout_time', 'gap_pct', 'body_ratio', 'volume_ratio', 'trend_4h', 'bullish_ratio', 'entry_style']],
    on='breakout_time',
    suffixes=('', '_calc')
)

# Gap 컬럼 충돌 해결
if 'gap_pct_calc' in signals_classified.columns:
    signals_classified['gap_pct'] = signals_classified['gap_pct_calc']
    signals_classified = signals_classified.drop(columns=['gap_pct_calc'])

signals_classified.to_csv('signals_classified.csv', index=False)

print(f"\n✅ 분류 완료!")
print(f"   파일: signals_classified.csv")
print(f"   공격적: {len(signals_classified[signals_classified['entry_style'] == 'AGGRESSIVE'])} 개")
print(f"   보수적: {len(signals_classified[signals_classified['entry_style'] == 'CONSERVATIVE'])} 개")

print(f"\n{'=' * 60}")
print("📌 분류 기준 (시나리오 2 적용)")
print(f"{'=' * 60}")
print(f"공격적 진입:")
print(f"  ✓ Gap ≥ {features_df['gap_pct'].median():.3f}%")
print(f"  ✓ Volume Ratio ≥ {features_df['volume_ratio'].median():.2f}x")
print(f"  ✓ 4h Trend ≥ 0%")
print(f"\n보수적 진입:")
print(f"  • 위 조건 중 하나라도 미달")
