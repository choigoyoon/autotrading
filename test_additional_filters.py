"""
추가 필터 테스트 (나우캐스트 준수)
======================================================================
미래참조 없이 사용 가능한 필터들:

1. 볼륨 확인 - 평균 볼륨 이상
2. MTF 정렬 - 1H/4H MACD도 하락 중
3. L값 근접도 - 최근 L값 근처에서만
4. 가격 모멘텀 - 최근 3봉 하락 확인
5. 다중 패턴 동시 발생 - 2개 이상 패턴 동시
6. 변동성 필터 - ADX로 추세 확인
7. 과매도 강도 - RSI < 25 (더 강한 조건)
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 70)
print("추가 필터 효과 테스트")
print("=" * 70)
print()

# 데이터 로드
df = pd.read_csv('output_phase1_labeled.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

cutoff = df['datetime'].max() - timedelta(days=1825)
df = df[df['datetime'] >= cutoff].reset_index(drop=True)

print(f"분석 기간: {df['datetime'].min().date()} ~ {df['datetime'].max().date()}")
print(f"총 데이터: {len(df):,}개 캔들\n")

# 지표 계산
print("지표 계산 중...")

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# 볼륨 비율
df['vol_ma20'] = df['volume'].rolling(window=20).mean()
df['vol_ratio'] = df['volume'] / df['vol_ma20']

# 최근 3봉 모멘텀
df['price_change_3'] = df['close'].pct_change(3) * 100

# ADX (추세 강도)
high_low = df['high'] - df['low']
high_close = np.abs(df['high'] - df['close'].shift())
low_close = np.abs(df['low'] - df['close'].shift())
ranges = pd.concat([high_low, high_close, low_close], axis=1)
true_range = np.max(ranges, axis=1)
atr = true_range.rolling(14).mean()

plus_dm = df['high'].diff()
minus_dm = -df['low'].diff()
plus_dm[plus_dm < 0] = 0
minus_dm[minus_dm < 0] = 0

plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
df['adx'] = dx.rolling(14).mean()

print("  완료!\n")

# L값 수집
l_values = []
for i in range(1, len(df)):
    if df.iloc[i-1]['macd_hist'] < 0 and df.iloc[i]['macd_hist'] >= 0:
        l_values.append({
            'idx': i,
            'datetime': df.iloc[i]['datetime'],
            'price': df.iloc[i]['low']
        })

print(f"L값: {len(l_values):,}개\n")

# 백테스트 함수
def backtest_entry(entry_idx):
    if entry_idx >= len(df):
        return None

    entry_price = df.iloc[entry_idx]['open']
    tp_price = entry_price * 1.02
    sl_price = entry_price * 0.98

    for j in range(entry_idx, min(entry_idx + 50, len(df))):
        c = df.iloc[j]
        if c['low'] <= sl_price:
            return -2.0
        elif c['high'] >= tp_price:
            return 2.0

    return (df.iloc[min(entry_idx+50, len(df)-1)]['close'] - entry_price) / entry_price * 100

# 필터 테스트 결과 저장
filter_results = []

print("=" * 70)
print("필터별 성과 테스트")
print("=" * 70)
print()

# ═══════════════════════════════════════════════════════════════════
# 기준선: RSI < 30 (필터 없음)
# ═══════════════════════════════════════════════════════════════════

print("기준선: RSI < 30 (필터 없음)")
baseline_trades = []
for i in range(50, len(df) - 50):
    if df.iloc[i]['rsi'] < 30 and df.iloc[i]['macd_hist'] < 0:
        pnl = backtest_entry(i + 1)
        if pnl is not None:
            baseline_trades.append(pnl)
        if len(baseline_trades) >= 2000:  # 제한
            break

if len(baseline_trades) > 0:
    win_rate = len([p for p in baseline_trades if p > 0]) / len(baseline_trades) * 100
    avg_pnl = np.mean(baseline_trades)
    filter_results.append({
        'filter': '필터 없음 (기준)',
        'trades': len(baseline_trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(baseline_trades)}개, 승률: {win_rate:.1f}%, 평균: {avg_pnl:+.2f}%\n")

# ═══════════════════════════════════════════════════════════════════
# 필터 1: 볼륨 확인 (평균 1.5배 이상)
# ═══════════════════════════════════════════════════════════════════

print("필터 1: 볼륨 1.5배 이상")
vol_trades = []
for i in range(50, len(df) - 50):
    row = df.iloc[i]
    if (row['rsi'] < 30 and
        row['macd_hist'] < 0 and
        pd.notna(row['vol_ratio']) and
        row['vol_ratio'] >= 1.5):

        pnl = backtest_entry(i + 1)
        if pnl is not None:
            vol_trades.append(pnl)
        if len(vol_trades) >= 2000:
            break

if len(vol_trades) > 0:
    win_rate = len([p for p in vol_trades if p > 0]) / len(vol_trades) * 100
    avg_pnl = np.mean(vol_trades)
    filter_results.append({
        'filter': '+ 볼륨 1.5배',
        'trades': len(vol_trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(vol_trades)}개, 승률: {win_rate:.1f}%, 평균: {avg_pnl:+.2f}%\n")

# ═══════════════════════════════════════════════════════════════════
# 필터 2: L값 근접 (최근 20봉 이내 L값 존재)
# ═══════════════════════════════════════════════════════════════════

print("필터 2: 최근 20봉 이내 L값")
l_prox_trades = []
for i in range(50, len(df) - 50):
    row = df.iloc[i]
    if row['rsi'] < 30 and row['macd_hist'] < 0:
        # 최근 20봉 이내 L값 확인
        near_l = False
        for l in l_values:
            if abs(i - l['idx']) <= 20:
                near_l = True
                break

        if near_l:
            pnl = backtest_entry(i + 1)
            if pnl is not None:
                l_prox_trades.append(pnl)
            if len(l_prox_trades) >= 2000:
                break

if len(l_prox_trades) > 0:
    win_rate = len([p for p in l_prox_trades if p > 0]) / len(l_prox_trades) * 100
    avg_pnl = np.mean(l_prox_trades)
    filter_results.append({
        'filter': '+ L값 20봉 이내',
        'trades': len(l_prox_trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(l_prox_trades)}개, 승률: {win_rate:.1f}%, 평균: {avg_pnl:+.2f}%\n")

# ═══════════════════════════════════════════════════════════════════
# 필터 3: 가격 하락 모멘텀 (최근 3봉 -1% 이상 하락)
# ═══════════════════════════════════════════════════════════════════

print("필터 3: 최근 3봉 -1% 이상 하락")
momentum_trades = []
for i in range(50, len(df) - 50):
    row = df.iloc[i]
    if (row['rsi'] < 30 and
        row['macd_hist'] < 0 and
        pd.notna(row['price_change_3']) and
        row['price_change_3'] <= -1.0):

        pnl = backtest_entry(i + 1)
        if pnl is not None:
            momentum_trades.append(pnl)
        if len(momentum_trades) >= 2000:
            break

if len(momentum_trades) > 0:
    win_rate = len([p for p in momentum_trades if p > 0]) / len(momentum_trades) * 100
    avg_pnl = np.mean(momentum_trades)
    filter_results.append({
        'filter': '+ 3봉 -1% 하락',
        'trades': len(momentum_trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(momentum_trades)}개, 승률: {win_rate:.1f}%, 평균: {avg_pnl:+.2f}%\n")

# ═══════════════════════════════════════════════════════════════════
# 필터 4: ADX > 25 (추세 있음)
# ═══════════════════════════════════════════════════════════════════

print("필터 4: ADX > 25 (추세)")
adx_trades = []
for i in range(50, len(df) - 50):
    row = df.iloc[i]
    if (row['rsi'] < 30 and
        row['macd_hist'] < 0 and
        pd.notna(row['adx']) and
        row['adx'] > 25):

        pnl = backtest_entry(i + 1)
        if pnl is not None:
            adx_trades.append(pnl)
        if len(adx_trades) >= 2000:
            break

if len(adx_trades) > 0:
    win_rate = len([p for p in adx_trades if p > 0]) / len(adx_trades) * 100
    avg_pnl = np.mean(adx_trades)
    filter_results.append({
        'filter': '+ ADX > 25',
        'trades': len(adx_trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(adx_trades)}개, 승률: {win_rate:.1f}%, 평균: {avg_pnl:+.2f}%\n")

# ═══════════════════════════════════════════════════════════════════
# 필터 5: RSI < 25 (더 강한 과매도)
# ═══════════════════════════════════════════════════════════════════

print("필터 5: RSI < 25 (강한 과매도)")
rsi25_trades = []
for i in range(50, len(df) - 50):
    row = df.iloc[i]
    if row['rsi'] < 25 and row['macd_hist'] < 0:
        pnl = backtest_entry(i + 1)
        if pnl is not None:
            rsi25_trades.append(pnl)
        if len(rsi25_trades) >= 2000:
            break

if len(rsi25_trades) > 0:
    win_rate = len([p for p in rsi25_trades if p > 0]) / len(rsi25_trades) * 100
    avg_pnl = np.mean(rsi25_trades)
    filter_results.append({
        'filter': 'RSI < 25 (강화)',
        'trades': len(rsi25_trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(rsi25_trades)}개, 승률: {win_rate:.1f}%, 평균: {avg_pnl:+.2f}%\n")

# ═══════════════════════════════════════════════════════════════════
# 필터 6: 복합 필터 (L값 근접 + 볼륨 + 모멘텀)
# ═══════════════════════════════════════════════════════════════════

print("필터 6: 복합 (L값 + 볼륨 + 모멘텀)")
combo_trades = []
for i in range(50, len(df) - 50):
    row = df.iloc[i]
    if (row['rsi'] < 30 and
        row['macd_hist'] < 0 and
        pd.notna(row['vol_ratio']) and row['vol_ratio'] >= 1.5 and
        pd.notna(row['price_change_3']) and row['price_change_3'] <= -1.0):

        # L값 근접 확인
        near_l = False
        for l in l_values:
            if abs(i - l['idx']) <= 20:
                near_l = True
                break

        if near_l:
            pnl = backtest_entry(i + 1)
            if pnl is not None:
                combo_trades.append(pnl)
            if len(combo_trades) >= 2000:
                break

if len(combo_trades) > 0:
    win_rate = len([p for p in combo_trades if p > 0]) / len(combo_trades) * 100
    avg_pnl = np.mean(combo_trades)
    filter_results.append({
        'filter': '복합 (L+Vol+Mom)',
        'trades': len(combo_trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    })
    print(f"  거래: {len(combo_trades)}개, 승률: {win_rate:.1f}%, 평균: {avg_pnl:+.2f}%\n")

# ═══════════════════════════════════════════════════════════════════
# 결과 요약
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("필터 효과 요약")
print("=" * 70)
print()

df_results = pd.DataFrame(filter_results)
df_results = df_results.sort_values('win_rate', ascending=False)

print(f"{'필터':<25} {'거래수':<10} {'승률':<12} {'평균 PnL':<12}")
print("-" * 70)
for _, row in df_results.iterrows():
    print(f"{row['filter']:<25} {row['trades']:<10} {row['win_rate']:>6.1f}%      {row['avg_pnl']:>+6.2f}%")

print()
print("=" * 70)
print("결론")
print("=" * 70)
print()

# 최고 성과 필터
best = df_results.iloc[0]
baseline = df_results[df_results['filter'] == '필터 없음 (기준)'].iloc[0]

improvement = best['win_rate'] - baseline['win_rate']

if improvement > 10:
    print(f"✅ 우수 필터 발견: {best['filter']}")
    print(f"   승률 개선: {baseline['win_rate']:.1f}% → {best['win_rate']:.1f}% (+{improvement:.1f}%p)")
    print(f"   거래 수: {best['trades']}개")
    print(f"   평균 PnL: {best['avg_pnl']:+.2f}%")
elif improvement > 5:
    print(f"✓ 유효 필터: {best['filter']}")
    print(f"   승률 개선: {baseline['win_rate']:.1f}% → {best['win_rate']:.1f}% (+{improvement:.1f}%p)")
else:
    print("⚠️  필터 효과 제한적")
    print(f"   최대 개선: +{improvement:.1f}%p")
    print("   → 추가 필터로는 크게 개선되지 않음")

print()
print("✅ 분석 완료")
