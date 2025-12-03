import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# === 1. 데이터 로드 ===
print("=" * 60)
print("📊 Final Strategy: Long 90% + Short 10%")
print("=" * 60)

# 롱 신호 로드 및 컬럼명 통합
long_signals = pd.read_csv('valid_signals.csv')
long_signals['signal_type'] = 'LONG'

# 컬럼명 통일: h1/h2 -> p1/p2 (point 1, point 2)
long_signals = long_signals.rename(columns={
    'h1_time': 'p1_time',
    'h1_price': 'p1_price',
    'h2_time': 'p2_time',
    'h2_price': 'p2_price',
    'hl_confirm_time': 'confirm_time',
    'hl_price': 'confirm_price'
})

print(f"\n✅ 롱 신호 로드 완료: {len(long_signals)} 개")

# 숏 신호 생성 (롱 신호의 역전: L-L 상승추세선 하향 돌파 + LH 확인)
analysis = pd.read_csv('analysis_15m.csv')
analysis['datetime'] = pd.to_datetime(analysis['datetime'])

# 숏 신호 조건: swing_low 연속 2개 이상 & 상승추세 돌파
short_signals = []
swing_lows = analysis[analysis['swing_low'] == True].copy()

for i in range(len(swing_lows) - 1):
    l1 = swing_lows.iloc[i]
    l2 = swing_lows.iloc[i + 1]
    
    # L-L 상승추세선 (L1 -> L2가 상승)
    if l2['low'] <= l1['low']:
        continue
    
    # 추세선 하향 돌파 체크
    l1_time = l1['datetime']
    l2_time = l2['datetime']
    l1_price = l1['low']
    l2_price = l2['low']
    
    slope = (l2_price - l1_price) / ((l2_time - l1_time).total_seconds() / 3600)
    
    breakout_candles = analysis[
        (analysis['datetime'] > l2_time) & 
        (analysis['datetime'] < l2_time + timedelta(hours=168))
    ]
    
    for idx, candle in breakout_candles.iterrows():
        hours_since_l2 = (candle['datetime'] - l2_time).total_seconds() / 3600
        trendline_price = l2_price + slope * hours_since_l2
        
        # 하향 돌파 조건
        if candle['close'] < trendline_price and candle['low'] < trendline_price * 0.997:
            gap_pct = ((trendline_price - candle['close']) / trendline_price) * 100
            
            # LH(Lower High) 확인 (돌파 후 5봉 내)
            lh_candles = analysis[
                (analysis['datetime'] > candle['datetime']) &
                (analysis['datetime'] <= candle['datetime'] + timedelta(hours=5))
            ]
            
            lh_found = False
            lh_time = None
            lh_price = None
            
            for lh_idx, lh_candle in lh_candles.iterrows():
                lh_hours = (lh_candle['datetime'] - candle['datetime']).total_seconds() / 3600
                lh_trendline = l2_price + slope * ((lh_candle['datetime'] - l2_time).total_seconds() / 3600)
                
                if lh_candle['high'] > lh_trendline and lh_candle['close'] < lh_trendline:
                    lh_found = True
                    lh_time = lh_candle['datetime']
                    lh_price = lh_candle['high']
                    break
            
            if lh_found:
                short_signals.append({
                    'p1_time': l1_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'p1_price': l1_price,
                    'p2_time': l2_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'p2_price': l2_price,
                    'breakout_time': candle['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                    'breakout_price': candle['close'],
                    'trendline_price': trendline_price,
                    'gap_pct': gap_pct,
                    'confirm_time': lh_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'confirm_price': lh_price,
                    'signal_type': 'SHORT'
                })
            break

short_signals_df = pd.DataFrame(short_signals)
print(f"✅ 숏 신호 생성 완료: {len(short_signals_df)} 개")

# === 2. 비중 조정: 롱 90% + 숏 10% ===
n_long = int(len(long_signals) * 0.90)
n_short = max(int(len(short_signals_df) * 0.10), 1) if len(short_signals_df) > 0 else 0

selected_long = long_signals.head(n_long).copy()
selected_short = short_signals_df.head(n_short).copy() if n_short > 0 else pd.DataFrame()

print(f"\n📌 신호 선택:")
print(f"   롱 신호: {len(selected_long)} / {len(long_signals)} (90%)")
print(f"   숏 신호: {len(selected_short)} / {len(short_signals_df)}")

# === 3. 신호 병합 및 시간순 정렬 ===
if len(selected_short) > 0:
    all_signals = pd.concat([selected_long, selected_short], ignore_index=True)
else:
    all_signals = selected_long.copy()

all_signals['breakout_time'] = pd.to_datetime(all_signals['breakout_time'])
all_signals = all_signals.sort_values('breakout_time').reset_index(drop=True)

print(f"\n✅ 총 신호 수: {len(all_signals)} (시간순 정렬 완료)")

# === 4. 백테스트 실행 ===
print(f"\n{'=' * 60}")
print("🔄 백테스트 시작 (추세선 마감 손절 로직)")
print(f"{'=' * 60}")

results = []
capital = 100.0

for idx, signal in all_signals.iterrows():
    signal_type = signal['signal_type']
    entry_time = signal['breakout_time']
    entry_price = signal['breakout_price']
    
    # 추세선 정보 추출
    p1_time = pd.to_datetime(signal['p1_time'])
    p2_time = pd.to_datetime(signal['p2_time'])
    p1_price = signal['p1_price']
    p2_price = signal['p2_price']
    slope = (p2_price - p1_price) / ((p2_time - p1_time).total_seconds() / 3600)
    
    # 청산 조건 체크
    exit_candles = analysis[
        (analysis['datetime'] > entry_time) &
        (analysis['datetime'] <= entry_time + timedelta(hours=72))
    ]
    
    exit_reason = None
    exit_time = None
    exit_price = None
    max_profit = -999
    
    for ex_idx, candle in exit_candles.iterrows():
        # 롱 포지션
        if signal_type == 'LONG':
            current_profit = ((candle['high'] - entry_price) / entry_price) * 100
            max_profit = max(max_profit, current_profit)
            
            # TP 5%
            if current_profit >= 5.0:
                exit_reason = 'TP'
                exit_time = candle['datetime']
                exit_price = entry_price * 1.05
                break
            
            # SL -2%
            if ((candle['low'] - entry_price) / entry_price) * 100 <= -2.0:
                exit_reason = 'SL'
                exit_time = candle['datetime']
                exit_price = entry_price * 0.98
                break
            
            # 추세선 마감 손절
            hours_since = (candle['datetime'] - p2_time).total_seconds() / 3600
            trendline_price = p2_price + slope * hours_since
            
            if candle['close'] < trendline_price:
                exit_reason = 'TRENDLINE'
                exit_time = candle['datetime']
                exit_price = candle['close']
                break
        
        # 숏 포지션
        else:
            current_profit = ((entry_price - candle['low']) / entry_price) * 100
            max_profit = max(max_profit, current_profit)
            
            # TP 5%
            if current_profit >= 5.0:
                exit_reason = 'TP'
                exit_time = candle['datetime']
                exit_price = entry_price * 0.95
                break
            
            # SL -2%
            if ((entry_price - candle['high']) / entry_price) * 100 <= -2.0:
                exit_reason = 'SL'
                exit_time = candle['datetime']
                exit_price = entry_price * 1.02
                break
            
            # 추세선 마감 손절 (상승추세선 위로 마감)
            hours_since = (candle['datetime'] - p2_time).total_seconds() / 3600
            trendline_price = p2_price + slope * hours_since
            
            if candle['close'] > trendline_price:
                exit_reason = 'TRENDLINE'
                exit_time = candle['datetime']
                exit_price = candle['close']
                break
    
    # 시간 만료
    if exit_reason is None:
        exit_reason = 'TIME'
        exit_time = entry_time + timedelta(hours=72)
        last_candle = exit_candles.iloc[-1] if len(exit_candles) > 0 else None
        exit_price = last_candle['close'] if last_candle is not None else entry_price
    
    # 수익률 계산
    if signal_type == 'LONG':
        profit_pct = ((exit_price - entry_price) / entry_price) * 100
    else:
        profit_pct = ((entry_price - exit_price) / entry_price) * 100
    
    capital *= (1 + profit_pct / 100)
    
    results.append({
        'signal_type': signal_type,
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': exit_time,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'profit_pct': profit_pct,
        'max_profit': max_profit,
        'capital': capital
    })
    
    if (idx + 1) % 50 == 0:
        print(f"진행: {idx + 1}/{len(all_signals)} ({(idx+1)/len(all_signals)*100:.1f}%)")

results_df = pd.DataFrame(results)

# === 5. 결과 출력 ===
print(f"\n{'=' * 60}")
print("📊 최종 결과")
print(f"{'=' * 60}")

long_results = results_df[results_df['signal_type'] == 'LONG']
short_results = results_df[results_df['signal_type'] == 'SHORT']

print(f"\n전체 거래 수: {len(results_df)}")
print(f"롱 거래: {len(long_results)} ({len(long_results)/len(results_df)*100:.1f}%)")
print(f"숏 거래: {len(short_results)} ({len(short_results)/len(results_df)*100:.1f}%)")

win_rate = (results_df['profit_pct'] > 0).sum() / len(results_df) * 100
avg_profit = results_df['profit_pct'].mean()
total_return = ((capital - 100) / 100) * 100

print(f"\n전체 승률: {win_rate:.1f}%")
print(f"전체 평균 수익률: {avg_profit:+.2f}%")
print(f"총 수익: {total_return:+.1f}%")
print(f"최종 자본: {capital:.2f}")

# 롱/숏 개별 분석
if len(long_results) > 0:
    long_win = (long_results['profit_pct'] > 0).sum() / len(long_results) * 100
    long_avg = long_results['profit_pct'].mean()
    print(f"\n롱 승률: {long_win:.1f}% | 평균 수익: {long_avg:+.2f}%")

if len(short_results) > 0:
    short_win = (short_results['profit_pct'] > 0).sum() / len(short_results) * 100
    short_avg = short_results['profit_pct'].mean()
    print(f"숏 승률: {short_win:.1f}% | 평균 수익: {short_avg:+.2f}%")

# 연복리 수익률 계산
start_date = results_df['entry_time'].min()
end_date = results_df['entry_time'].max()
years = (end_date - start_date).days / 365.25
cagr = (pow(capital / 100, 1 / years) - 1) * 100

print(f"\n기간: {start_date.date()} ~ {end_date.date()} ({years:.1f}년)")
print(f"연복리 수익률(CAGR): {cagr:.2f}%")

# 출구 사유 분석
print(f"\n{'=' * 60}")
print("청산 사유 분석")
print(f"{'=' * 60}")
for reason in ['TP', 'TRENDLINE', 'SL', 'TIME']:
    subset = results_df[results_df['exit_reason'] == reason]
    if len(subset) > 0:
        print(f"{reason}: {len(subset)} ({len(subset)/len(results_df)*100:.1f}%) - 평균 수익: {subset['profit_pct'].mean():+.2f}%")

# 결과 저장
results_df.to_csv('final_long90_short10_results.csv', index=False)
print(f"\n✅ 결과 저장 완료: final_long90_short10_results.csv")
