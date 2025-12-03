import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 80)
print("전략 확장: 롱 + 숏 전체 시그널 분석")
print("=" * 80)

# Load data
candles_df = pd.read_csv('analysis_15m.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])

print(f"\n데이터 기간: {candles_df['datetime'].min()} ~ {candles_df['datetime'].max()}")
print(f"총 캔들: {len(candles_df):,}개")

# MACD 기반 H/L 값 추출 (이미 있는 swing_high, swing_low 활용)
def find_swing_points():
    """Swing High/Low 찾기"""
    H_values = []
    L_values = []
    
    for idx, row in candles_df.iterrows():
        if pd.notna(row['swing_high']):
            H_values.append({
                'time': row['datetime'],
                'price': row['swing_high'],
                'type': 'H'
            })
        
        if pd.notna(row['swing_low']):
            L_values.append({
                'time': row['datetime'],
                'price': row['swing_low'],
                'type': 'L'
            })
    
    return pd.DataFrame(H_values), pd.DataFrame(L_values)

print("\n⏳ Swing High/Low 추출 중...")
H_df, L_df = find_swing_points()

print(f"H 값: {len(H_df)}개")
print(f"L 값: {len(L_df)}개")

def find_trendlines(points_df, direction='down'):
    """추세선 찾기 (H-H 하락선 또는 L-L 상승선)"""
    trendlines = []
    points = points_df.to_dict('records')
    
    for i in range(len(points)):
        for j in range(i+1, min(i+100, len(points))):  # 최대 100개 뒤까지
            p1 = points[i]
            p2 = points[j]
            
            # 추세선 기울기 계산
            time_diff = (p2['time'] - p1['time']).total_seconds() / 3600
            if time_diff < 4:  # 최소 4시간 이상
                continue
            
            price_diff = p2['price'] - p1['price']
            slope = price_diff / time_diff
            
            # 방향 체크
            if direction == 'down' and slope >= 0:  # 하락선은 음의 기울기
                continue
            if direction == 'up' and slope <= 0:  # 상승선은 양의 기울기
                continue
            
            # 중간 캔들이 추세선을 뚫는지 체크
            p1_idx = candles_df[candles_df['datetime'] == p1['time']].index[0]
            p2_idx = candles_df[candles_df['datetime'] == p2['time']].index[0]
            
            between_candles = candles_df.iloc[p1_idx+1:p2_idx]
            
            valid = True
            for idx, candle in between_candles.iterrows():
                candle_hours = (candle['datetime'] - p1['time']).total_seconds() / 3600
                expected_price = p1['price'] + slope * candle_hours
                
                # 허용 오차 0.2%
                tolerance = expected_price * 0.002
                
                if direction == 'down':
                    if candle['high'] > expected_price + tolerance:
                        valid = False
                        break
                else:  # up
                    if candle['low'] < expected_price - tolerance:
                        valid = False
                        break
            
            if valid:
                trendlines.append({
                    'p1_time': p1['time'],
                    'p1_price': p1['price'],
                    'p2_time': p2['time'],
                    'p2_price': p2['price'],
                    'slope': slope,
                    'direction': direction
                })
    
    return trendlines

print("\n⏳ 추세선 찾는 중...")
print("  - H-H 하락 추세선 (롱 진입용)...")
down_trendlines = find_trendlines(H_df, direction='down')
print(f"    → {len(down_trendlines)}개 발견")

print("  - L-L 상승 추세선 (숏 진입용)...")
up_trendlines = find_trendlines(L_df, direction='up')
print(f"    → {len(up_trendlines)}개 발견")

def find_breakout_signals(trendlines, direction='long'):
    """추세선 돌파 + HL/LH 확인"""
    signals = []
    
    for tl in trendlines:
        p2_idx = candles_df[candles_df['datetime'] == tl['p2_time']].index[0]
        
        # 돌파 후 100캔들 확인 (약 25시간)
        future_candles = candles_df.iloc[p2_idx+1:p2_idx+101]
        
        for idx, candle in future_candles.iterrows():
            candle_hours = (candle['datetime'] - tl['p1_time']).total_seconds() / 3600
            trendline_price = tl['p1_price'] + tl['slope'] * candle_hours
            
            # 돌파 체크
            breakout = False
            gap_pct = 0
            
            if direction == 'long':
                # 상향 돌파 (close가 추세선 위)
                if candle['close'] > trendline_price:
                    breakout = True
                    gap_pct = (candle['close'] - trendline_price) / trendline_price * 100
            else:  # short
                # 하향 돌파 (close가 추세선 아래)
                if candle['close'] < trendline_price:
                    breakout = True
                    gap_pct = (trendline_price - candle['close']) / trendline_price * 100
            
            if not breakout:
                continue
            
            # HL/LH 확인 (돌파 후 24시간 내)
            confirm_candles = candles_df.iloc[idx+1:idx+97]  # 24시간
            
            confirmed = False
            confirm_time = None
            confirm_price = None
            
            for c_idx, c_candle in confirm_candles.iterrows():
                if direction == 'long':
                    # HL 확인: 저점이 추세선보다 높게 형성
                    if pd.notna(c_candle['swing_low']) and c_candle['swing_low'] > trendline_price * 0.995:
                        confirmed = True
                        confirm_time = c_candle['datetime']
                        confirm_price = c_candle['swing_low']
                        break
                else:  # short
                    # LH 확인: 고점이 추세선보다 낮게 형성
                    c_hours = (c_candle['datetime'] - tl['p1_time']).total_seconds() / 3600
                    c_trendline = tl['p1_price'] + tl['slope'] * c_hours
                    if pd.notna(c_candle['swing_high']) and c_candle['swing_high'] < c_trendline * 1.005:
                        confirmed = True
                        confirm_time = c_candle['datetime']
                        confirm_price = c_candle['swing_high']
                        break
            
            if confirmed:
                signals.append({
                    'direction': direction,
                    'h1_time': tl['p1_time'],
                    'h1_price': tl['p1_price'],
                    'h2_time': tl['p2_time'],
                    'h2_price': tl['p2_price'],
                    'breakout_time': candle['datetime'],
                    'breakout_price': candle['close'],
                    'trendline_price': trendline_price,
                    'gap_pct': gap_pct,
                    'confirm_time': confirm_time,
                    'confirm_price': confirm_price
                })
                break  # 하나의 추세선당 하나의 신호만
    
    return signals

print("\n⏳ 돌파 시그널 찾는 중...")
print("  - 롱 시그널 (하락선 상향 돌파 + HL 확인)...")
long_signals = find_breakout_signals(down_trendlines, direction='long')
print(f"    → {len(long_signals)}개 발견")

print("  - 숏 시그널 (상승선 하향 돌파 + LH 확인)...")
short_signals = find_breakout_signals(up_trendlines, direction='short')
print(f"    → {len(short_signals)}개 발견")

# 통합
all_signals = long_signals + short_signals
all_signals_df = pd.DataFrame(all_signals)

print("\n" + "=" * 80)
print("시그널 통계")
print("=" * 80)

print(f"\n총 시그널: {len(all_signals)}개")
print(f"  롱: {len(long_signals)}개 ({len(long_signals)/len(all_signals)*100:.1f}%)")
print(f"  숏: {len(short_signals)}개 ({len(short_signals)/len(all_signals)*100:.1f}%)")

years = 5.7
print(f"\n연평균: {len(all_signals)/years:.1f}개 (롱 {len(long_signals)/years:.1f} + 숏 {len(short_signals)/years:.1f})")
print(f"월평균: {len(all_signals)/years/12:.1f}개")

# Gap 분포
print("\n" + "=" * 80)
print("Gap 분포")
print("=" * 80)

for direction in ['long', 'short']:
    subset = all_signals_df[all_signals_df['direction'] == direction]
    if len(subset) > 0:
        print(f"\n{direction.upper()}:")
        print(f"  평균 Gap: {subset['gap_pct'].mean():.3f}%")
        print(f"  중앙값 Gap: {subset['gap_pct'].median():.3f}%")
        print(f"  Gap > 0.5%: {(subset['gap_pct'] > 0.5).sum()}개 ({(subset['gap_pct'] > 0.5).mean()*100:.1f}%)")
        print(f"  Gap > 1.0%: {(subset['gap_pct'] > 1.0).sum()}개 ({(subset['gap_pct'] > 1.0).mean()*100:.1f}%)")

# 저장
all_signals_df.to_csv('all_signals_long_short.csv', index=False)
print("\n✅ 저장: all_signals_long_short.csv")

# 간단한 백테스트
print("\n" + "=" * 80)
print("간단 백테스트 (TP 5%, SL 2%, 72시간)")
print("=" * 80)

def quick_backtest(signal):
    """간단 백테스트"""
    direction = signal['direction']
    entry_time = signal['breakout_time']
    entry_price = signal['breakout_price']
    
    entry_idx = candles_df[candles_df['datetime'] == entry_time].index
    if len(entry_idx) == 0:
        return None
    
    entry_idx = entry_idx[0]
    future = candles_df.iloc[entry_idx+1:entry_idx+289]  # 72시간
    
    if len(future) == 0:
        return None
    
    exit_reason = None
    pnl = 0
    
    for idx, candle in future.iterrows():
        if direction == 'long':
            # TP
            if candle['high'] >= entry_price * 1.05:
                exit_reason = 'TP'
                pnl = 5.0
                break
            # SL
            if candle['low'] <= entry_price * 0.98:
                exit_reason = 'SL'
                pnl = -2.0
                break
        else:  # short
            # TP
            if candle['low'] <= entry_price * 0.95:
                exit_reason = 'TP'
                pnl = 5.0
                break
            # SL
            if candle['high'] >= entry_price * 1.02:
                exit_reason = 'SL'
                pnl = -2.0
                break
    
    if exit_reason is None:
        last_candle = future.iloc[-1]
        exit_reason = 'TIME'
        if direction == 'long':
            pnl = (last_candle['close'] - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - last_candle['close']) / entry_price * 100
    
    return {
        'direction': direction,
        'exit_reason': exit_reason,
        'pnl': pnl
    }

results = []
for idx, signal in all_signals_df.iterrows():
    result = quick_backtest(signal)
    if result:
        results.append(result)

results_df = pd.DataFrame(results)

print(f"\n백테스트 결과: {len(results)}건")

for direction in ['long', 'short']:
    subset = results_df[results_df['direction'] == direction]
    if len(subset) > 0:
        print(f"\n{direction.upper()} ({len(subset)}건):")
        print(f"  평균 수익: {subset['pnl'].mean():.2f}%")
        print(f"  승률: {(subset['pnl'] > 0).mean()*100:.1f}%")
        print(f"  TP: {(subset['exit_reason'] == 'TP').sum()}건 ({(subset['exit_reason'] == 'TP').mean()*100:.1f}%)")
        print(f"  SL: {(subset['exit_reason'] == 'SL').sum()}건 ({(subset['exit_reason'] == 'SL').mean()*100:.1f}%)")
        print(f"  TIME: {(subset['exit_reason'] == 'TIME').sum()}건 ({(subset['exit_reason'] == 'TIME').mean()*100:.1f}%)")

# 전체 통합
print(f"\n전체 통합:")
print(f"  평균 수익: {results_df['pnl'].mean():.2f}%")
print(f"  승률: {(results_df['pnl'] > 0).mean()*100:.1f}%")
print(f"  TP 도달률: {(results_df['exit_reason'] == 'TP').mean()*100:.1f}%")

# 복리 계산
capital = 100
for pnl in results_df['pnl']:
    capital *= (1 + pnl / 100)

total_return = (capital - 100) / 100 * 100
cagr = (capital / 100) ** (1 / years) - 1

print(f"\n복리 시뮬레이션:")
print(f"  초기: 100")
print(f"  최종: {capital:.2f}")
print(f"  총 수익: {total_return:.2f}%")
print(f"  연평균 (CAGR): {cagr*100:.2f}%")

print("\n" + "=" * 80)
print("비교: 롱 Only vs 롱+숏")
print("=" * 80)

print(f"\n롱 Only (기존):")
print(f"  시그널: 371건 (연 65건)")
print(f"  연수익: 45.5%")

print(f"\n롱 + 숏 (확장):")
print(f"  시그널: {len(results)}건 (연 {len(results)/years:.1f}건)")
print(f"  연수익: {cagr*100:.2f}%")
print(f"  개선: +{len(results)-371}건 시그널 ({(len(results)-371)/371*100:.1f}% 증가)")

