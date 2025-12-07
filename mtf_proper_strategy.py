#!/usr/bin/env python3
"""
MTF 전략 - 제대로 된 버전
- 일봉: 전체 추세 판단
- 4시간: 실제 매매 진입
- 1시간: 방향 확인
- 15분: 세부 타이밍

목표: 월 2~5회 거래, 회당 3~5%
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("MTF 전략 - 4시간봉 매매, 상위 시간대 필터")
print("=" * 80)

# 데이터 로드
df_1d = pd.read_csv('analysis_1d.csv')
df_4h = pd.read_csv('analysis_4h.csv')
df_1h = pd.read_csv('analysis_1h.csv')
df_15m = pd.read_csv('analysis_15m.csv')

for df in [df_1d, df_4h, df_1h, df_15m]:
    df['datetime'] = pd.to_datetime(df['datetime'])

print(f"일봉: {len(df_1d):,}개")
print(f"4시간: {len(df_4h):,}개")
print(f"1시간: {len(df_1h):,}개")
print(f"15분: {len(df_15m):,}개")

# ============================================================
# 지표 계산 (필요시)
# ============================================================

def calc_indicators(df, bb_period=20, rsi_period=14):
    """지표 계산"""
    # BB
    df['bb_mid'] = df['close'].rolling(bb_period).mean()
    df['bb_std'] = df['close'].rolling(bb_period).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_squeeze'] = df['bb_width'] / df['bb_mid'] * 100  # 밴드폭 비율
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # EMA
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # 추세 판단
    df['trend_ema'] = np.where(df['ema20'] > df['ema50'], 'UP', 
                               np.where(df['ema20'] < df['ema50'], 'DOWN', 'FLAT'))
    
    return df

# 각 시간대 지표 계산
df_1d = calc_indicators(df_1d)
df_4h = calc_indicators(df_4h)
df_1h = calc_indicators(df_1h)
df_15m = calc_indicators(df_15m)

# NaN 제거
df_1d = df_1d.dropna(subset=['bb_mid', 'rsi']).reset_index(drop=True)
df_4h = df_4h.dropna(subset=['bb_mid', 'rsi']).reset_index(drop=True)
df_1h = df_1h.dropna(subset=['bb_mid', 'rsi']).reset_index(drop=True)
df_15m = df_15m.dropna(subset=['bb_mid', 'rsi']).reset_index(drop=True)

print(f"\n지표 계산 후:")
print(f"일봉: {len(df_1d):,}개")
print(f"4시간: {len(df_4h):,}개")

# ============================================================
# 상위 시간대 데이터 매핑
# ============================================================

def get_higher_tf_data(dt, df_higher, lookback_hours=0):
    """특정 시점의 상위 시간대 데이터 가져오기"""
    # 해당 시점 이전의 가장 최근 데이터
    mask = df_higher['datetime'] <= dt
    if not mask.any():
        return None
    return df_higher[mask].iloc[-1]

# 4시간 봉에 일봉 추세 매핑
print("\n상위 시간대 매핑 중...")
daily_trends = []
for dt in df_4h['datetime']:
    daily = get_higher_tf_data(dt, df_1d)
    if daily is not None:
        daily_trends.append({
            'trend_1d': daily['trend_ema'],
            'rsi_1d': daily['rsi'],
            'close_vs_ema200_1d': 'ABOVE' if daily['close'] > daily['ema200'] else 'BELOW'
        })
    else:
        daily_trends.append({'trend_1d': None, 'rsi_1d': None, 'close_vs_ema200_1d': None})

df_4h_extended = pd.concat([df_4h, pd.DataFrame(daily_trends)], axis=1)
df_4h_extended = df_4h_extended.dropna(subset=['trend_1d']).reset_index(drop=True)

print(f"매핑 완료: {len(df_4h_extended):,}개 4시간봉")

# ============================================================
# 매매 시뮬레이션 함수
# ============================================================

COST = 0.18  # 수수료 + 슬리피지

def simulate_trade_4h(df, entry_idx, direction, sl_pct=3.0, tp_pct=5.0, max_bars=30):
    """
    4시간봉 기준 매매 시뮬레이션
    max_bars=30 → 약 5일 (30 * 4시간 = 120시간)
    """
    entry_price = df.iloc[entry_idx]['close']
    
    if direction == 'LONG':
        sl = entry_price * (1 - sl_pct / 100)
        tp = entry_price * (1 + tp_pct / 100)
    else:
        sl = entry_price * (1 + sl_pct / 100)
        tp = entry_price * (1 - tp_pct / 100)
    
    mfe = 0  # Maximum Favorable Excursion
    mae = 0  # Maximum Adverse Excursion
    
    for i in range(entry_idx + 1, min(entry_idx + max_bars + 1, len(df))):
        high = df.iloc[i]['high']
        low = df.iloc[i]['low']
        
        if direction == 'LONG':
            mfe = max(mfe, (high - entry_price) / entry_price * 100)
            mae = min(mae, (low - entry_price) / entry_price * 100)
            
            # 손절 먼저 체크
            if low <= sl:
                return {
                    'pnl': -sl_pct,
                    'real_pnl': -sl_pct - COST,
                    'reason': 'SL',
                    'bars': i - entry_idx,
                    'mfe': mfe,
                    'mae': mae
                }
            # 익절
            if high >= tp:
                return {
                    'pnl': tp_pct,
                    'real_pnl': tp_pct - COST,
                    'reason': 'TP',
                    'bars': i - entry_idx,
                    'mfe': mfe,
                    'mae': mae
                }
        else:  # SHORT
            mfe = max(mfe, (entry_price - low) / entry_price * 100)
            mae = min(mae, (entry_price - high) / entry_price * 100)
            
            if high >= sl:
                return {
                    'pnl': -sl_pct,
                    'real_pnl': -sl_pct - COST,
                    'reason': 'SL',
                    'bars': i - entry_idx,
                    'mfe': mfe,
                    'mae': mae
                }
            if low <= tp:
                return {
                    'pnl': tp_pct,
                    'real_pnl': tp_pct - COST,
                    'reason': 'TP',
                    'bars': i - entry_idx,
                    'mfe': mfe,
                    'mae': mae
                }
    
    # 타임아웃 - 현재가로 청산
    exit_price = df.iloc[min(entry_idx + max_bars, len(df) - 1)]['close']
    if direction == 'LONG':
        pnl = (exit_price - entry_price) / entry_price * 100
    else:
        pnl = (entry_price - exit_price) / entry_price * 100
    
    return {
        'pnl': pnl,
        'real_pnl': pnl - COST,
        'reason': 'TIMEOUT',
        'bars': max_bars,
        'mfe': mfe,
        'mae': mae
    }


# ============================================================
# 전략 1: 일봉 추세 + 4시간 BB 터치
# ============================================================
print("\n" + "=" * 80)
print("전략 1: 일봉 추세 방향 + 4시간 BB 터치 진입")
print("=" * 80)

def strategy_trend_bb(df, rsi_filter=True, sl_pct=3.0, tp_pct=5.0):
    """
    일봉 상승추세 + 4시간 BB 하단 터치 → LONG
    일봉 하락추세 + 4시간 BB 상단 터치 → SHORT
    """
    trades = []
    last_trade_idx = 0
    min_interval = 6  # 24시간 간격 (6 * 4시간)
    
    for i in range(50, len(df) - 50):
        if i < last_trade_idx + min_interval:
            continue
        
        row = df.iloc[i]
        
        # LONG 조건: 일봉 상승 + 4H BB 하단 터치 + (RSI < 40)
        if row['trend_1d'] == 'UP' and row['close_vs_ema200_1d'] == 'ABOVE':
            if row['close'] <= row['bb_lower']:
                if not rsi_filter or row['rsi'] < 40:
                    result = simulate_trade_4h(df, i, 'LONG', sl_pct, tp_pct)
                    result['direction'] = 'LONG'
                    result['datetime'] = row['datetime']
                    result['entry_price'] = row['close']
                    result['rsi'] = row['rsi']
                    result['trend_1d'] = row['trend_1d']
                    trades.append(result)
                    last_trade_idx = i
        
        # SHORT 조건: 일봉 하락 + 4H BB 상단 터치 + (RSI > 60)
        elif row['trend_1d'] == 'DOWN' and row['close_vs_ema200_1d'] == 'BELOW':
            if row['close'] >= row['bb_upper']:
                if not rsi_filter or row['rsi'] > 60:
                    result = simulate_trade_4h(df, i, 'SHORT', sl_pct, tp_pct)
                    result['direction'] = 'SHORT'
                    result['datetime'] = row['datetime']
                    result['entry_price'] = row['close']
                    result['rsi'] = row['rsi']
                    result['trend_1d'] = row['trend_1d']
                    trades.append(result)
                    last_trade_idx = i
    
    return pd.DataFrame(trades)


# 파라미터 테스트
print(f"\n{'RSI필터':>8} {'SL%':>6} {'TP%':>6} {'거래수':>8} {'승률':>8} {'평균PnL':>10} {'기대값':>10}")
print("-" * 70)

best_result = None
best_expected = -999

for rsi_filter in [False, True]:
    for sl_pct in [2.0, 3.0, 4.0, 5.0]:
        for tp_pct in [3.0, 5.0, 7.0, 10.0]:
            df_trades = strategy_trend_bb(df_4h_extended, rsi_filter, sl_pct, tp_pct)
            
            if len(df_trades) >= 20:
                win_rate = len(df_trades[df_trades['reason'] == 'TP']) / len(df_trades) * 100
                avg_pnl = df_trades['real_pnl'].mean()
                
                # 기대값 계산
                tp_trades = df_trades[df_trades['reason'] == 'TP']
                sl_trades = df_trades[df_trades['reason'] == 'SL']
                timeout_trades = df_trades[df_trades['reason'] == 'TIMEOUT']
                
                expected = avg_pnl  # 전체 평균이 기대값
                
                rsi_str = 'Y' if rsi_filter else 'N'
                
                if expected > 0.3:  # 유의미한 결과만 출력
                    print(f"{rsi_str:>8} {sl_pct:>6.1f} {tp_pct:>6.1f} {len(df_trades):>8} {win_rate:>8.1f}% {avg_pnl:>10.2f}% {expected:>10.2f}%")
                
                if expected > best_expected:
                    best_expected = expected
                    best_result = {
                        'rsi_filter': rsi_filter,
                        'sl': sl_pct,
                        'tp': tp_pct,
                        'trades': df_trades.copy()
                    }


# ============================================================
# 전략 2: 일봉 추세 전환 + 4시간 확인
# ============================================================
print("\n" + "=" * 80)
print("전략 2: 일봉 EMA 골든/데드크로스 + 4시간 확인")
print("=" * 80)

def strategy_ema_cross(df, df_daily, sl_pct=4.0, tp_pct=8.0):
    """
    일봉 EMA20이 EMA50 돌파 → 4시간 눌림 시 진입
    """
    trades = []
    last_trade_idx = 0
    min_interval = 12  # 48시간 간격
    
    # 일봉 크로스 감지
    df_daily['ema_cross'] = 'NONE'
    for i in range(1, len(df_daily)):
        prev = df_daily.iloc[i-1]
        curr = df_daily.iloc[i]
        
        # 골든 크로스
        if prev['ema20'] <= prev['ema50'] and curr['ema20'] > curr['ema50']:
            df_daily.loc[df_daily.index[i], 'ema_cross'] = 'GOLDEN'
        # 데드 크로스
        elif prev['ema20'] >= prev['ema50'] and curr['ema20'] < curr['ema50']:
            df_daily.loc[df_daily.index[i], 'ema_cross'] = 'DEAD'
    
    # 최근 크로스 기록
    last_cross = None
    last_cross_date = None
    
    for i in range(50, len(df) - 50):
        if i < last_trade_idx + min_interval:
            continue
        
        row = df.iloc[i]
        dt = row['datetime']
        
        # 해당 시점의 일봉 확인
        daily = get_higher_tf_data(dt, df_daily)
        if daily is None:
            continue
        
        # 최근 7일 이내 크로스 확인
        recent_daily = df_daily[(df_daily['datetime'] > dt - pd.Timedelta(days=7)) & 
                                 (df_daily['datetime'] <= dt)]
        
        golden = recent_daily[recent_daily['ema_cross'] == 'GOLDEN']
        dead = recent_daily[recent_daily['ema_cross'] == 'DEAD']
        
        # 골든 크로스 후 4H BB 하단 눌림 → LONG
        if len(golden) > 0 and row['close'] <= row['bb_lower'] * 1.01:
            result = simulate_trade_4h(df, i, 'LONG', sl_pct, tp_pct)
            result['direction'] = 'LONG'
            result['datetime'] = row['datetime']
            result['entry_price'] = row['close']
            result['signal'] = 'GOLDEN_CROSS_PULLBACK'
            trades.append(result)
            last_trade_idx = i
        
        # 데드 크로스 후 4H BB 상단 반등 → SHORT
        elif len(dead) > 0 and row['close'] >= row['bb_upper'] * 0.99:
            result = simulate_trade_4h(df, i, 'SHORT', sl_pct, tp_pct)
            result['direction'] = 'SHORT'
            result['datetime'] = row['datetime']
            result['entry_price'] = row['close']
            result['signal'] = 'DEAD_CROSS_BOUNCE'
            trades.append(result)
            last_trade_idx = i
    
    return pd.DataFrame(trades)


print(f"\n{'SL%':>6} {'TP%':>6} {'거래수':>8} {'승률':>8} {'평균PnL':>10}")
print("-" * 50)

for sl_pct in [3.0, 4.0, 5.0]:
    for tp_pct in [5.0, 8.0, 10.0, 15.0]:
        df_trades = strategy_ema_cross(df_4h_extended, df_1d, sl_pct, tp_pct)
        
        if len(df_trades) >= 10:
            win_rate = len(df_trades[df_trades['reason'] == 'TP']) / len(df_trades) * 100
            avg_pnl = df_trades['real_pnl'].mean()
            
            if avg_pnl > 0.3:
                print(f"{sl_pct:>6.1f} {tp_pct:>6.1f} {len(df_trades):>8} {win_rate:>8.1f}% {avg_pnl:>10.2f}%")
            
            if avg_pnl > best_expected:
                best_expected = avg_pnl
                best_result = {
                    'strategy': 'EMA_CROSS',
                    'sl': sl_pct,
                    'tp': tp_pct,
                    'trades': df_trades.copy()
                }


# ============================================================
# 최적 결과 상세 분석
# ============================================================
if best_result and best_expected > 0:
    print("\n" + "=" * 80)
    print("★★★ 최적 전략 결과 ★★★")
    print("=" * 80)
    
    trades = best_result['trades']
    
    tp = trades[trades['reason'] == 'TP']
    sl = trades[trades['reason'] == 'SL']
    timeout = trades[trades['reason'] == 'TIMEOUT']
    
    print(f"""
■ 설정:
  - 손절: {best_result['sl']}%
  - 익절: {best_result['tp']}%
  
■ 거래 분포:
  - 총 거래: {len(trades)}건 (약 {len(trades)/(5.5*12):.1f}회/월)
  - 익절(TP): {len(tp)}건 ({len(tp)/len(trades)*100:.1f}%)
  - 손절(SL): {len(sl)}건 ({len(sl)/len(trades)*100:.1f}%)
  - 타임아웃: {len(timeout)}건 ({len(timeout)/len(trades)*100:.1f}%)

■ 손익:
  - 평균 PnL (수수료 포함): {trades['real_pnl'].mean():+.2f}%
  - 익절 시 평균: +{tp['real_pnl'].mean():.2f}%
  - 손절 시 평균: {sl['real_pnl'].mean():.2f}%
  - 손익비: {abs(tp['real_pnl'].mean() / sl['real_pnl'].mean()):.2f}

■ MFE/MAE 분석:
  - 평균 MFE: {trades['mfe'].mean():.2f}% (최대 유리 방향)
  - 평균 MAE: {trades['mae'].mean():.2f}% (최대 불리 방향)

■ 월 15% 달성 가능성:
  - 월 평균 거래: {len(trades)/(5.5*12):.1f}회
  - 거래당 평균: {trades['real_pnl'].mean():+.2f}%
  - 월 기대 수익: {trades['real_pnl'].mean() * len(trades)/(5.5*12):.2f}%
""")
    
    # 연도별
    trades['year'] = pd.to_datetime(trades['datetime']).dt.year
    print("\n■ 연도별 성과:")
    yearly = trades.groupby('year').agg({
        'real_pnl': ['count', 'mean', 'sum']
    })
    yearly.columns = ['거래수', '평균PnL%', '총PnL%']
    print(yearly.round(2))
    
    # 방향별
    print("\n■ 방향별 성과:")
    for direction in ['LONG', 'SHORT']:
        subset = trades[trades['direction'] == direction]
        if len(subset) > 0:
            win = len(subset[subset['reason'] == 'TP']) / len(subset) * 100
            print(f"  {direction}: {len(subset)}건, 승률 {win:.1f}%, 평균 {subset['real_pnl'].mean():+.2f}%")
    
    # 저장
    trades.to_csv('mtf_proper_results.csv', index=False)
    print(f"\n저장: mtf_proper_results.csv")

else:
    print("\n수익나는 전략 못찾음. 추가 분석 필요...")
