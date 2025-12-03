import pandas as pd
import numpy as np

"""
추세선 돌파 전략 (Trendline Breakout Strategy)

핵심 로직:
1. Swing High (H) 식별 - 10캔들 딜레이로 확정 (미래 참조 X)
2. H1, H2, H3 (세 개의 연속 하락 고점) 추세선 형성
3. H3 돌파 시 3캔들 대기 후 진입

참고: verify_strict_leakage.py 검증 방식 적용
"""

# 데이터 로드
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])

print("="*100)
print("📈 추세선 돌파 전략 백테스트 (H3 Breakout)")
print("="*100)

# 파라미터
WINDOW = 10  # Swing Point 확정에 필요한 캔들 수
CONFIRM_CANDLES = 3  # 돌파 후 확인 캔들 수
TP_PCT = 2.0  # 익절 %
SL_PCT = 2.0  # 손절 %

trades = []
position = None

# Swing High 리스트 (확정된 것만)
confirmed_highs = []  # [(index, price, datetime), ...]

# 돌파 대기 상태
breakout_pending = None  # {'h3_price': ..., 'breakout_idx': ..., 'confirm_count': 0}

print(f"\n파라미터:")
print(f"  Swing Window: {WINDOW}캔들")
print(f"  확인 캔들: {CONFIRM_CANDLES}개")
print(f"  TP: {TP_PCT}%, SL: {SL_PCT}%")
print("\n시뮬레이션 시작...")

for i in range(WINDOW * 2 + 100, len(candles_df)):
    candle = candles_df.iloc[i]
    
    # === 1. Swing High 확정 (10캔들 딜레이) ===
    # 현재 i 시점에서 i-WINDOW 캔들이 Swing High였는지 확인
    check_idx = i - WINDOW
    
    # check_idx 캔들이 [check_idx - WINDOW, check_idx + WINDOW] 구간에서 최고점인가?
    # 범위: [i - 2*WINDOW, i] -> 현재까지의 데이터만 사용 (미래 참조 X)
    start_idx = max(0, check_idx - WINDOW)
    end_idx = check_idx + WINDOW + 1  # 현재 i까지
    
    if end_idx <= len(candles_df):
        window_highs = candles_df.iloc[start_idx:end_idx]['high'].values
        check_high = candles_df.iloc[check_idx]['high']
        
        if check_high == window_highs.max():
            # Swing High 확정!
            confirmed_highs.append({
                'index': check_idx,
                'price': check_high,
                'time': candles_df.iloc[check_idx]['datetime']
            })
            
            # 너무 많으면 오래된 것 제거
            if len(confirmed_highs) > 10:
                confirmed_highs.pop(0)
    
    # === 2. 청산 체크 ===
    if position is not None:
        entry_price = position['entry_price']
        
        # SL 체크
        sl_price = entry_price * (1 - SL_PCT / 100)
        if candle['low'] <= sl_price:
            pnl_pct = -SL_PCT - 0.11  # 수수료 반영
            trades.append({
                'entry_time': position['entry_time'],
                'entry_price': entry_price,
                'exit_time': candle['datetime'],
                'exit_price': sl_price,
                'exit_reason': 'SL',
                'pnl_pct': pnl_pct,
                'h3_price': position['h3_price']
            })
            position = None
            continue
        
        # TP 체크
        tp_price = entry_price * (1 + TP_PCT / 100)
        if candle['high'] >= tp_price:
            pnl_pct = TP_PCT - 0.11  # 수수료 반영
            trades.append({
                'entry_time': position['entry_time'],
                'entry_price': entry_price,
                'exit_time': candle['datetime'],
                'exit_price': tp_price,
                'exit_reason': 'TP',
                'pnl_pct': pnl_pct,
                'h3_price': position['h3_price']
            })
            position = None
            continue
    
    # === 3. 돌파 대기 중이면 확인 캔들 카운트 ===
    if breakout_pending is not None and position is None:
        breakout_pending['confirm_count'] += 1
        
        # 3캔들 확인 완료 → 진입!
        if breakout_pending['confirm_count'] >= CONFIRM_CANDLES:
            # 현재가가 여전히 H3 위에 있는지 확인
            if candle['close'] > breakout_pending['h3_price']:
                position = {
                    'entry_time': candle['datetime'],
                    'entry_price': candle['close'],
                    'h3_price': breakout_pending['h3_price']
                }
            breakout_pending = None
            continue
    
    # === 4. H3 돌파 감지 ===
    if position is None and breakout_pending is None:
        # 최소 3개의 Swing High 필요
        if len(confirmed_highs) >= 3:
            # 최근 3개의 고점
            h1 = confirmed_highs[-3]
            h2 = confirmed_highs[-2]
            h3 = confirmed_highs[-1]
            
            # 하락 추세선 조건: H1 > H2 > H3 (Lower Highs)
            if h1['price'] > h2['price'] > h3['price']:
                # 현재 종가가 H3를 돌파했는가?
                if candle['close'] > h3['price']:
                    # 이전 캔들은 H3 아래였는가? (돌파 확인)
                    prev_close = candles_df.iloc[i-1]['close']
                    if prev_close <= h3['price']:
                        # 돌파 발생! 3캔들 대기 시작
                        breakout_pending = {
                            'h3_price': h3['price'],
                            'breakout_idx': i,
                            'confirm_count': 0
                        }

print(f"\n백테스트 완료!")

# 결과 분석
trades_df = pd.DataFrame(trades)

print(f"\n📊 추세선 돌파 통계:")
print(f"  확정된 Swing High: {len(confirmed_highs)}개 (마지막 10개 유지)")
print(f"  총 거래: {len(trades_df)}건")

if len(trades_df) > 0:
    tp_trades = trades_df[trades_df['exit_reason'] == 'TP']
    sl_trades = trades_df[trades_df['exit_reason'] == 'SL']
    
    win_rate = len(tp_trades) / len(trades_df) * 100
    total_pnl = trades_df['pnl_pct'].sum()
    avg_pnl = trades_df['pnl_pct'].mean()
    
    print("\n" + "="*100)
    print("📈 추세선 돌파 전략 결과")
    print("="*100)
    print(f"  총 거래: {len(trades_df)}건")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  총 PNL: {total_pnl:.2f}%")
    print(f"  평균 PNL: {avg_pnl:.2f}%")
    
    # 연도별
    trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
    print("\n연도별:")
    for year in sorted(trades_df['year'].unique()):
        yt = trades_df[trades_df['year'] == year]
        yr_win = len(yt[yt['exit_reason'] == 'TP']) / len(yt) * 100
        print(f"  {year}: {len(yt)}건, PNL {yt['pnl_pct'].sum():+.2f}%, 승률 {yr_win:.1f}%")
    
    # 기존 전략과 비교
    print("\n" + "="*100)
    print("📊 전략 비교")
    print("="*100)
    
    old_df = pd.read_csv('backtest_inflection_no_lookahead_results.csv')
    old_pnl = old_df['pnl_pct'].sum()
    old_win_rate = len(old_df[old_df['exit_reason'].str.contains('TP')]) / len(old_df) * 100
    
    print(f"\n{'지표':<15} {'기존 HL전략':<20} {'추세선 돌파':<20} {'차이':<15}")
    print("-"*70)
    print(f"{'거래 수':<15} {len(old_df):<20} {len(trades_df):<20} {len(trades_df)-len(old_df):+d}")
    print(f"{'총 PNL':<15} {old_pnl:<20.2f} {total_pnl:<20.2f} {total_pnl-old_pnl:+.2f}%p")
    print(f"{'승률':<15} {old_win_rate:<20.1f} {win_rate:<20.1f} {win_rate-old_win_rate:+.1f}%p")
    
    trades_df.to_csv('backtest_trendline_breakout_results.csv', index=False)
    print(f"\n✅ 결과 저장 완료")
else:
    print("\n⚠️ 거래가 없습니다.")
