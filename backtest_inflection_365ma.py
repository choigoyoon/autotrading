import pandas as pd
import numpy as np

# 데이터 로드
candles_df = pd.read_csv('btc_15m_ohlcv.csv')

# 타임스탬프 변환
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])

print("="*100)
print("변곡점 캔들 진입 백테스트 + 365일 MA 돌파 필터")
print("="*100)

# === 365일 이동평균 계산 ===
# 15분봉 기준: 365일 = 365 * 24 * 4 = 35,040개 봉
# 실용적으로 365일 MA = 35040개 봉의 평균
print("\n365일 이동평균 계산 중...")

MA_365_PERIOD = 365 * 24 * 4  # 35,040개 15분봉 = 365일
candles_df['ma_365d'] = candles_df['close'].rolling(window=MA_365_PERIOD).mean()

# MA 위/아래 판단
candles_df['above_365ma'] = candles_df['close'] > candles_df['ma_365d']

print(f"✅ 365일 MA 계산 완료 (기간: {MA_365_PERIOD}개 봉)")
print(f"   365일 MA 유효 데이터: {candles_df['ma_365d'].notna().sum()}개")

# 실시간 시뮬레이션
trades = []
position = None

# 상태 변수
recent_lows = []
last_L = None
current_hl_detected = False
searching_inflection = False
hl_event = None

# 필터링 통계
ma_filter_count = 0
total_inflection_candidates = 0

print("\n실시간 시뮬레이션 시작 (365일 MA 필터 적용)...")
print("(각 캔들을 순차적으로 처리, 미래 데이터 사용 안 함)\n")

# 365일 MA가 유효한 시점부터 시작
start_idx = max(100, MA_365_PERIOD + 10)

for i in range(start_idx, len(candles_df)):
    candle = candles_df.iloc[i]
    
    # 이전 20개 캔들
    prev_candles = candles_df.iloc[max(0, i-20):i]
    
    # === Step 1: 저점(L) 감지 ===
    if len(prev_candles) >= 10:
        recent_low = prev_candles['low'].tail(10).min()
        
        if candle['low'] <= recent_low * 1.002:
            recent_lows.append({
                'price': candle['low'],
                'time': candle['datetime'],
                'index': i
            })
            
            if len(recent_lows) > 5:
                recent_lows.pop(0)
    
    # === Step 2: HL (Higher Low) 감지 ===
    if len(recent_lows) >= 2 and not searching_inflection:
        current_low = recent_lows[-1]['price']
        previous_low = recent_lows[-2]['price']
        
        if current_low > previous_low:
            if candle['close'] > current_low * 1.003:
                hl_strength = ((current_low - previous_low) / previous_low) * 100
                
                if hl_strength >= 0.5:
                    hl_event = {
                        'hl_time': recent_lows[-1]['time'],
                        'hl_price': current_low,
                        'hl_strength': hl_strength,
                        'hl_index': recent_lows[-1]['index']
                    }
                    
                    searching_inflection = True
                    current_hl_detected = True
    
    # === Step 3: 포지션 있으면 청산 체크 ===
    if position is not None:
        # SL 체크
        if candle['low'] <= position['sl_price']:
            exit_price = position['sl_price']
            pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
            
            trades.append({
                'hl_time': position['hl_time'],
                'hl_price': position['hl_price'],
                'hl_strength': position['hl_strength'],
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': candle['datetime'],
                'exit_price': exit_price,
                'exit_reason': 'SL',
                'pnl_pct': pnl_pct,
                'hold_hours': (candle['datetime'] - position['entry_time']).total_seconds() / 3600,
                'ma_365d': position['ma_365d'],
                'above_ma': position['above_ma']
            })
            position = None
            searching_inflection = False
            continue
        
        # TP2 체크
        if candle['high'] >= position['tp2_price']:
            exit_price = position['tp2_price']
            pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
            
            trades.append({
                'hl_time': position['hl_time'],
                'hl_price': position['hl_price'],
                'hl_strength': position['hl_strength'],
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': candle['datetime'],
                'exit_price': exit_price,
                'exit_reason': 'TP2_Full',
                'pnl_pct': pnl_pct,
                'hold_hours': (candle['datetime'] - position['entry_time']).total_seconds() / 3600,
                'ma_365d': position['ma_365d'],
                'above_ma': position['above_ma']
            })
            position = None
            searching_inflection = False
            continue
        
        # TP1 체크
        if candle['high'] >= position['tp1_price']:
            exit_price = position['tp1_price']
            pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
            
            trades.append({
                'hl_time': position['hl_time'],
                'hl_price': position['hl_price'],
                'hl_strength': position['hl_strength'],
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': candle['datetime'],
                'exit_price': exit_price,
                'exit_reason': 'TP1_Partial',
                'pnl_pct': pnl_pct,
                'hold_hours': (candle['datetime'] - position['entry_time']).total_seconds() / 3600,
                'ma_365d': position['ma_365d'],
                'above_ma': position['above_ma']
            })
            position = None
            searching_inflection = False
            continue
    
    # === Step 4: 변곡점 캔들 찾기 ===
    if position is None and searching_inflection and hl_event is not None:
        # HL 이후 20개 캔들까지만 탐색
        if i - hl_event['hl_index'] > 20:
            searching_inflection = False
            hl_event = None
            continue
        
        # 변곡점 캔들 조건 체크
        # 1. 양봉
        if candle['close'] <= candle['open']:
            continue
        
        # 2. 바디 크기
        body_size = candle['close'] - candle['open']
        body_pct = (body_size / candle['open']) * 100
        
        if body_pct < 0.3:
            continue
        
        # 3. 바디/전체 비율
        total_range = candle['high'] - candle['low']
        if total_range == 0:
            continue
        
        body_to_range = (body_size / total_range) * 100
        
        if body_to_range < 60:
            continue
        
        # 4. 거래량
        if i >= 5:
            prev_volume_avg = candles_df.iloc[i-5:i]['volume'].mean()
            volume_ratio = candle['volume'] / prev_volume_avg if prev_volume_avg > 0 else 1
            
            if volume_ratio < 1.0:
                continue
        
        # ⭐ 변곡점 캔들 후보 발견!
        total_inflection_candidates += 1
        
        # === NEW: 365일 MA 필터 적용 ===
        ma_365d = candles_df.iloc[i]['ma_365d']
        above_ma = candles_df.iloc[i]['above_365ma']
        
        # 365일 MA 위에 있어야 진입 (상승장)
        if pd.notna(ma_365d) and not above_ma:
            ma_filter_count += 1
            continue
        
        # 🎯 365일 MA 필터 통과! 진입!
        entry_price = candle['close']
        
        # TP/SL 설정
        hl_strength = hl_event['hl_strength']
        if hl_strength >= 5:
            tp1_pct, tp2_pct = 2.0, 4.0
        elif hl_strength >= 2:
            tp1_pct, tp2_pct = 1.5, 3.0
        elif hl_strength >= 1:
            tp1_pct, tp2_pct = 1.0, 2.0
        else:
            tp1_pct, tp2_pct = 0.7, 1.5
        
        tp1_price = entry_price * (1 + tp1_pct / 100)
        tp2_price = entry_price * (1 + tp2_pct / 100)
        sl_price = hl_event['hl_price'] * 0.99
        
        position = {
            'hl_time': hl_event['hl_time'],
            'hl_price': hl_event['hl_price'],
            'hl_strength': hl_strength,
            'entry_time': candle['datetime'],
            'entry_price': entry_price,
            'tp1_price': tp1_price,
            'tp2_price': tp2_price,
            'sl_price': sl_price,
            'ma_365d': ma_365d,
            'above_ma': above_ma
        }
        
        searching_inflection = False

print(f"백테스트 완료!\n")

# 결과 분석
trades_df = pd.DataFrame(trades)

print(f"\n📊 365일 MA 필터링 통계:")
print(f"  변곡점 캔들 후보: {total_inflection_candidates}개")
print(f"  365일 MA 아래 (차단): {ma_filter_count}개 ({ma_filter_count/total_inflection_candidates*100 if total_inflection_candidates > 0 else 0:.1f}%)")
print(f"  365일 MA 위 (진입): {len(trades_df)}개 ({len(trades_df)/total_inflection_candidates*100 if total_inflection_candidates > 0 else 0:.1f}%)")

if len(trades_df) == 0:
    print("\n⚠️ 거래가 없습니다.")
else:
    print(f"\n총 거래: {len(trades_df)}건")
    
    # 승률 계산
    tp_trades = trades_df[trades_df['exit_reason'].str.contains('TP')]
    sl_trades = trades_df[trades_df['exit_reason'] == 'SL']
    
    win_rate = len(tp_trades) / len(trades_df) * 100
    
    # PNL 계산
    total_pnl = trades_df['pnl_pct'].sum()
    avg_pnl = trades_df['pnl_pct'].mean()
    
    # 청산 사유별 통계
    print("\n" + "="*100)
    print("청산 사유별 통계")
    print("="*100)
    
    for reason in trades_df['exit_reason'].unique():
        reason_trades = trades_df[trades_df['exit_reason'] == reason]
        print(f"\n{reason}:")
        print(f"  거래 수: {len(reason_trades)} ({len(reason_trades)/len(trades_df)*100:.1f}%)")
        print(f"  평균 PNL: {reason_trades['pnl_pct'].mean():.2f}%")
        print(f"  총 PNL: {reason_trades['pnl_pct'].sum():.2f}%")
    
    # 전체 통계
    print("\n" + "="*100)
    print("전체 백테스트 결과 (365일 MA 필터 적용)")
    print("="*100)
    
    print(f"\n📊 기본 통계:")
    print(f"  총 거래: {len(trades_df)}건")
    print(f"  승률: {win_rate:.2f}%")
    print(f"  SL 비율: {len(sl_trades)/len(trades_df)*100:.1f}%")
    
    print(f"\n💰 수익성:")
    print(f"  총 PNL: {total_pnl:.2f}%")
    print(f"  평균 PNL: {avg_pnl:.2f}%")
    print(f"  최대 이익: {trades_df['pnl_pct'].max():.2f}%")
    print(f"  최대 손실: {trades_df['pnl_pct'].min():.2f}%")
    
    print(f"\n⏱️ 시간:")
    print(f"  평균 보유: {trades_df['hold_hours'].mean():.2f}시간")
    
    # 연도별 통계
    trades_df['year'] = trades_df['entry_time'].dt.year
    
    print("\n" + "="*100)
    print("연도별 성과")
    print("="*100)
    
    for year in sorted(trades_df['year'].unique()):
        year_trades = trades_df[trades_df['year'] == year]
        year_win_rate = len(year_trades[year_trades['exit_reason'].str.contains('TP')]) / len(year_trades) * 100
        
        print(f"\n{year}년:")
        print(f"  거래 수: {len(year_trades)}건")
        print(f"  총 PNL: {year_trades['pnl_pct'].sum():.2f}%")
        print(f"  평균 PNL: {year_trades['pnl_pct'].mean():.2f}%")
        print(f"  승률: {year_win_rate:.1f}%")
    
    # 저장
    trades_df.to_csv('backtest_inflection_365ma_results.csv', index=False)
    print(f"\n✅ 결과 저장: backtest_inflection_365ma_results.csv")
    
    # 기존 전략과 비교
    print("\n" + "="*100)
    print("📊 전략 비교 (기존 vs 365일 MA)")
    print("="*100)
    
    try:
        old_df = pd.read_csv('backtest_inflection_no_lookahead_results.csv')
        old_pnl = old_df['pnl_pct'].sum()
        old_win_rate = len(old_df[old_df['exit_reason'].str.contains('TP')]) / len(old_df) * 100
        old_sl_rate = len(old_df[old_df['exit_reason'] == 'SL']) / len(old_df) * 100
        old_avg_pnl = old_df['pnl_pct'].mean()
        old_trades = len(old_df)
        
        print(f"\n기존 전략 (필터 없음):")
        print(f"  총 거래: {old_trades}건")
        print(f"  총 PNL: {old_pnl:.2f}%")
        print(f"  평균 PNL: {old_avg_pnl:.2f}%")
        print(f"  승률: {old_win_rate:.2f}%")
        print(f"  SL 비율: {old_sl_rate:.1f}%")
        
        print(f"\n365일 MA 전략:")
        print(f"  총 거래: {len(trades_df)}건 ({len(trades_df) - old_trades:+d})")
        print(f"  총 PNL: {total_pnl:.2f}% ({total_pnl - old_pnl:+.2f}%p)")
        print(f"  평균 PNL: {avg_pnl:.2f}% ({avg_pnl - old_avg_pnl:+.2f}%p)")
        print(f"  승률: {win_rate:.2f}% ({win_rate - old_win_rate:+.2f}%p)")
        print(f"  SL 비율: {len(sl_trades)/len(trades_df)*100:.1f}% ({len(sl_trades)/len(trades_df)*100 - old_sl_rate:+.1f}%p)")
        
    except FileNotFoundError:
        print("\n⚠️ 기존 결과 파일을 찾을 수 없습니다.")
