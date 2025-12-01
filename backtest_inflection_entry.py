import pandas as pd
import numpy as np
from datetime import timedelta

# 데이터 로드
candles_df = pd.read_csv('btc_15m_ohlcv.csv')
l_values_df = pd.read_csv('all_L_values.csv')

# 타임스탬프 변환
candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])
l_values_df['datetime'] = pd.to_datetime(l_values_df['datetime'])

print("="*100)
print("변곡점 캔들 진입 백테스트")
print("="*100)

# HL 이벤트 찾기 (현재 L > 이전 L)
hl_events = []
for i in range(1, len(l_values_df)):
    curr_L = l_values_df.iloc[i]['L_value']
    prev_L = l_values_df.iloc[i-1]['L_value']
    
    if curr_L > prev_L:
        hl_events.append({
            'hl_time': l_values_df.iloc[i]['datetime'],
            'hl_price': curr_L,
            'hl_strength': ((curr_L - prev_L) / prev_L) * 100
        })

print(f"\n총 HL 이벤트: {len(hl_events)}개")

# 각 HL 이후 변곡점 캔들 찾기 및 진입
trades = []
position = None

for hl_event in hl_events:
    hl_time = hl_event['hl_time']
    hl_price = hl_event['hl_price']
    hl_strength = hl_event['hl_strength']
    
    # HL 캔들 찾기
    hl_candle_idx = candles_df[candles_df['datetime'] == hl_time].index
    if len(hl_candle_idx) == 0:
        continue
    hl_candle_idx = hl_candle_idx[0]
    
    # HL 이후 최대 20개 캔들 탐색
    max_search_idx = min(hl_candle_idx + 20, len(candles_df))
    
    inflection_found = False
    
    for i in range(hl_candle_idx + 1, max_search_idx):
        candle = candles_df.iloc[i]
        
        # 이미 포지션이 있으면 청산 확인
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
                    'hold_hours': (candle['datetime'] - position['entry_time']).total_seconds() / 3600
                })
                position = None
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
                    'hold_hours': (candle['datetime'] - position['entry_time']).total_seconds() / 3600
                })
                position = None
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
                    'hold_hours': (candle['datetime'] - position['entry_time']).total_seconds() / 3600
                })
                position = None
                continue
        
        # 포지션이 없고 변곡점 캔들을 아직 못 찾았으면 탐색
        if position is None and not inflection_found:
            # 1. 양봉인가?
            is_green = candle['close'] > candle['open']
            if not is_green:
                continue
            
            # 2. 바디 크기
            body_size = candle['close'] - candle['open']
            body_pct = (body_size / candle['open']) * 100
            
            if body_pct < 0.3:  # 최소 0.3%
                continue
            
            # 3. 바디/전체 비율
            total_range = candle['high'] - candle['low']
            if total_range == 0:
                continue
            
            body_to_range = (body_size / total_range) * 100
            
            if body_to_range < 60:  # 최소 60%
                continue
            
            # 4. 거래량 체크 (이전 5개 캔들 평균)
            if i >= 5:
                prev_volume_avg = candles_df.iloc[i-5:i]['volume'].mean()
                volume_ratio = candle['volume'] / prev_volume_avg if prev_volume_avg > 0 else 1
                
                if volume_ratio < 1.0:  # 최소 1.0배
                    continue
            
            # 🎯 변곡점 캔들 발견! 진입!
            entry_price = candle['close']
            
            # TP/SL 설정 (HL 강도별)
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
            sl_price = hl_price * 0.99
            
            position = {
                'hl_time': hl_time,
                'hl_price': hl_price,
                'hl_strength': hl_strength,
                'entry_time': candle['datetime'],
                'entry_price': entry_price,
                'tp1_price': tp1_price,
                'tp2_price': tp2_price,
                'sl_price': sl_price
            }
            
            inflection_found = True

# 결과 분석
trades_df = pd.DataFrame(trades)

if len(trades_df) == 0:
    print("\n⚠️ 거래가 없습니다. 조건을 완화해야 합니다.")
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
        print(f"  최대 이익: {reason_trades['pnl_pct'].max():.2f}%")
        print(f"  최대 손실: {reason_trades['pnl_pct'].min():.2f}%")
        print(f"  평균 보유: {reason_trades['hold_hours'].mean():.2f}시간")
    
    # 전체 통계
    print("\n" + "="*100)
    print("전체 백테스트 결과")
    print("="*100)
    
    print(f"\n📊 기본 통계:")
    print(f"  총 거래: {len(trades_df)}건")
    print(f"  승률: {win_rate:.2f}%")
    print(f"  TP2 성공률: {len(trades_df[trades_df['exit_reason'] == 'TP2_Full'])/len(trades_df)*100:.1f}%")
    print(f"  TP1 성공률: {len(trades_df[trades_df['exit_reason'] == 'TP1_Partial'])/len(trades_df)*100:.1f}%")
    print(f"  SL 비율: {len(sl_trades)/len(trades_df)*100:.1f}%")
    
    print(f"\n💰 수익성:")
    print(f"  총 PNL: {total_pnl:.2f}%")
    print(f"  평균 PNL: {avg_pnl:.2f}%")
    print(f"  최대 이익: {trades_df['pnl_pct'].max():.2f}%")
    print(f"  최대 손실: {trades_df['pnl_pct'].min():.2f}%")
    
    print(f"\n⏱️ 시간:")
    print(f"  평균 보유: {trades_df['hold_hours'].mean():.2f}시간")
    print(f"  중앙값 보유: {trades_df['hold_hours'].median():.2f}시간")
    
    # 기간별 통계
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
    
    # HL 강도별 통계
    print("\n" + "="*100)
    print("HL 강도별 성과")
    print("="*100)
    
    def classify_strength(strength):
        if strength >= 5:
            return '극강 (5%+)'
        elif strength >= 2:
            return '매우강함 (2-5%)'
        elif strength >= 1:
            return '강함 (1-2%)'
        else:
            return '보통 (0.5-1%)'
    
    trades_df['strength_group'] = trades_df['hl_strength'].apply(classify_strength)
    
    for group in ['극강 (5%+)', '매우강함 (2-5%)', '강함 (1-2%)', '보통 (0.5-1%)']:
        group_trades = trades_df[trades_df['strength_group'] == group]
        if len(group_trades) == 0:
            continue
        
        group_win_rate = len(group_trades[group_trades['exit_reason'].str.contains('TP')]) / len(group_trades) * 100
        
        print(f"\n{group}:")
        print(f"  거래 수: {len(group_trades)}건")
        print(f"  평균 PNL: {group_trades['pnl_pct'].mean():.2f}%")
        print(f"  승률: {group_win_rate:.1f}%")
    
    # 저장
    trades_df.to_csv('backtest_inflection_entry_results.csv', index=False)
    print(f"\n✅ 결과 저장: backtest_inflection_entry_results.csv")
    
    # 기존 전략과 비교
    print("\n" + "="*100)
    print("기존 전략 대비 개선")
    print("="*100)
    
    print("\n기존 HL 기본 전략:")
    print("  총 PNL: +17.29%")
    print("  승률: 35.8%")
    print("  거래 수: 1,356건")
    
    print(f"\n변곡점 진입 전략:")
    print(f"  총 PNL: {total_pnl:+.2f}%")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  거래 수: {len(trades_df)}건")
    
    print(f"\n개선:")
    print(f"  PNL 차이: {total_pnl - 17.29:+.2f}%p")
    print(f"  승률 차이: {win_rate - 35.8:+.1f}%p")

