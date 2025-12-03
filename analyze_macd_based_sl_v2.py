import pandas as pd
import numpy as np
from datetime import timedelta

"""
MACD 기반 손절 분석 V2 (사용자 지시사항 반영)
- 진입 시 MACD < 0: 기존 SL(L값 기반) 사용
- 진입 시 MACD > 0: MACD가 0 아래로 떨어지면 청산

HL 구조 분석 포함
"""

print("=" * 100)
print("📊 MACD 기반 Stop Loss 전략 분석 V2")
print("=" * 100)

# 데이터 로드
print("\n데이터 로딩 중...")
analysis_df = pd.read_csv('analysis_15m.csv')
analysis_df['datetime'] = pd.to_datetime(analysis_df['datetime'])
analysis_df.set_index('datetime', inplace=True)

breakouts = pd.read_csv('breakouts_v2.csv')
breakouts['breakout_time'] = pd.to_datetime(breakouts['breakout_time'])

h_values = pd.read_csv('h_values_v2.csv')
h_values['datetime'] = pd.to_datetime(h_values['datetime'])

l_values = pd.read_csv('l_values_v2.csv')
l_values['datetime'] = pd.to_datetime(l_values['datetime'])

print(f"분석 데이터: {len(analysis_df)}행")
print(f"돌파 신호: {len(breakouts)}건")
print(f"H 값: {len(h_values)}건, L 값: {len(l_values)}건")

# 함수들
def get_previous_l(breakout_idx):
    prev_l = l_values[l_values['idx'] < breakout_idx].sort_values('idx', ascending=False)
    if len(prev_l) > 0:
        return prev_l.iloc[0]['price']
    return None

def get_h_resistance_values(breakout_idx, entry_price):
    prev_h = h_values[(h_values['idx'] < breakout_idx)]
    prev_h = prev_h.sort_values('idx', ascending=False)
    
    h_resistance_list = []
    for _, h in prev_h.head(50).iterrows():
        if h['price'] > entry_price:
            h_resistance_list.append(h['price'])
    
    h_resistance_list = sorted(set(h_resistance_list))
    return h_resistance_list[:5]

def get_hl_count(breakout_idx):
    """돌파 이전의 HL(Higher Low) 개수 계산"""
    prev_l = l_values[(l_values['idx'] < breakout_idx) & (l_values['is_HL'] == True)]
    return len(prev_l.tail(20))  # 최근 20개 L값 중 HL 개수

def is_bullish_candle(candle):
    """양봉인지 확인"""
    return candle['close'] > candle['open']

def get_candle_body_ratio(candle):
    """캔들 몸통 비율"""
    body = abs(candle['close'] - candle['open'])
    total = candle['high'] - candle['low']
    if total > 0:
        return body / total
    return 0

# MACD 기반 청산 분석
print("\n" + "=" * 100)
print("🔄 MACD 기반 청산 시뮬레이션")
print("=" * 100)

results = []
lookforward = 96  # 24시간

for idx, br in breakouts.iterrows():
    try:
        breakout_time = br['breakout_time']
        breakout_idx = br['breakout_idx']
        entry_price = br['breakout_price']
        
        if breakout_time not in analysis_df.index:
            continue
        
        entry_iloc = analysis_df.index.get_loc(breakout_time)
        entry_candle = analysis_df.iloc[entry_iloc]
        
        # 진입 시점 MACD
        entry_macd = entry_candle['macd']
        macd_above_zero = entry_macd > 0
        
        # 진입 캔들 정보
        bullish_entry = is_bullish_candle(entry_candle)
        body_ratio = get_candle_body_ratio(entry_candle)
        
        # HL 개수
        hl_count = get_hl_count(breakout_idx)
        
        # SL 설정: L값 기반
        sl_price = get_previous_l(breakout_idx)
        if sl_price is None:
            continue
        
        # TP 설정
        h_resistance_list = get_h_resistance_values(breakout_idx, entry_price)
        if len(h_resistance_list) == 0:
            tp1 = entry_price * 1.01
            tp2 = entry_price * 1.02
            tp3 = entry_price * 1.03
        else:
            tp1 = h_resistance_list[0] if len(h_resistance_list) >= 1 else entry_price * 1.01
            tp2 = h_resistance_list[1] if len(h_resistance_list) >= 2 else entry_price * 1.02
            tp3 = h_resistance_list[2] if len(h_resistance_list) >= 3 else entry_price * 1.03
        
        # 손익비 계산
        sl_dist = (entry_price - sl_price) / entry_price * 100
        tp1_dist = (tp1 - entry_price) / entry_price * 100
        
        # 시뮬레이션
        exit_price = None
        exit_reason = None
        exit_time = None
        tp1_hit = False
        tp2_hit = False
        tp3_hit = False
        macd_exit_triggered = False
        candles_to_exit = 0
        max_profit_seen = 0
        
        for i in range(1, min(lookforward, len(analysis_df) - entry_iloc)):
            candle = analysis_df.iloc[entry_iloc + i]
            candle_time = analysis_df.index[entry_iloc + i]
            current_high = candle['high']
            current_low = candle['low']
            current_macd = candle['macd']
            
            # 최대 이익 추적
            current_profit = (current_high - entry_price) / entry_price * 100
            if current_profit > max_profit_seen:
                max_profit_seen = current_profit
            
            # MACD 청산 조건 (MACD > 0 진입만)
            if macd_above_zero and current_macd < 0 and not tp1_hit:
                exit_price = candle['close']
                exit_reason = 'MACD_CROSS_BELOW_0'
                exit_time = candle_time
                macd_exit_triggered = True
                candles_to_exit = i
                break
            
            # SL 확인 (MACD < 0 진입, 또는 MACD > 0이지만 TP1 도달 후)
            if current_low <= sl_price:
                if not macd_above_zero or tp1_hit:
                    exit_price = sl_price
                    exit_reason = 'SL_HIT'
                    exit_time = candle_time
                    candles_to_exit = i
                    break
            
            # TP 확인
            if not tp1_hit and current_high >= tp1:
                tp1_hit = True
            if tp2 and not tp2_hit and current_high >= tp2:
                tp2_hit = True
            if tp3 and not tp3_hit and current_high >= tp3:
                tp3_hit = True
                exit_price = tp3
                exit_reason = 'TP3_FULL'
                exit_time = candle_time
                candles_to_exit = i
                break
        
        # 시간 만료
        if exit_price is None:
            last_candle = analysis_df.iloc[min(entry_iloc + lookforward - 1, len(analysis_df) - 1)]
            exit_price = last_candle['close']
            exit_reason = 'TIME_STOP'
            exit_time = analysis_df.index[min(entry_iloc + lookforward - 1, len(analysis_df) - 1)]
            candles_to_exit = lookforward
        
        # PnL 계산
        if exit_reason == 'MACD_CROSS_BELOW_0':
            pnl_100_tp1 = (exit_price - entry_price) / entry_price * 100
        elif tp1_hit:
            pnl_100_tp1 = tp1_dist
        else:
            pnl_100_tp1 = (exit_price - entry_price) / entry_price * 100
        
        # 50/50 분할
        if exit_reason == 'MACD_CROSS_BELOW_0':
            pnl_50_50 = (exit_price - entry_price) / entry_price * 100
        elif tp1_hit and tp2 and tp2_hit:
            tp2_dist = (tp2 - entry_price) / entry_price * 100
            pnl_50_50 = 0.5 * tp1_dist + 0.5 * tp2_dist
        elif tp1_hit and tp2:
            final_pnl = (exit_price - entry_price) / entry_price * 100
            pnl_50_50 = 0.5 * tp1_dist + 0.5 * final_pnl
        elif tp1_hit:
            pnl_50_50 = tp1_dist
        else:
            pnl_50_50 = (exit_price - entry_price) / entry_price * 100
        
        # Gap 계산
        gap = br['break_strength'] * 100 if 'break_strength' in br and pd.notna(br['break_strength']) else 0
        
        results.append({
            'breakout_time': breakout_time,
            'entry_price': entry_price,
            'entry_macd': entry_macd,
            'macd_above_zero': macd_above_zero,
            'bullish_entry': bullish_entry,
            'body_ratio': body_ratio,
            'hl_count': hl_count,
            'sl_price': sl_price,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'sl_dist': sl_dist,
            'tp1_dist': tp1_dist,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'exit_time': exit_time,
            'tp1_hit': tp1_hit,
            'tp2_hit': tp2_hit,
            'tp3_hit': tp3_hit,
            'macd_exit': macd_exit_triggered,
            'candles_to_exit': candles_to_exit,
            'max_profit_seen': max_profit_seen,
            'pnl_100_tp1': pnl_100_tp1,
            'pnl_50_50': pnl_50_50,
            'gap': gap
        })
        
    except Exception as e:
        continue

results_df = pd.DataFrame(results)
print(f"\n분석 완료: {len(results_df)}건")

# 결과 분석
print("\n" + "=" * 100)
print("📊 전체 결과 분석")
print("=" * 100)

print(f"\n총 거래: {len(results_df)}건")
print(f"  - MACD > 0 진입: {results_df['macd_above_zero'].sum()}건 ({results_df['macd_above_zero'].mean()*100:.1f}%)")
print(f"  - MACD < 0 진입: {(~results_df['macd_above_zero']).sum()}건 ({(~results_df['macd_above_zero']).mean()*100:.1f}%)")

# MACD 청산 통계
macd_exits = results_df[results_df['macd_exit'] == True]
print(f"\n🔴 MACD 0선 하향돌파 청산: {len(macd_exits)}건")
if len(macd_exits) > 0:
    print(f"  - 평균 PnL: {macd_exits['pnl_100_tp1'].mean():.3f}%")
    print(f"  - 청산까지 평균 캔들: {macd_exits['candles_to_exit'].mean():.1f}개")
    print(f"  - 청산 전 최대 이익: {macd_exits['max_profit_seen'].mean():.3f}%")

# 청산 사유별 통계
print("\n청산 사유별 통계:")
for reason in results_df['exit_reason'].unique():
    subset = results_df[results_df['exit_reason'] == reason]
    print(f"  {reason}: {len(subset)}건 ({len(subset)/len(results_df)*100:.1f}%), 평균 PnL: {subset['pnl_100_tp1'].mean():.3f}%")

# MACD 조건별 분석
print("\n" + "=" * 100)
print("📊 MACD 진입 조건별 성과 분석")
print("=" * 100)

macd_pos = results_df[results_df['macd_above_zero'] == True]
macd_neg = results_df[results_df['macd_above_zero'] == False]

print(f"\n🟢 MACD > 0 진입 ({len(macd_pos)}건):")
print(f"  - 100% TP1 승률: {(macd_pos['pnl_100_tp1'] > 0).mean()*100:.1f}%, 평균 PnL: {macd_pos['pnl_100_tp1'].mean():.3f}%")
print(f"  - 50/50 분할 승률: {(macd_pos['pnl_50_50'] > 0).mean()*100:.1f}%, 평균 PnL: {macd_pos['pnl_50_50'].mean():.3f}%")
print(f"  - MACD 청산: {macd_pos['macd_exit'].sum()}건 ({macd_pos['macd_exit'].mean()*100:.1f}%)")
print(f"  - TP1 도달: {macd_pos['tp1_hit'].sum()}건 ({macd_pos['tp1_hit'].mean()*100:.1f}%)")

print(f"\n🔴 MACD < 0 진입 ({len(macd_neg)}건):")
print(f"  - 100% TP1 승률: {(macd_neg['pnl_100_tp1'] > 0).mean()*100:.1f}%, 평균 PnL: {macd_neg['pnl_100_tp1'].mean():.3f}%")
print(f"  - 50/50 분할 승률: {(macd_neg['pnl_50_50'] > 0).mean()*100:.1f}%, 평균 PnL: {macd_neg['pnl_50_50'].mean():.3f}%")
print(f"  - SL 도달: {macd_neg[macd_neg['exit_reason']=='SL_HIT'].shape[0]}건")
print(f"  - TP1 도달: {macd_neg['tp1_hit'].sum()}건 ({macd_neg['tp1_hit'].mean()*100:.1f}%)")

# HL 개수별 분석
print("\n" + "=" * 100)
print("📊 HL(Higher Low) 개수별 성과 분석")
print("=" * 100)

for hl in sorted(results_df['hl_count'].unique()):
    if hl <= 10:  # 10개까지만 표시
        subset = results_df[results_df['hl_count'] == hl]
        if len(subset) >= 10:  # 10건 이상만
            win_rate = (subset['pnl_50_50'] > 0).mean() * 100
            avg_pnl = subset['pnl_50_50'].mean()
            print(f"  HL={hl}: {len(subset)}건, 승률 {win_rate:.1f}%, 평균 PnL {avg_pnl:.3f}%")

# 양봉 분석
print("\n" + "=" * 100)
print("📊 진입 캔들 분석")
print("=" * 100)

bullish = results_df[results_df['bullish_entry'] == True]
bearish = results_df[results_df['bullish_entry'] == False]

print(f"\n양봉 진입 ({len(bullish)}건):")
print(f"  - 승률: {(bullish['pnl_50_50'] > 0).mean()*100:.1f}%, 평균 PnL: {bullish['pnl_50_50'].mean():.3f}%")

print(f"\n음봉 진입 ({len(bearish)}건):")
print(f"  - 승률: {(bearish['pnl_50_50'] > 0).mean()*100:.1f}%, 평균 PnL: {bearish['pnl_50_50'].mean():.3f}%")

# 복합 조건 분석
print("\n" + "=" * 100)
print("🏆 최적 조건 조합 분석")
print("=" * 100)

conditions = [
    ('전체', results_df),
    ('MACD > 0', macd_pos),
    ('MACD < 0', macd_neg),
    ('MACD < 0 + 양봉', macd_neg[macd_neg['bullish_entry'] == True]),
    ('MACD < 0 + HL >= 2', macd_neg[macd_neg['hl_count'] >= 2]),
    ('MACD < 0 + 양봉 + HL >= 2', macd_neg[(macd_neg['bullish_entry'] == True) & (macd_neg['hl_count'] >= 2)]),
    ('MACD < 0 + 장대양봉(70%+)', macd_neg[(macd_neg['bullish_entry'] == True) & (macd_neg['body_ratio'] >= 0.7)]),
]

print(f"\n{'조건':<30} {'건수':>8} {'승률':>10} {'평균PnL':>10} {'총PnL':>12}")
print("-" * 75)
for name, df in conditions:
    if len(df) > 0:
        win_rate = (df['pnl_50_50'] > 0).mean() * 100
        avg_pnl = df['pnl_50_50'].mean()
        total_pnl = df['pnl_50_50'].sum()
        print(f"{name:<30} {len(df):>8} {win_rate:>9.1f}% {avg_pnl:>9.3f}% {total_pnl:>11.2f}%")

# 실패 사례 분석 (MACD > 0 진입 후 MACD 청산)
print("\n" + "=" * 100)
print("🔍 MACD 청산 실패 사례 분석")
print("=" * 100)

if len(macd_exits) > 0:
    print(f"\nMACD 청산 사례 (MACD > 0 진입 후 MACD가 0 아래로 떨어져 청산):")
    print(f"  - 총 {len(macd_exits)}건")
    print(f"  - 평균 PnL: {macd_exits['pnl_100_tp1'].mean():.3f}%")
    print(f"  - 양봉 진입 비율: {macd_exits['bullish_entry'].mean()*100:.1f}%")
    print(f"  - 평균 HL 개수: {macd_exits['hl_count'].mean():.1f}")
    print(f"  - 청산 전 최대 이익: {macd_exits['max_profit_seen'].mean():.3f}%")
    
    # MACD 청산 시점에 이미 이익인 경우
    profitable_macd_exit = macd_exits[macd_exits['pnl_100_tp1'] > 0]
    losing_macd_exit = macd_exits[macd_exits['pnl_100_tp1'] <= 0]
    print(f"\n  - 이익으로 청산: {len(profitable_macd_exit)}건 ({len(profitable_macd_exit)/len(macd_exits)*100:.1f}%)")
    print(f"  - 손실로 청산: {len(losing_macd_exit)}건 ({len(losing_macd_exit)/len(macd_exits)*100:.1f}%)")
    
    if len(losing_macd_exit) > 0:
        print(f"    → 손실 청산 평균 손실: {losing_macd_exit['pnl_100_tp1'].mean():.3f}%")
        print(f"    → 손실 청산 전 최대 이익: {losing_macd_exit['max_profit_seen'].mean():.3f}%")

# 핵심 인사이트
print("\n" + "=" * 100)
print("💡 핵심 인사이트 및 결론")
print("=" * 100)

best_condition = macd_neg[(macd_neg['bullish_entry'] == True) & (macd_neg['hl_count'] >= 2)]
best_win_rate = (best_condition['pnl_50_50'] > 0).mean() * 100 if len(best_condition) > 0 else 0
best_pnl = best_condition['pnl_50_50'].mean() if len(best_condition) > 0 else 0

print(f"""
📌 MACD 기반 SL 전략 결론:

1. MACD > 0 진입 (승률 {(macd_pos['pnl_50_50'] > 0).mean()*100:.1f}%, 평균 PnL {macd_pos['pnl_50_50'].mean():.3f}%):
   - MACD가 0 아래로 떨어지면 청산하는 규칙 적용
   - MACD 청산 발생률: {macd_pos['macd_exit'].mean()*100:.1f}%
   - MACD 청산 시 평균 손실: {macd_exits['pnl_100_tp1'].mean():.3f}%
   - ⚠️ MACD > 0 진입은 성과가 낮음 (추세 이미 진행 중)

2. MACD < 0 진입 (승률 {(macd_neg['pnl_50_50'] > 0).mean()*100:.1f}%, 평균 PnL {macd_neg['pnl_50_50'].mean():.3f}%):
   - 기존 L값 기반 SL 사용
   - ✅ MACD < 0 진입이 더 유리 (추세 반전 초기 진입)

3. 최적 조건 (MACD < 0 + 양봉 + HL >= 2):
   - {len(best_condition)}건, 승률 {best_win_rate:.1f}%, 평균 PnL {best_pnl:.3f}%

4. 전략 권고:
   - MACD < 0에서 진입하는 것이 유리
   - MACD > 0 진입 시 MACD 0선 하향돌파 청산은 손실 최소화에 도움
   - HL >= 2 + 양봉 조건 추가 시 승률 개선
""")

# 결과 저장
results_df.to_csv('macd_sl_analysis_v2_results.csv', index=False)
print(f"\n✅ 결과 저장: macd_sl_analysis_v2_results.csv")
