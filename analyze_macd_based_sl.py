import pandas as pd
import numpy as np
from datetime import timedelta

"""
MACD 기반 손절 분석 (사용자 지시사항)
- 진입 시 MACD < 0: 기존 SL(L값 기반) 사용
- 진입 시 MACD > 0: MACD가 0 아래로 떨어지면 청산

분할 익절:
- TP1: 돌파 전 H_inner (첫 번째 저항)
- TP2: 그 다음 H_inner
- TP3: 세 번째 H_inner
"""

print("=" * 100)
print("📊 MACD 기반 Stop Loss 전략 분석")
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

# 함수: 특정 시점 이전의 가장 최근 L값 찾기
def get_previous_l(breakout_time, breakout_idx):
    prev_l = l_values[l_values['idx'] < breakout_idx].sort_values('idx', ascending=False)
    if len(prev_l) > 0:
        return prev_l.iloc[0]['price']
    return None

# 함수: 특정 시점 이전의 H 값들 중 진입가보다 높은 저항선 찾기
def get_h_resistance_values(breakout_time, breakout_idx, entry_price):
    # 돌파 이전의 H값들 중 진입가보다 높은 것들 (저항선)
    prev_h = h_values[(h_values['idx'] < breakout_idx)]
    prev_h = prev_h.sort_values('idx', ascending=False)
    
    h_resistance_list = []
    for _, h in prev_h.head(50).iterrows():  # 최근 50개 중에서
        if h['price'] > entry_price:  # 진입가보다 높은 H값만
            h_resistance_list.append(h['price'])
    
    # 가격 기준 정렬 (낮은 것부터 = 첫 번째 저항)
    h_resistance_list = sorted(set(h_resistance_list))  # 중복 제거 후 정렬
    return h_resistance_list[:5]  # 상위 5개 저항선

# MACD 기반 청산 분석
print("\n" + "=" * 100)
print("🔄 MACD 기반 청산 시뮬레이션")
print("=" * 100)

results = []
lookforward = 96  # 24시간 (15분 * 96 = 24시간)

for idx, br in breakouts.iterrows():
    try:
        breakout_time = br['breakout_time']
        breakout_idx = br['breakout_idx']
        entry_price = br['breakout_price']
        trendline_price = br['trendline_price']
        
        # 진입 시점 인덱스 찾기
        if breakout_time not in analysis_df.index:
            continue
        
        entry_iloc = analysis_df.index.get_loc(breakout_time)
        
        # 진입 시점 MACD
        entry_macd = analysis_df.iloc[entry_iloc]['macd']
        macd_above_zero = entry_macd > 0
        
        # SL 설정: L값 기반
        sl_price = get_previous_l(breakout_time, breakout_idx)
        if sl_price is None:
            continue
        
        # TP 설정: H값 기반 저항선
        h_resistance_list = get_h_resistance_values(breakout_time, breakout_idx, entry_price)
        
        # H 저항선이 없으면 고정 TP 사용
        if len(h_resistance_list) == 0:
            # 고정 TP: 1%, 2%, 3%
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
        
        # 청산까지 추적
        for i in range(1, min(lookforward, len(analysis_df) - entry_iloc)):
            candle = analysis_df.iloc[entry_iloc + i]
            candle_time = analysis_df.index[entry_iloc + i]
            current_high = candle['high']
            current_low = candle['low']
            current_macd = candle['macd']
            
            # MACD 청산 조건 확인 (MACD > 0에서 진입한 경우만)
            if macd_above_zero and current_macd < 0 and not tp1_hit:
                # MACD가 0 아래로 떨어지면 청산 (TP1 도달 전)
                exit_price = candle['close']
                exit_reason = 'MACD_CROSS_BELOW_0'
                exit_time = candle_time
                macd_exit_triggered = True
                break
            
            # SL 확인 (MACD < 0에서 진입한 경우, 또는 MACD > 0이지만 TP1 도달 후)
            if current_low <= sl_price:
                if not macd_above_zero or tp1_hit:
                    exit_price = sl_price
                    exit_reason = 'SL_HIT'
                    exit_time = candle_time
                    break
            
            # TP1 확인
            if not tp1_hit and current_high >= tp1:
                tp1_hit = True
                # TP1 도달 후에는 L값 기반 SL로 전환
            
            # TP2 확인
            if tp2 and not tp2_hit and current_high >= tp2:
                tp2_hit = True
            
            # TP3 확인
            if tp3 and not tp3_hit and current_high >= tp3:
                tp3_hit = True
                exit_price = tp3
                exit_reason = 'TP3_FULL'
                exit_time = candle_time
                break
        
        # 시간 만료
        if exit_price is None:
            last_candle = analysis_df.iloc[min(entry_iloc + lookforward - 1, len(analysis_df) - 1)]
            exit_price = last_candle['close']
            exit_reason = 'TIME_STOP'
            exit_time = analysis_df.index[min(entry_iloc + lookforward - 1, len(analysis_df) - 1)]
        
        # PnL 계산 (분할 익절 시나리오)
        # 시나리오 1: 100% TP1
        if exit_reason == 'MACD_CROSS_BELOW_0':
            pnl_100_tp1 = (exit_price - entry_price) / entry_price * 100
        elif tp1_hit:
            pnl_100_tp1 = tp1_dist
        else:
            pnl_100_tp1 = (exit_price - entry_price) / entry_price * 100
        
        # 시나리오 2: 50% TP1 + 50% TP2 (또는 청산)
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
        
        # 시나리오 3: 33% TP1 + 33% TP2 + 34% TP3
        if exit_reason == 'MACD_CROSS_BELOW_0':
            pnl_33_33_34 = (exit_price - entry_price) / entry_price * 100
        elif tp1_hit and tp2_hit and tp3_hit:
            tp2_dist = (tp2 - entry_price) / entry_price * 100
            tp3_dist = (tp3 - entry_price) / entry_price * 100
            pnl_33_33_34 = 0.33 * tp1_dist + 0.33 * tp2_dist + 0.34 * tp3_dist
        elif tp1_hit and tp2_hit and tp3:
            tp2_dist = (tp2 - entry_price) / entry_price * 100
            final_pnl = (exit_price - entry_price) / entry_price * 100
            pnl_33_33_34 = 0.33 * tp1_dist + 0.33 * tp2_dist + 0.34 * final_pnl
        elif tp1_hit and tp2:
            final_pnl = (exit_price - entry_price) / entry_price * 100
            pnl_33_33_34 = 0.33 * tp1_dist + 0.33 * final_pnl + 0.34 * final_pnl
        elif tp1_hit:
            pnl_33_33_34 = tp1_dist
        else:
            pnl_33_33_34 = (exit_price - entry_price) / entry_price * 100
        
        # 조건 분류
        gap = br['break_strength'] * 100 if 'break_strength' in br else 0
        
        results.append({
            'breakout_time': breakout_time,
            'entry_price': entry_price,
            'entry_macd': entry_macd,
            'macd_above_zero': macd_above_zero,
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
            'pnl_100_tp1': pnl_100_tp1,
            'pnl_50_50': pnl_50_50,
            'pnl_33_33_34': pnl_33_33_34,
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

# 청산 사유별 통계
print("\n청산 사유별 통계:")
for reason in results_df['exit_reason'].unique():
    subset = results_df[results_df['exit_reason'] == reason]
    print(f"  {reason}: {len(subset)}건, 평균 PnL: {subset['pnl_100_tp1'].mean():.3f}%")

# 시나리오별 성과
print("\n" + "=" * 100)
print("📈 분할 익절 시나리오별 성과")
print("=" * 100)

print(f"\n시나리오 1 (100% TP1):")
print(f"  - 평균 PnL: {results_df['pnl_100_tp1'].mean():.3f}%")
print(f"  - 승률: {(results_df['pnl_100_tp1'] > 0).mean()*100:.1f}%")
print(f"  - 총 PnL: {results_df['pnl_100_tp1'].sum():.2f}%")

print(f"\n시나리오 2 (50% TP1 + 50% TP2):")
print(f"  - 평균 PnL: {results_df['pnl_50_50'].mean():.3f}%")
print(f"  - 승률: {(results_df['pnl_50_50'] > 0).mean()*100:.1f}%")
print(f"  - 총 PnL: {results_df['pnl_50_50'].sum():.2f}%")

print(f"\n시나리오 3 (33% TP1 + 33% TP2 + 34% TP3):")
print(f"  - 평균 PnL: {results_df['pnl_33_33_34'].mean():.3f}%")
print(f"  - 승률: {(results_df['pnl_33_33_34'] > 0).mean()*100:.1f}%")
print(f"  - 총 PnL: {results_df['pnl_33_33_34'].sum():.2f}%")

# MACD 조건별 분석
print("\n" + "=" * 100)
print("📊 MACD 진입 조건별 성과 분석")
print("=" * 100)

# MACD > 0 진입
macd_pos = results_df[results_df['macd_above_zero'] == True]
print(f"\n🟢 MACD > 0 진입 ({len(macd_pos)}건):")
print(f"  - 시나리오 1 (100% TP1): 평균 {macd_pos['pnl_100_tp1'].mean():.3f}%, 승률 {(macd_pos['pnl_100_tp1'] > 0).mean()*100:.1f}%")
print(f"  - 시나리오 2 (50/50): 평균 {macd_pos['pnl_50_50'].mean():.3f}%, 승률 {(macd_pos['pnl_50_50'] > 0).mean()*100:.1f}%")
print(f"  - MACD 청산 발생: {macd_pos['macd_exit'].sum()}건 ({macd_pos['macd_exit'].mean()*100:.1f}%)")

macd_pos_macd_exit = macd_pos[macd_pos['macd_exit'] == True]
if len(macd_pos_macd_exit) > 0:
    print(f"    → MACD 청산 평균 PnL: {macd_pos_macd_exit['pnl_100_tp1'].mean():.3f}%")

macd_pos_tp_hit = macd_pos[macd_pos['tp1_hit'] == True]
print(f"  - TP1 도달: {len(macd_pos_tp_hit)}건 ({len(macd_pos_tp_hit)/len(macd_pos)*100:.1f}%)")

# MACD < 0 진입
macd_neg = results_df[results_df['macd_above_zero'] == False]
print(f"\n🔴 MACD < 0 진입 ({len(macd_neg)}건):")
print(f"  - 시나리오 1 (100% TP1): 평균 {macd_neg['pnl_100_tp1'].mean():.3f}%, 승률 {(macd_neg['pnl_100_tp1'] > 0).mean()*100:.1f}%")
print(f"  - 시나리오 2 (50/50): 평균 {macd_neg['pnl_50_50'].mean():.3f}%, 승률 {(macd_neg['pnl_50_50'] > 0).mean()*100:.1f}%")
print(f"  - TP1 도달: {macd_neg['tp1_hit'].sum()}건 ({macd_neg['tp1_hit'].mean()*100:.1f}%)")

# Gap 조건별 분석
print("\n" + "=" * 100)
print("📊 Gap + MACD 복합 조건 분석")
print("=" * 100)

# Gap >= 1%
gap_1 = results_df[results_df['gap'] >= 1]
print(f"\n📈 Gap >= 1% ({len(gap_1)}건):")
print(f"  - 시나리오 1: 평균 {gap_1['pnl_100_tp1'].mean():.3f}%, 승률 {(gap_1['pnl_100_tp1'] > 0).mean()*100:.1f}%")
print(f"  - 시나리오 2: 평균 {gap_1['pnl_50_50'].mean():.3f}%, 승률 {(gap_1['pnl_50_50'] > 0).mean()*100:.1f}%")

# Gap >= 1% + MACD > 0
gap_1_macd_pos = results_df[(results_df['gap'] >= 1) & (results_df['macd_above_zero'] == True)]
print(f"\n📈 Gap >= 1% + MACD > 0 ({len(gap_1_macd_pos)}건):")
if len(gap_1_macd_pos) > 0:
    print(f"  - 시나리오 1: 평균 {gap_1_macd_pos['pnl_100_tp1'].mean():.3f}%, 승률 {(gap_1_macd_pos['pnl_100_tp1'] > 0).mean()*100:.1f}%")
    print(f"  - 시나리오 2: 평균 {gap_1_macd_pos['pnl_50_50'].mean():.3f}%, 승률 {(gap_1_macd_pos['pnl_50_50'] > 0).mean()*100:.1f}%")

# Gap >= 1% + MACD < 0
gap_1_macd_neg = results_df[(results_df['gap'] >= 1) & (results_df['macd_above_zero'] == False)]
print(f"\n📈 Gap >= 1% + MACD < 0 ({len(gap_1_macd_neg)}건):")
if len(gap_1_macd_neg) > 0:
    print(f"  - 시나리오 1: 평균 {gap_1_macd_neg['pnl_100_tp1'].mean():.3f}%, 승률 {(gap_1_macd_neg['pnl_100_tp1'] > 0).mean()*100:.1f}%")
    print(f"  - 시나리오 2: 평균 {gap_1_macd_neg['pnl_50_50'].mean():.3f}%, 승률 {(gap_1_macd_neg['pnl_50_50'] > 0).mean()*100:.1f}%")

# 최적 조건 분석
print("\n" + "=" * 100)
print("🏆 최적 조건 분석")
print("=" * 100)

# 승률 74% 목표 달성 조건 찾기
conditions = [
    ('전체', results_df),
    ('MACD > 0', macd_pos),
    ('MACD < 0', macd_neg),
    ('Gap >= 1%', gap_1),
    ('Gap >= 1% + MACD > 0', gap_1_macd_pos),
    ('Gap >= 1% + MACD < 0', gap_1_macd_neg),
]

print(f"\n{'조건':<25} {'건수':>8} {'승률':>10} {'평균PnL':>10} {'TP1도달':>10}")
print("-" * 70)
for name, df in conditions:
    if len(df) > 0:
        win_rate = (df['pnl_50_50'] > 0).mean() * 100
        avg_pnl = df['pnl_50_50'].mean()
        tp1_rate = df['tp1_hit'].mean() * 100
        print(f"{name:<25} {len(df):>8} {win_rate:>9.1f}% {avg_pnl:>9.3f}% {tp1_rate:>9.1f}%")

# 결과 저장
results_df.to_csv('macd_sl_analysis_results.csv', index=False)
print(f"\n✅ 결과 저장: macd_sl_analysis_results.csv")

# 핵심 인사이트
print("\n" + "=" * 100)
print("💡 핵심 인사이트")
print("=" * 100)

print(f"""
📌 MACD 기반 SL 전략 요약:

1. 진입 시 MACD > 0 ({len(macd_pos)}건, {len(macd_pos)/len(results_df)*100:.1f}%):
   - MACD가 0 아래로 떨어지면 즉시 청산
   - MACD 청산 발생률: {macd_pos['macd_exit'].mean()*100:.1f}%
   - 평균 PnL: {macd_pos['pnl_50_50'].mean():.3f}%

2. 진입 시 MACD < 0 ({len(macd_neg)}건, {len(macd_neg)/len(results_df)*100:.1f}%):
   - 기존 L값 기반 SL 사용
   - TP1 도달률: {macd_neg['tp1_hit'].mean()*100:.1f}%
   - 평균 PnL: {macd_neg['pnl_50_50'].mean():.3f}%

3. 분할 익절 효과:
   - 100% TP1: 평균 {results_df['pnl_100_tp1'].mean():.3f}%
   - 50/50 분할: 평균 {results_df['pnl_50_50'].mean():.3f}%
   - 33/33/34 분할: 평균 {results_df['pnl_33_33_34'].mean():.3f}%
""")
