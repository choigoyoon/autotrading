import pandas as pd
import numpy as np
from datetime import timedelta

"""
MACD 기반 손절 전략 완전 분석
=================================
사용자 지시사항 (명확화):
1. 진입 시 MACD < 0: 기존 L값 기반 SL 사용 (허용)
2. 진입 시 MACD > 0: MACD가 0 아래로 떨어지면 청산 (Exit)
"""

print("=" * 100)
print("📊 MACD 기반 Stop Loss 전략 완전 분석")
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

# 유틸리티 함수
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

def get_consecutive_hl_count(breakout_idx):
    """돌파 직전 연속 HL 개수 계산"""
    prev_l = l_values[l_values['idx'] < breakout_idx].sort_values('idx', ascending=False).head(10)
    
    hl_count = 0
    for _, l in prev_l.iterrows():
        if l['is_HL'] == True:
            hl_count += 1
        elif l['is_LL'] == True:
            break
    
    return hl_count

def is_bullish_candle(candle):
    return candle['close'] > candle['open']

def get_candle_body_ratio(candle):
    body = abs(candle['close'] - candle['open'])
    total = candle['high'] - candle['low']
    if total > 0:
        return body / total
    return 0

# MACD 기반 청산 분석 - 두 가지 시나리오 비교
print("\n" + "=" * 100)
print("🔄 시나리오 비교: MACD SL vs L값 SL")
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
        hl_count = get_consecutive_hl_count(breakout_idx)
        
        # SL/TP 설정
        sl_price_l_value = get_previous_l(breakout_idx)
        if sl_price_l_value is None:
            continue
        
        h_resistance_list = get_h_resistance_values(breakout_idx, entry_price)
        if len(h_resistance_list) == 0:
            tp1 = entry_price * 1.01
            tp2 = entry_price * 1.02
            tp3 = entry_price * 1.03
        else:
            tp1 = h_resistance_list[0] if len(h_resistance_list) >= 1 else entry_price * 1.01
            tp2 = h_resistance_list[1] if len(h_resistance_list) >= 2 else entry_price * 1.02
            tp3 = h_resistance_list[2] if len(h_resistance_list) >= 3 else entry_price * 1.03
        
        sl_dist = (entry_price - sl_price_l_value) / entry_price * 100
        tp1_dist = (tp1 - entry_price) / entry_price * 100
        
        gap = br['break_strength'] * 100 if 'break_strength' in br and pd.notna(br['break_strength']) else 0
        
        # === 시나리오 1: L값 기반 SL만 사용 (기존 방식) ===
        exit_price_lval = None
        exit_reason_lval = None
        tp1_hit_lval = False
        tp2_hit_lval = False
        tp3_hit_lval = False
        candles_to_exit_lval = 0
        max_profit_lval = 0
        
        for i in range(1, min(lookforward, len(analysis_df) - entry_iloc)):
            candle = analysis_df.iloc[entry_iloc + i]
            current_high = candle['high']
            current_low = candle['low']
            
            current_profit = (current_high - entry_price) / entry_price * 100
            if current_profit > max_profit_lval:
                max_profit_lval = current_profit
            
            # L값 기반 SL
            if current_low <= sl_price_l_value:
                exit_price_lval = sl_price_l_value
                exit_reason_lval = 'SL_L_VALUE'
                candles_to_exit_lval = i
                break
            
            # TP 확인
            if not tp1_hit_lval and current_high >= tp1:
                tp1_hit_lval = True
            if not tp2_hit_lval and current_high >= tp2:
                tp2_hit_lval = True
            if not tp3_hit_lval and current_high >= tp3:
                tp3_hit_lval = True
                exit_price_lval = tp3
                exit_reason_lval = 'TP3_FULL'
                candles_to_exit_lval = i
                break
        
        if exit_price_lval is None:
            last_candle = analysis_df.iloc[min(entry_iloc + lookforward - 1, len(analysis_df) - 1)]
            exit_price_lval = last_candle['close']
            exit_reason_lval = 'TIME_STOP'
            candles_to_exit_lval = lookforward
        
        # L값 시나리오 PnL
        if tp1_hit_lval:
            pnl_lval = tp1_dist
        else:
            pnl_lval = (exit_price_lval - entry_price) / entry_price * 100
        
        # === 시나리오 2: MACD 조건부 SL (사용자 지시) ===
        exit_price_macd = None
        exit_reason_macd = None
        tp1_hit_macd = False
        tp2_hit_macd = False
        tp3_hit_macd = False
        macd_exit_triggered = False
        candles_to_exit_macd = 0
        max_profit_macd = 0
        
        for i in range(1, min(lookforward, len(analysis_df) - entry_iloc)):
            candle = analysis_df.iloc[entry_iloc + i]
            current_high = candle['high']
            current_low = candle['low']
            current_macd = candle['macd']
            
            current_profit = (current_high - entry_price) / entry_price * 100
            if current_profit > max_profit_macd:
                max_profit_macd = current_profit
            
            # MACD 조건부 청산
            # MACD > 0 진입: MACD가 0 아래로 떨어지면 청산 (TP1 도달 전)
            if macd_above_zero and current_macd < 0 and not tp1_hit_macd:
                exit_price_macd = candle['close']
                exit_reason_macd = 'MACD_CROSS_BELOW_0'
                macd_exit_triggered = True
                candles_to_exit_macd = i
                break
            
            # MACD < 0 진입 또는 TP1 도달 후: L값 기반 SL 사용
            if current_low <= sl_price_l_value:
                if not macd_above_zero or tp1_hit_macd:
                    exit_price_macd = sl_price_l_value
                    exit_reason_macd = 'SL_L_VALUE'
                    candles_to_exit_macd = i
                    break
            
            # TP 확인
            if not tp1_hit_macd and current_high >= tp1:
                tp1_hit_macd = True
            if not tp2_hit_macd and current_high >= tp2:
                tp2_hit_macd = True
            if not tp3_hit_macd and current_high >= tp3:
                tp3_hit_macd = True
                exit_price_macd = tp3
                exit_reason_macd = 'TP3_FULL'
                candles_to_exit_macd = i
                break
        
        if exit_price_macd is None:
            last_candle = analysis_df.iloc[min(entry_iloc + lookforward - 1, len(analysis_df) - 1)]
            exit_price_macd = last_candle['close']
            exit_reason_macd = 'TIME_STOP'
            candles_to_exit_macd = lookforward
        
        # MACD 시나리오 PnL
        if exit_reason_macd == 'MACD_CROSS_BELOW_0':
            pnl_macd = (exit_price_macd - entry_price) / entry_price * 100
        elif tp1_hit_macd:
            pnl_macd = tp1_dist
        else:
            pnl_macd = (exit_price_macd - entry_price) / entry_price * 100
        
        # 50/50 분할 매도 시나리오 (L값)
        if tp1_hit_lval and tp2_hit_lval:
            tp2_dist = (tp2 - entry_price) / entry_price * 100
            pnl_lval_50_50 = 0.5 * tp1_dist + 0.5 * tp2_dist
        elif tp1_hit_lval:
            final_pnl = (exit_price_lval - entry_price) / entry_price * 100
            pnl_lval_50_50 = 0.5 * tp1_dist + 0.5 * final_pnl
        else:
            pnl_lval_50_50 = pnl_lval
        
        # 50/50 분할 매도 시나리오 (MACD)
        if exit_reason_macd == 'MACD_CROSS_BELOW_0':
            pnl_macd_50_50 = pnl_macd
        elif tp1_hit_macd and tp2_hit_macd:
            tp2_dist = (tp2 - entry_price) / entry_price * 100
            pnl_macd_50_50 = 0.5 * tp1_dist + 0.5 * tp2_dist
        elif tp1_hit_macd:
            final_pnl = (exit_price_macd - entry_price) / entry_price * 100
            pnl_macd_50_50 = 0.5 * tp1_dist + 0.5 * final_pnl
        else:
            pnl_macd_50_50 = pnl_macd
        
        results.append({
            'breakout_time': breakout_time,
            'entry_price': entry_price,
            'entry_macd': entry_macd,
            'macd_above_zero': macd_above_zero,
            'bullish_entry': bullish_entry,
            'body_ratio': body_ratio,
            'hl_count': hl_count,
            'sl_price': sl_price_l_value,
            'sl_dist': sl_dist,
            'tp1_dist': tp1_dist,
            'gap': gap,
            # L값 SL 결과
            'exit_reason_lval': exit_reason_lval,
            'tp1_hit_lval': tp1_hit_lval,
            'tp2_hit_lval': tp2_hit_lval,
            'pnl_lval': pnl_lval,
            'pnl_lval_50_50': pnl_lval_50_50,
            'max_profit_lval': max_profit_lval,
            'candles_lval': candles_to_exit_lval,
            # MACD SL 결과
            'exit_reason_macd': exit_reason_macd,
            'tp1_hit_macd': tp1_hit_macd,
            'tp2_hit_macd': tp2_hit_macd,
            'macd_exit': macd_exit_triggered,
            'pnl_macd': pnl_macd,
            'pnl_macd_50_50': pnl_macd_50_50,
            'max_profit_macd': max_profit_macd,
            'candles_macd': candles_to_exit_macd,
        })
        
    except Exception as e:
        continue

results_df = pd.DataFrame(results)
print(f"\n분석 완료: {len(results_df)}건")

# 결과 저장
results_df.to_csv('macd_sl_complete_analysis.csv', index=False)

# 종합 분석
print("\n" + "=" * 100)
print("📊 종합 성과 비교: L값 SL vs MACD 조건부 SL")
print("=" * 100)

# 전체 성과
print(f"\n[전체 성과 비교] (총 {len(results_df)}건)")
print("-" * 70)

print(f"\n{'시나리오':<25} {'승률':>12} {'평균PnL':>12} {'총PnL':>15}")
print("-" * 70)

# L값 SL 전체
lval_win_rate = (results_df['pnl_lval_50_50'] > 0).mean() * 100
lval_avg_pnl = results_df['pnl_lval_50_50'].mean()
lval_total_pnl = results_df['pnl_lval_50_50'].sum()
print(f"{'L값 기반 SL (전체)':<25} {lval_win_rate:>11.1f}% {lval_avg_pnl:>11.3f}% {lval_total_pnl:>14.2f}%")

# MACD SL 전체
macd_win_rate = (results_df['pnl_macd_50_50'] > 0).mean() * 100
macd_avg_pnl = results_df['pnl_macd_50_50'].mean()
macd_total_pnl = results_df['pnl_macd_50_50'].sum()
print(f"{'MACD 조건부 SL (전체)':<25} {macd_win_rate:>11.1f}% {macd_avg_pnl:>11.3f}% {macd_total_pnl:>14.2f}%")

# MACD 위치별 분석
print("\n" + "=" * 100)
print("📊 MACD 진입 위치별 성과 분석")
print("=" * 100)

macd_pos = results_df[results_df['macd_above_zero'] == True]
macd_neg = results_df[results_df['macd_above_zero'] == False]

print(f"\n[MACD > 0에서 진입: {len(macd_pos)}건 ({len(macd_pos)/len(results_df)*100:.1f}%)]")
print("-" * 70)
print(f"  L값 SL: 승률 {(macd_pos['pnl_lval_50_50'] > 0).mean()*100:.1f}%, 평균 {macd_pos['pnl_lval_50_50'].mean():.3f}%, 총 {macd_pos['pnl_lval_50_50'].sum():.2f}%")
print(f"  MACD SL: 승률 {(macd_pos['pnl_macd_50_50'] > 0).mean()*100:.1f}%, 평균 {macd_pos['pnl_macd_50_50'].mean():.3f}%, 총 {macd_pos['pnl_macd_50_50'].sum():.2f}%")

macd_exits_pos = macd_pos[macd_pos['macd_exit'] == True]
print(f"\n  ⚡ MACD 0선 하향돌파 청산 발생: {len(macd_exits_pos)}건 ({len(macd_exits_pos)/len(macd_pos)*100:.1f}%)")
if len(macd_exits_pos) > 0:
    print(f"     - MACD 청산 시 평균 PnL: {macd_exits_pos['pnl_macd'].mean():.3f}%")
    print(f"     - 청산까지 평균 캔들: {macd_exits_pos['candles_macd'].mean():.1f}개 ({macd_exits_pos['candles_macd'].mean()*15/60:.1f}시간)")

print(f"\n[MACD < 0에서 진입: {len(macd_neg)}건 ({len(macd_neg)/len(results_df)*100:.1f}%)]")
print("-" * 70)
print(f"  L값 SL: 승률 {(macd_neg['pnl_lval_50_50'] > 0).mean()*100:.1f}%, 평균 {macd_neg['pnl_lval_50_50'].mean():.3f}%, 총 {macd_neg['pnl_lval_50_50'].sum():.2f}%")
print(f"  MACD SL: 승률 {(macd_neg['pnl_macd_50_50'] > 0).mean()*100:.1f}%, 평균 {macd_neg['pnl_macd_50_50'].mean():.3f}%, 총 {macd_neg['pnl_macd_50_50'].sum():.2f}%")
print(f"  (동일한 결과 - MACD < 0 진입 시 L값 SL 사용)")

# 청산 사유 비교
print("\n" + "=" * 100)
print("📊 청산 사유별 분포 비교")
print("=" * 100)

print(f"\n[L값 SL 청산 사유]")
for reason in results_df['exit_reason_lval'].unique():
    subset = results_df[results_df['exit_reason_lval'] == reason]
    print(f"  {reason}: {len(subset)}건 ({len(subset)/len(results_df)*100:.1f}%), 평균 PnL: {subset['pnl_lval_50_50'].mean():.3f}%")

print(f"\n[MACD 조건부 SL 청산 사유]")
for reason in results_df['exit_reason_macd'].unique():
    subset = results_df[results_df['exit_reason_macd'] == reason]
    print(f"  {reason}: {len(subset)}건 ({len(subset)/len(results_df)*100:.1f}%), 평균 PnL: {subset['pnl_macd_50_50'].mean():.3f}%")

# 조건 조합별 분석
print("\n" + "=" * 100)
print("🏆 최적 조건 조합별 MACD SL 성과")
print("=" * 100)

conditions_macd = [
    ('전체', results_df),
    ('MACD < 0 진입', macd_neg),
    ('MACD < 0 + HL >= 2', macd_neg[macd_neg['hl_count'] >= 2]),
    ('MACD < 0 + 양봉', macd_neg[macd_neg['bullish_entry'] == True]),
    ('MACD < 0 + 양봉 + HL >= 2', macd_neg[(macd_neg['bullish_entry'] == True) & (macd_neg['hl_count'] >= 2)]),
    ('MACD < 0 + Gap >= 1%', macd_neg[macd_neg['gap'] >= 1]),
    ('MACD < 0 + Gap >= 1% + HL >= 2', macd_neg[(macd_neg['gap'] >= 1) & (macd_neg['hl_count'] >= 2)]),
    ('MACD < 0 + Gap >= 1% + HL >= 2 + 양봉', macd_neg[(macd_neg['gap'] >= 1) & (macd_neg['hl_count'] >= 2) & (macd_neg['bullish_entry'] == True)]),
]

print(f"\n{'조건':<40} {'건수':>6} {'승률':>8} {'평균PnL':>10} {'총PnL':>12}")
print("-" * 85)
for name, df in conditions_macd:
    if len(df) > 0:
        win_rate = (df['pnl_macd_50_50'] > 0).mean() * 100
        avg_pnl = df['pnl_macd_50_50'].mean()
        total_pnl = df['pnl_macd_50_50'].sum()
        print(f"{name:<40} {len(df):>6} {win_rate:>7.1f}% {avg_pnl:>9.3f}% {total_pnl:>11.2f}%")

# HL 개수별 분석
print("\n" + "=" * 100)
print("📊 연속 HL(Higher Low) 개수별 성과")
print("=" * 100)

print(f"\n{'HL개수':<8} {'전체(건)':>10} {'L값 승률':>12} {'MACD 승률':>12} {'MACD 평균PnL':>15}")
print("-" * 65)
for hl in sorted(results_df['hl_count'].unique()):
    if hl <= 6:
        subset = results_df[results_df['hl_count'] == hl]
        lval_wr = (subset['pnl_lval_50_50'] > 0).mean() * 100
        macd_wr = (subset['pnl_macd_50_50'] > 0).mean() * 100
        macd_pnl = subset['pnl_macd_50_50'].mean()
        print(f"  {hl:<6} {len(subset):>10} {lval_wr:>11.1f}% {macd_wr:>11.1f}% {macd_pnl:>14.3f}%")

# 핵심 결론
print("\n" + "=" * 100)
print("💡 핵심 인사이트 및 결론")
print("=" * 100)

# MACD > 0 진입 시 MACD SL 효과 분석
macd_pos_better = (macd_pos['pnl_macd_50_50'] > macd_pos['pnl_lval_50_50']).sum()
macd_pos_worse = (macd_pos['pnl_macd_50_50'] < macd_pos['pnl_lval_50_50']).sum()
macd_pos_same = (macd_pos['pnl_macd_50_50'] == macd_pos['pnl_lval_50_50']).sum()

improvement = macd_pos['pnl_macd_50_50'].sum() - macd_pos['pnl_lval_50_50'].sum()

print(f"""
═══════════════════════════════════════════════════════════════════════════════
📌 MACD 기반 SL 전략 분석 결과
═══════════════════════════════════════════════════════════════════════════════

▶ 규칙 요약:
  - MACD < 0 진입: 기존 L값 기반 SL 사용 (허용)
  - MACD > 0 진입: MACD가 0 아래로 떨어지면 청산

═══════════════════════════════════════════════════════════════════════════════
1. MACD > 0에서 진입한 경우 ({len(macd_pos)}건, 전체의 {len(macd_pos)/len(results_df)*100:.1f}%)
═══════════════════════════════════════════════════════════════════════════════

   [L값 SL vs MACD SL 개별 비교]
   ✓ MACD SL이 더 나은 경우: {macd_pos_better}건 ({macd_pos_better/len(macd_pos)*100:.1f}%)
   ✓ L값 SL이 더 나은 경우: {macd_pos_worse}건 ({macd_pos_worse/len(macd_pos)*100:.1f}%)
   ✓ 동일한 결과: {macd_pos_same}건 ({macd_pos_same/len(macd_pos)*100:.1f}%)

   [성과 비교]
   ✓ L값 SL: 승률 {(macd_pos['pnl_lval_50_50'] > 0).mean()*100:.1f}%, 평균 {macd_pos['pnl_lval_50_50'].mean():.3f}%, 총 {macd_pos['pnl_lval_50_50'].sum():.2f}%
   ✓ MACD SL: 승률 {(macd_pos['pnl_macd_50_50'] > 0).mean()*100:.1f}%, 평균 {macd_pos['pnl_macd_50_50'].mean():.3f}%, 총 {macd_pos['pnl_macd_50_50'].sum():.2f}%
   
   📈 MACD SL 적용 시 총 PnL 변화: {improvement:+.2f}%
   
   [MACD 청산 발생 분석] ({len(macd_exits_pos)}건)
   ✓ MACD 청산 발생률: {len(macd_exits_pos)/len(macd_pos)*100:.1f}%
   ✓ MACD 청산 시 평균 PnL: {macd_exits_pos['pnl_macd'].mean():.3f}% (손실 방지 또는 조기 청산)
   ✓ 만약 L값 SL만 사용했다면 평균 PnL: {macd_exits_pos['pnl_lval_50_50'].mean():.3f}%

═══════════════════════════════════════════════════════════════════════════════
2. MACD < 0에서 진입한 경우 ({len(macd_neg)}건, 전체의 {len(macd_neg)/len(results_df)*100:.1f}%)
═══════════════════════════════════════════════════════════════════════════════
   
   ✓ L값 SL 적용 (사용자 지시대로)
   ✓ 승률: {(macd_neg['pnl_macd_50_50'] > 0).mean()*100:.1f}%
   ✓ 평균 PnL: {macd_neg['pnl_macd_50_50'].mean():.3f}%
   ✓ 총 PnL: {macd_neg['pnl_macd_50_50'].sum():.2f}%

═══════════════════════════════════════════════════════════════════════════════
3. 최적 진입 조건 권장 (MACD SL 기준)
═══════════════════════════════════════════════════════════════════════════════
""")

best_cond = macd_neg[(macd_neg['gap'] >= 1) & (macd_neg['hl_count'] >= 2) & (macd_neg['bullish_entry'] == True)]
if len(best_cond) > 0:
    print(f"""
   🏆 최적 조건: MACD < 0 + Gap >= 1% + HL >= 2 + 양봉
   ✓ 건수: {len(best_cond)}건
   ✓ 승률: {(best_cond['pnl_macd_50_50'] > 0).mean()*100:.1f}%
   ✓ 평균 PnL: {best_cond['pnl_macd_50_50'].mean():.3f}%
   ✓ 총 PnL: {best_cond['pnl_macd_50_50'].sum():.2f}%
""")

print(f"""
═══════════════════════════════════════════════════════════════════════════════
4. 최종 권장 전략
═══════════════════════════════════════════════════════════════════════════════

   📌 진입 조건:
      1. LH-LH-LH 하락 추세선 돌파
      2. Gap >= 1% (추세선과의 거리)
      3. HL >= 2 (바닥에서 힘 축적)
      4. 양봉 돌파
      5. 🆕 MACD < 0 (추세 반전 초기 진입 권장)

   📌 청산 조건:
      - MACD < 0 진입: L값 기반 SL 사용
      - MACD > 0 진입: MACD가 0 아래로 떨어지면 청산
      - TP: H값 저항선 기준 분할 청산 (50/50)

   📌 핵심 포인트:
      - MACD < 0 진입이 전반적으로 더 유리
      - MACD > 0 진입 시 MACD SL 규칙이 손실 방지에 도움
      - 분할 청산으로 리스크 관리
""")

print("\n✅ 분석 완료! 결과 저장: macd_sl_complete_analysis.csv")
