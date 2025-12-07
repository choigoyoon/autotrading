import pandas as pd
import numpy as np

df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

print("="*80)
print("🔍 미래 데이터 사용 여부 체크")
print("="*80)

print("\n[현재 로직 분석]")
print("-"*40)

print("""
1. FVG 탐지 (4H 기준)
   - 사용 데이터: i-2, i-1, i 봉 (완성된 봉)
   - 판정 시점: i번째 4H봉 마감 후
   ✅ 미래 데이터 없음

2. 리테스트 탐지 (15분봉)
   - FVG 발생 후 15분봉에서 FVG 상단 터치 확인
   - entry_price = 터치한 봉의 close
   ⚠️ 문제: 터치 봉의 close는 해당 봉 마감 후에만 알 수 있음
   → 실제로는 터치 시점에 진입해야 함

3. 진입 조건: dist_from_fvg >= 0.2%
   - entry_price가 FVG 상단 대비 0.2% 위인지 확인
   ⚠️ 문제: entry_price(close) 기준이므로 봉 마감 후 판단
   → 실시간에서는 알 수 없음

4. TP/SL 판정
   - 진입 후 100봉(25시간) 동안 max_profit, max_loss 계산
   ✅ 미래 데이터 아님 (진입 후 결과)
""")

print("\n[문제점 상세]")
print("-"*40)
print("""
핵심 문제: "FVG 위 0.2%+"는 봉 마감 후에만 알 수 있음

예시:
- FVG 상단 = 50000
- 0.2% 위 = 50100
- 15분봉이 50100에서 마감해야 "FVG 위 0.2%+" 확인 가능
- 하지만 실제 매매에서는 50000 터치 시점에 진입 결정해야 함

즉, "FVG 위 0.2%+"는 미래 데이터!
""")

print("\n" + "="*80)
print("🔧 수정된 로직으로 재테스트")
print("="*80)

# FVG 탐지
def detect_4h_fvg(df):
    fvgs = []
    for i in range(2, len(df)):
        prev2, prev1, curr = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        if prev2['high'] < curr['low'] and curr['close'] > curr['open']:
            gap_size = (curr['low'] - prev2['high']) / prev2['high'] * 100
            body_size = abs(curr['close'] - curr['open']) / curr['open'] * 100
            fvgs.append({
                'datetime': curr['datetime'],
                'fvg_top': curr['low'],
                'fvg_bottom': prev2['high'],
                'fvg_size': gap_size,
                'body_size': body_size,
            })
    return fvgs

fvgs_4h = detect_4h_fvg(df_4h)
print(f"\n4H FVG 개수: {len(fvgs_4h)}")

# 수정된 진입 로직
results_old = []  # 기존 (미래 데이터 사용)
results_new = []  # 수정 (실시간 가능)

for fvg in fvgs_4h:
    fvg_time = fvg['datetime']
    fvg_top = fvg['fvg_top']
    fvg_bottom = fvg['fvg_bottom']
    body_size = fvg['body_size']
    
    mask = df_15m['datetime'] > fvg_time
    future_15m = df_15m[mask].head(200)
    
    if len(future_15m) < 50:
        continue
    
    # 기존 로직: FVG 터치 후 close 기준 판단
    for idx, row in future_15m.iterrows():
        if row['low'] <= fvg_top:  # FVG 상단 터치
            entry_price_old = row['close']  # 봉 마감 가격 (미래 데이터!)
            dist_from_fvg_old = (entry_price_old - fvg_top) / fvg_top * 100
            
            entry_loc = df_15m.index.get_loc(idx)
            post_entry = df_15m.iloc[entry_loc+1:entry_loc+101]
            if len(post_entry) < 50:
                break
            
            max_profit = (post_entry['high'].max() - entry_price_old) / entry_price_old * 100
            max_loss = (post_entry['low'].min() - entry_price_old) / entry_price_old * 100
            
            results_old.append({
                'body_size': body_size,
                'dist_from_fvg': dist_from_fvg_old,
                'max_profit': max_profit,
                'max_loss': max_loss,
            })
            break
    
    # 수정된 로직: FVG 터치 시 다음봉 시가로 진입
    for i, (idx, row) in enumerate(future_15m.iterrows()):
        if row['low'] <= fvg_top:  # FVG 상단 터치
            # 다음 봉 시가로 진입 (실시간 가능)
            remaining = future_15m.iloc[i+1:]
            if len(remaining) < 50:
                break
            
            entry_price_new = remaining.iloc[0]['open']  # 다음봉 시가
            dist_from_fvg_new = (entry_price_new - fvg_top) / fvg_top * 100
            
            post_entry = remaining.iloc[1:101]
            if len(post_entry) < 50:
                break
            
            max_profit = (post_entry['high'].max() - entry_price_new) / entry_price_new * 100
            max_loss = (post_entry['low'].min() - entry_price_new) / entry_price_new * 100
            
            results_new.append({
                'body_size': body_size,
                'dist_from_fvg': dist_from_fvg_new,
                'max_profit': max_profit,
                'max_loss': max_loss,
            })
            break

df_old = pd.DataFrame(results_old)
df_new = pd.DataFrame(results_new)

print(f"\n기존 로직 데이터: {len(df_old)}건")
print(f"수정 로직 데이터: {len(df_new)}건")

def calc_stats(df, tp, sl):
    wins = sum(df['max_profit'] >= tp)
    losses = sum((df['max_profit'] < tp) & (df['max_loss'] <= sl))
    total = wins + losses
    if total == 0:
        return 0, 0, 0, 0, 0, 0, 0
    wr = wins/total*100
    pnl = wins * tp + losses * abs(sl) * -1
    ev = (wr/100) * tp - ((100-wr)/100) * abs(sl)
    monthly = total / 60
    return wins, losses, total, wr, pnl, ev, monthly

print("\n" + "="*80)
print("📊 기존 로직 vs 수정 로직 비교")
print("="*80)

print("\n[전략 B: FVG 위 0.2%+ 조건]")
print("-"*40)

# 기존 로직
cond_old = df_old['dist_from_fvg'] >= 0.2
subset_old = df_old[cond_old]
print(f"\n기존 로직 (미래 데이터 사용):")
print(f"  조건 충족: {len(subset_old)}건")
for tp in [0.5, 1.0, 1.5]:
    wins, losses, total, wr, pnl, ev, monthly = calc_stats(subset_old, tp, -0.5)
    print(f"  TP {tp}%: {total}건, 승률 {wr:.1f}%, EV {ev:.3f}%")

# 수정 로직
cond_new = df_new['dist_from_fvg'] >= 0.2
subset_new = df_new[cond_new]
print(f"\n수정 로직 (실시간 가능):")
print(f"  조건 충족: {len(subset_new)}건")
for tp in [0.5, 1.0, 1.5]:
    wins, losses, total, wr, pnl, ev, monthly = calc_stats(subset_new, tp, -0.5)
    print(f"  TP {tp}%: {total}건, 승률 {wr:.1f}%, EV {ev:.3f}%")

print("\n[전체 데이터 - 조건 없이]")
print("-"*40)

print(f"\n기존 로직:")
for tp in [0.5, 1.0, 1.5]:
    wins, losses, total, wr, pnl, ev, monthly = calc_stats(df_old, tp, -0.5)
    print(f"  TP {tp}%: {total}건, 승률 {wr:.1f}%, EV {ev:.3f}%")

print(f"\n수정 로직:")
for tp in [0.5, 1.0, 1.5]:
    wins, losses, total, wr, pnl, ev, monthly = calc_stats(df_new, tp, -0.5)
    print(f"  TP {tp}%: {total}건, 승률 {wr:.1f}%, EV {ev:.3f}%")

print("\n" + "="*80)
print("⚠️ 결론")
print("="*80)
print("""
'FVG 위 0.2%+' 조건은 미래 데이터 문제가 있음!

수정 방안:
1. 조건 제거: FVG 터치 시 바로 진입
2. 조건 변경: FVG 터치 후 "다음봉 시가"가 FVG 위인지 확인
3. 지정가 주문: FVG 상단 + 0.2% 위치에 지정가 매수

실시간 적용 가능한 로직으로 재검증 필요!
""")

