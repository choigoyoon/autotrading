import pandas as pd
import numpy as np

df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])

print("="*80)
print("🔧 수정된 로직 - TP/SL 순서 체크 + 터치 방향 구분")
print("="*80)

# FVG 탐지
def detect_4h_fvg(df):
    fvgs = []
    for i in range(2, len(df)):
        prev2, prev1, curr = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
        if prev2['high'] < curr['low'] and curr['close'] > curr['open']:
            fvgs.append({
                'datetime': curr['datetime'],
                'fvg_top': curr['low'],
                'fvg_bottom': prev2['high'],
                'body_size': abs(curr['close'] - curr['open']) / curr['open'] * 100,
            })
    return fvgs

fvgs_4h = detect_4h_fvg(df_4h)
print(f"4H FVG 개수: {len(fvgs_4h)}")

# 수정된 진입 로직
results = []

for fvg in fvgs_4h:
    fvg_time = fvg['datetime']
    fvg_top = fvg['fvg_top']
    body_size = fvg['body_size']
    
    mask = df_15m['datetime'] > fvg_time
    future_15m = df_15m[mask].head(200)
    
    if len(future_15m) < 50:
        continue
    
    # 터치 방향 체크를 위한 이전 가격
    prev_close = None
    
    for i, (idx, row) in enumerate(future_15m.iterrows()):
        # 첫 봉은 이전 가격 저장만
        if prev_close is None:
            prev_close = row['close']
            continue
        
        # FVG 상단 터치 체크
        if row['low'] <= fvg_top:
            # 터치 방향: 위에서 내려옴 vs 아래에서 올라옴
            touch_direction = 'from_above' if prev_close > fvg_top else 'from_below'
            
            # 다음봉 시가로 진입
            remaining = future_15m.iloc[i+1:]
            if len(remaining) < 50:
                break
            
            entry_price = remaining.iloc[0]['open']
            dist_from_fvg = (entry_price - fvg_top) / fvg_top * 100
            
            # ✅ TP/SL 순서 체크 (봉 단위로)
            tp_pct = 1.5
            sl_pct = -0.5
            
            result = None
            exit_bar = 0
            
            for j, (_, bar) in enumerate(remaining.iloc[1:101].iterrows()):
                high_pct = (bar['high'] - entry_price) / entry_price * 100
                low_pct = (bar['low'] - entry_price) / entry_price * 100
                
                # 같은 봉에서 TP/SL 둘 다 도달할 수 있음
                # → 시가 기준으로 방향 판단
                open_pct = (bar['open'] - entry_price) / entry_price * 100
                
                if high_pct >= tp_pct and low_pct <= sl_pct:
                    # 둘 다 도달 - 시가 기준으로 먼저 도달한 쪽 판단
                    if open_pct >= 0:  # 시가가 위쪽이면 TP 먼저 도달 가능성
                        result = 'WIN'
                    else:  # 시가가 아래쪽이면 SL 먼저 도달 가능성
                        result = 'LOSS'
                    exit_bar = j + 1
                    break
                elif high_pct >= tp_pct:
                    result = 'WIN'
                    exit_bar = j + 1
                    break
                elif low_pct <= sl_pct:
                    result = 'LOSS'
                    exit_bar = j + 1
                    break
            
            if result is None:
                result = 'TIMEOUT'
            
            results.append({
                'body_size': body_size,
                'dist_from_fvg': dist_from_fvg,
                'touch_direction': touch_direction,
                'result': result,
                'exit_bar': exit_bar,
            })
            break
        
        prev_close = row['close']

df_results = pd.DataFrame(results)
print(f"총 데이터: {len(df_results)}건")

# 결과 분석
print("\n" + "="*80)
print("📊 결과 분석 (TP 1.5%, SL -0.5%)")
print("="*80)

def analyze(df, name):
    total = len(df)
    if total == 0:
        print(f"\n{name}: 데이터 없음")
        return
    
    wins = len(df[df['result'] == 'WIN'])
    losses = len(df[df['result'] == 'LOSS'])
    timeouts = len(df[df['result'] == 'TIMEOUT'])
    
    decided = wins + losses
    if decided == 0:
        print(f"\n{name}: 승패 결정 없음")
        return
    
    wr = wins / decided * 100
    pnl = wins * 1.5 + losses * (-0.5)
    ev = (wr/100) * 1.5 - ((100-wr)/100) * 0.5
    monthly = decided / 60
    
    print(f"\n{name}:")
    print(f"  총: {total}건 (승: {wins}, 패: {losses}, 타임아웃: {timeouts})")
    print(f"  승률: {wr:.1f}% ({wins}/{decided})")
    print(f"  EV: {ev:.3f}%")
    print(f"  5년 PnL: {pnl:.0f}%")
    print(f"  월평균: {monthly:.1f}회")

# 전체
analyze(df_results, "전체")

# 터치 방향별
analyze(df_results[df_results['touch_direction'] == 'from_above'], "위에서 터치")
analyze(df_results[df_results['touch_direction'] == 'from_below'], "아래에서 터치")

# FVG 위 조건별
analyze(df_results[df_results['dist_from_fvg'] >= 0.2], "FVG 위 0.2%+")
analyze(df_results[df_results['dist_from_fvg'] >= 0.1], "FVG 위 0.1%+")
analyze(df_results[df_results['dist_from_fvg'] >= 0], "FVG 위 (모든)")

# 4H 몸통 크기별
analyze(df_results[df_results['body_size'] >= 2], "4H 몸통 2%+")
analyze(df_results[df_results['body_size'] >= 3], "4H 몸통 3%+")

# 조합
cond_best = (df_results['dist_from_fvg'] >= 0.1) & (df_results['touch_direction'] == 'from_above')
analyze(df_results[cond_best], "FVG 위 0.1%+ & 위에서 터치")

print("\n" + "="*80)
print("📈 다양한 TP/SL 테스트")
print("="*80)

def test_tp_sl(df, tp, sl):
    results_list = []
    
    for fvg in fvgs_4h:
        fvg_time = fvg['datetime']
        fvg_top = fvg['fvg_top']
        
        mask = df_15m['datetime'] > fvg_time
        future_15m = df_15m[mask].head(200)
        
        if len(future_15m) < 50:
            continue
        
        for i, (idx, row) in enumerate(future_15m.iterrows()):
            if row['low'] <= fvg_top:
                remaining = future_15m.iloc[i+1:]
                if len(remaining) < 50:
                    break
                
                entry_price = remaining.iloc[0]['open']
                
                result = None
                for j, (_, bar) in enumerate(remaining.iloc[1:101].iterrows()):
                    high_pct = (bar['high'] - entry_price) / entry_price * 100
                    low_pct = (bar['low'] - entry_price) / entry_price * 100
                    open_pct = (bar['open'] - entry_price) / entry_price * 100
                    
                    if high_pct >= tp and low_pct <= sl:
                        result = 'WIN' if open_pct >= 0 else 'LOSS'
                        break
                    elif high_pct >= tp:
                        result = 'WIN'
                        break
                    elif low_pct <= sl:
                        result = 'LOSS'
                        break
                
                results_list.append(result)
                break
    
    wins = results_list.count('WIN')
    losses = results_list.count('LOSS')
    decided = wins + losses
    
    if decided == 0:
        return 0, 0, 0
    
    wr = wins / decided * 100
    ev = (wr/100) * tp - ((100-wr)/100) * abs(sl)
    pnl = wins * tp + losses * sl
    monthly = decided / 60
    
    return wr, ev, pnl, monthly, decided

print("\n[전체 데이터 - TP/SL 조합]")
for tp in [0.5, 1.0, 1.5, 2.0, 3.0]:
    for sl in [-0.5, -1.0, -1.5, -2.0]:
        wr, ev, pnl, monthly, total = test_tp_sl(df_results, tp, sl)
        if ev > 0:
            print(f"TP {tp}% SL {sl}%: {total}건, 승률 {wr:.1f}%, EV {ev:.3f}%, 5년PnL {pnl:.0f}%")

