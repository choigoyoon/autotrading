"""
빠른 분할 익절 전략 최적화 - 벡터화 연산 활용
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
signals = pd.read_csv('valid_signals.csv')
df_15m = pd.read_csv('analysis_15m.csv')

# 시간 컬럼 변환
signals['breakout_time'] = pd.to_datetime(signals['breakout_time'])
signals['h1_time'] = pd.to_datetime(signals['h1_time'])
signals['h2_time'] = pd.to_datetime(signals['h2_time'])

df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m.sort_values('datetime').reset_index(drop=True)

print("="*80)
print("분할 익절 전략 빠른 최적화")
print("="*80)
print(f"시그널: {len(signals)}개, 15분봉: {len(df_15m)}개")

# 각 시그널별로 미래 가격 데이터 미리 계산
def precompute_future_data(signals, df_15m, max_hours=72):
    """각 시그널별 미래 데이터 사전 계산"""
    all_future = {}
    
    for idx, signal in signals.iterrows():
        entry_time = signal['breakout_time']
        future = df_15m[df_15m['datetime'] > entry_time].head(max_hours * 4)
        
        if len(future) > 0:
            # 필요한 컬럼만 numpy array로 변환
            all_future[idx] = {
                'times': (future['datetime'] - entry_time).dt.total_seconds().values / 3600,
                'highs': future['high'].values,
                'lows': future['low'].values,
                'closes': future['close'].values
            }
    
    return all_future

print("\n미래 데이터 사전 계산 중...")
future_data_cache = precompute_future_data(signals, df_15m)
print(f"캐시 완료: {len(future_data_cache)}개 시그널")

def run_backtest_fast(signals_df, future_cache, tp1_ratio, tp2_ratio, split_ratio, 
                       sl_buffer_pct, use_trailing_sl, max_hold_hours):
    """최적화된 백테스트"""
    
    results = []
    
    for idx, signal in signals_df.iterrows():
        if idx not in future_cache:
            continue
            
        entry_price = signal['breakout_price']
        h1_price = signal['h1_price']
        h2_price = signal['h2_price']
        hl_price = signal['hl_price']
        
        # 트렌드라인 높이
        tl_height = abs(h1_price - h2_price)
        
        # TP/SL 목표
        tp1_target = entry_price + tl_height * tp1_ratio
        tp2_target = entry_price + tl_height * tp2_ratio
        sl_price = hl_price * (1 - sl_buffer_pct / 100)
        
        if sl_price >= entry_price:
            sl_price = entry_price * 0.99
        
        # 캐시된 데이터
        cache = future_cache[idx]
        times = cache['times']
        highs = cache['highs']
        lows = cache['lows']
        closes = cache['closes']
        
        # 시간 제한 적용
        valid_mask = times <= max_hold_hours
        times = times[valid_mask]
        highs = highs[valid_mask]
        lows = lows[valid_mask]
        closes = closes[valid_mask]
        
        if len(times) == 0:
            continue
        
        # 시뮬레이션
        position = 1.0
        total_pnl = 0
        tp1_done = False
        tp2_done = False
        sl_done = False
        current_sl = sl_price
        
        for i in range(len(times)):
            # 손절 체크
            if lows[i] <= current_sl and position > 0:
                sl_pnl = (current_sl - entry_price) / entry_price * 100 * position
                total_pnl += sl_pnl
                sl_done = True
                position = 0
                break
            
            # TP1 체크
            if not tp1_done and highs[i] >= tp1_target and position > 0:
                tp1_pnl = (tp1_target - entry_price) / entry_price * 100 * split_ratio
                total_pnl += tp1_pnl
                position -= split_ratio
                tp1_done = True
                
                if use_trailing_sl:
                    current_sl = entry_price
            
            # TP2 체크
            if not tp2_done and highs[i] >= tp2_target and position > 0:
                tp2_pnl = (tp2_target - entry_price) / entry_price * 100 * position
                total_pnl += tp2_pnl
                position = 0
                tp2_done = True
                break
        
        # 잔여 포지션 청산
        if position > 0 and len(closes) > 0:
            pnl = (closes[-1] - entry_price) / entry_price * 100 * position
            total_pnl += pnl
        
        results.append({
            'total_pnl': total_pnl,
            'tp1_done': tp1_done,
            'tp2_done': tp2_done,
            'sl_done': sl_done
        })
    
    return pd.DataFrame(results)

# 최적화 실행
print("\n" + "="*80)
print("파라미터 최적화 시작")
print("="*80)

# 축소된 파라미터
tp1_ratios = [0.2, 0.3, 0.5, 0.7]
tp2_ratios = [0.5, 1.0, 1.5, 2.0, 3.0]
split_ratios = [0.3, 0.5, 0.7]
sl_buffers = [0.3, 0.5, 1.0, 1.5]
trailing_options = [False, True]
max_hold_options = [24, 48, 72]

optimization_results = []
count = 0

for tp1_r in tp1_ratios:
    for tp2_r in tp2_ratios:
        if tp2_r <= tp1_r:
            continue
        for split_r in split_ratios:
            for sl_buf in sl_buffers:
                for trailing in trailing_options:
                    for max_hold in max_hold_options:
                        count += 1
                        
                        result_df = run_backtest_fast(
                            signals, future_data_cache, 
                            tp1_r, tp2_r, split_r, sl_buf, trailing, max_hold
                        )
                        
                        if len(result_df) == 0:
                            continue
                        
                        optimization_results.append({
                            'tp1_ratio': tp1_r,
                            'tp2_ratio': tp2_r,
                            'split_ratio': split_r,
                            'sl_buffer_pct': sl_buf,
                            'use_trailing': trailing,
                            'max_hold_hours': max_hold,
                            'signals': len(result_df),
                            'tp1_rate': result_df['tp1_done'].mean() * 100,
                            'tp2_rate': result_df['tp2_done'].mean() * 100,
                            'sl_rate': result_df['sl_done'].mean() * 100,
                            'win_rate': (result_df['total_pnl'] > 0).mean() * 100,
                            'avg_pnl': result_df['total_pnl'].mean(),
                            'total_pnl': result_df['total_pnl'].sum()
                        })

print(f"\n총 {count}개 조합 테스트 완료")

opt_df = pd.DataFrame(optimization_results)
opt_df = opt_df.sort_values('avg_pnl', ascending=False)

# 결과 출력
print("\n" + "="*80)
print("🏆 TOP 15 (평균 수익 기준)")
print("="*80)

for i, row in opt_df.head(15).iterrows():
    print(f"\nTP1: {row['tp1_ratio']*100:.0f}% | TP2: {row['tp2_ratio']*100:.0f}% | Split: {row['split_ratio']*100:.0f}%")
    print(f"  SL버퍼: {row['sl_buffer_pct']:.1f}% | Trailing: {row['use_trailing']} | 보유: {row['max_hold_hours']}h")
    print(f"  TP1: {row['tp1_rate']:.1f}% | TP2: {row['tp2_rate']:.1f}% | SL: {row['sl_rate']:.1f}%")
    print(f"  승률: {row['win_rate']:.1f}% | 평균: {row['avg_pnl']:+.2f}% | 총: {row['total_pnl']:+.1f}%")

# 승률 기준
print("\n" + "="*80)
print("🎯 TOP 10 (승률 기준)")
print("="*80)

for i, row in opt_df.sort_values('win_rate', ascending=False).head(10).iterrows():
    print(f"\nTP1: {row['tp1_ratio']*100:.0f}% | TP2: {row['tp2_ratio']*100:.0f}% | Split: {row['split_ratio']*100:.0f}%")
    print(f"  승률: {row['win_rate']:.1f}% | 평균: {row['avg_pnl']:+.2f}%")

# 최종 추천
print("\n" + "="*80)
print("🏆 최종 추천")
print("="*80)

# 필터: 승률 50%+, 평균수익 0.5%+
filtered = opt_df[(opt_df['win_rate'] >= 50) & (opt_df['avg_pnl'] >= 0.5)]
if len(filtered) > 0:
    best = filtered.sort_values('avg_pnl', ascending=False).iloc[0]
    print("\n✅ 균형 조건 (승률 50%+, 평균 0.5%+)")
else:
    best = opt_df.iloc[0]
    print("\n✅ 평균 수익 최고")

print(f"\n   TP1: 트렌드라인 높이 × {best['tp1_ratio']*100:.0f}%")
print(f"   TP2: 트렌드라인 높이 × {best['tp2_ratio']*100:.0f}%")
print(f"   분할: TP1에서 {best['split_ratio']*100:.0f}% 매도")
print(f"   SL: HL 가격 -{best['sl_buffer_pct']:.1f}%")
print(f"   Trailing: {best['use_trailing']}")
print(f"   최대 보유: {best['max_hold_hours']}h")
print(f"\n   ═══ 결과 ═══")
print(f"   TP1 달성: {best['tp1_rate']:.1f}%")
print(f"   TP2 달성: {best['tp2_rate']:.1f}%")
print(f"   손절률: {best['sl_rate']:.1f}%")
print(f"   승률: {best['win_rate']:.1f}%")
print(f"   평균 수익: {best['avg_pnl']:+.2f}%")
print(f"   총 수익: {best['total_pnl']:+.1f}%")

# 저장
opt_df.to_csv('partial_tp_optimization.csv', index=False)
print(f"\n결과 저장: partial_tp_optimization.csv ({len(opt_df)}개 조합)")

# 비교
print("\n" + "="*80)
print("📊 기존 대비 비교")
print("="*80)
print(f"\n기존 (TP 5% / SL 2%): TP달성 17.3%, 평균 +0.60%")
print(f"최적화: TP1달성 {best['tp1_rate']:.1f}%, 평균 {best['avg_pnl']:+.2f}%")
print(f"개선: TP달성 {best['tp1_rate']/17.3*100-100:+.0f}%, 평균수익 {(best['avg_pnl']/0.6-1)*100:+.0f}%")

