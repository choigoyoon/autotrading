"""
분할 익절 전략 완전 최적화
- TP 기준: 트렌드라인(저항선)까지의 거리
- SL 기준: HL 가격
- 분할 비율 최적화
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
if 'hl_confirm_time' in signals.columns:
    signals['hl_confirm_time'] = pd.to_datetime(signals['hl_confirm_time'])

df_15m['time'] = pd.to_datetime(df_15m['time'])
df_15m = df_15m.sort_values('time').reset_index(drop=True)

print("="*80)
print("분할 익절 전략 완전 최적화")
print("="*80)
print(f"\n총 시그널 수: {len(signals)}")

# 트렌드라인 기울기 계산 (진입 후 연장)
def calc_trendline_at_time(h1_time, h1_price, h2_time, h2_price, target_time):
    """특정 시간의 트렌드라인 가격 계산 (진입 후 연장)"""
    time_diff_h = (h2_time - h1_time).total_seconds() / 3600
    if time_diff_h == 0:
        return h2_price
    
    slope_per_hour = (h2_price - h1_price) / time_diff_h
    target_diff_h = (target_time - h1_time).total_seconds() / 3600
    
    return h1_price + slope_per_hour * target_diff_h

def run_backtest(signals_df, df_15m, tp1_ratio, tp2_ratio, split_ratio, sl_buffer_pct, 
                 use_trailing_sl, max_hold_hours):
    """
    분할 익절 백테스트
    
    tp1_ratio: 트렌드라인까지의 거리 중 1차 익절 지점 비율 (0.3 = 30% 지점)
    tp2_ratio: 트렌드라인까지의 거리 중 2차 익절 지점 비율 (1.0 = 트렌드라인 도달)
    split_ratio: 1차 익절 시 매도 비율 (0.5 = 50%)
    sl_buffer_pct: HL 아래 버퍼 (0.5 = HL 가격의 0.5% 아래)
    use_trailing_sl: 1차 익절 후 손절가 올릴지 여부
    max_hold_hours: 최대 보유 시간
    """
    
    results = []
    
    for idx, signal in signals_df.iterrows():
        # 기본 정보
        breakout_time = signal['breakout_time']
        breakout_price = signal['breakout_price']
        h1_time = signal['h1_time']
        h1_price = signal['h1_price']
        h2_time = signal['h2_time']
        h2_price = signal['h2_price']
        hl_price = signal['hl_price']
        
        # 진입가 = 돌파가
        entry_price = breakout_price
        entry_time = breakout_time
        
        # 진입 시점의 트렌드라인 가격 (돌파 직후)
        trendline_at_entry = calc_trendline_at_time(h1_time, h1_price, h2_time, h2_price, entry_time)
        
        # TP 거리 = 트렌드라인 - 진입가 (상승 기대)
        tp_distance = trendline_at_entry - entry_price
        
        # 만약 트렌드라인이 진입가 아래라면 (이미 돌파했으므로), 양수로 조정
        # 돌파 후이므로 트렌드라인은 저항선 역할 -> 트렌드라인이 위에 있어야 함
        # 하지만 하락 추세선이므로 돌파 시점에서는 가격이 트렌드라인 위에 있음
        # 따라서 tp_distance가 음수가 맞음 (트렌드라인이 아래에 있음)
        
        # 재정의: 상방 돌파 후이므로 추가 상승 목표 설정
        # 트렌드라인 기울기 기반으로 목표가 설정
        time_diff_h = (h2_time - h1_time).total_seconds() / 3600
        if time_diff_h == 0:
            slope_per_hour = 0
        else:
            slope_per_hour = (h2_price - h1_price) / time_diff_h
        
        # 목표 거리: |기울기| * 시간 or 트렌드라인 높이 기반
        # 간단하게: entry_price에서 일정 % 상승을 TP로
        # 또는: 트렌드라인 돌파 gap 기반
        
        # HL 기준 손절가
        sl_price = hl_price * (1 - sl_buffer_pct / 100)
        
        # TP 거리 재계산: 돌파 gap 기반
        # gap = entry_price - trendline_at_entry (돌파 시 가격 - 트렌드라인)
        gap = entry_price - trendline_at_entry
        
        # 트렌드라인 연장 후 저항선 역할 (미래 시점)
        # TP1: entry + gap * tp1_ratio (트렌드라인 위로 추가 상승)
        # TP2: entry + gap * tp2_ratio
        
        # 새로운 접근: 진입가에서 트렌드라인 높이(H1-H2 가격차)의 일정 비율을 TP로
        trendline_height = abs(h1_price - h2_price)
        
        tp1_target = entry_price + trendline_height * tp1_ratio
        tp2_target = entry_price + trendline_height * tp2_ratio
        
        # 백테스트 실행
        future_data = df_15m[df_15m['time'] > entry_time].head(max_hold_hours * 4)  # 15분봉 4개 = 1시간
        
        if len(future_data) == 0:
            continue
        
        position_remaining = 1.0  # 100%
        total_pnl = 0
        tp1_done = False
        tp2_done = False
        sl_done = False
        exit_log = []
        current_sl = sl_price
        
        for _, candle in future_data.iterrows():
            candle_time = candle['time']
            high = candle['high']
            low = candle['low']
            close = candle['close']
            
            # 시간 초과 체크
            hold_hours = (candle_time - entry_time).total_seconds() / 3600
            if hold_hours > max_hold_hours:
                # 시간 초과 시 현재가로 청산
                if position_remaining > 0:
                    pnl = (close - entry_price) / entry_price * 100 * position_remaining
                    total_pnl += pnl
                    exit_log.append(('TIME', position_remaining * 100, pnl))
                    position_remaining = 0
                break
            
            # 손절 체크 (저점이 SL 가격 이탈)
            if low <= current_sl and position_remaining > 0:
                sl_pnl = (current_sl - entry_price) / entry_price * 100 * position_remaining
                total_pnl += sl_pnl
                exit_log.append(('SL', position_remaining * 100, sl_pnl))
                sl_done = True
                position_remaining = 0
                break
            
            # TP1 체크
            if not tp1_done and high >= tp1_target and position_remaining > 0:
                tp1_pnl = (tp1_target - entry_price) / entry_price * 100 * split_ratio
                total_pnl += tp1_pnl
                exit_log.append(('TP1', split_ratio * 100, tp1_pnl))
                position_remaining -= split_ratio
                tp1_done = True
                
                # Trailing SL: TP1 달성 시 손절가를 진입가로 올림
                if use_trailing_sl:
                    current_sl = entry_price
            
            # TP2 체크
            if not tp2_done and high >= tp2_target and position_remaining > 0:
                tp2_pnl = (tp2_target - entry_price) / entry_price * 100 * position_remaining
                total_pnl += tp2_pnl
                exit_log.append(('TP2', position_remaining * 100, tp2_pnl))
                position_remaining = 0
                tp2_done = True
                break
        
        # 남은 포지션 처리 (마지막 봉 종가로)
        if position_remaining > 0 and len(future_data) > 0:
            last_close = future_data.iloc[-1]['close']
            pnl = (last_close - entry_price) / entry_price * 100 * position_remaining
            total_pnl += pnl
            exit_log.append(('EXIT', position_remaining * 100, pnl))
        
        results.append({
            'entry_time': entry_time,
            'entry_price': entry_price,
            'hl_price': hl_price,
            'sl_price': sl_price,
            'tp1_target': tp1_target,
            'tp2_target': tp2_target,
            'trendline_height': trendline_height,
            'total_pnl': total_pnl,
            'tp1_done': tp1_done,
            'tp2_done': tp2_done,
            'sl_done': sl_done,
            'exit_log': str(exit_log)
        })
    
    return pd.DataFrame(results)

# 최적화 파라미터 조합
print("\n" + "="*80)
print("파라미터 최적화 시작")
print("="*80)

# 파라미터 범위
tp1_ratios = [0.2, 0.3, 0.4, 0.5]  # 트렌드라인 높이의 20-50%
tp2_ratios = [0.5, 0.75, 1.0, 1.5]  # 트렌드라인 높이의 50-150%
split_ratios = [0.3, 0.5, 0.7]  # 1차 익절 시 30-70% 매도
sl_buffers = [0.3, 0.5, 1.0]  # HL 대비 0.3-1% 버퍼
trailing_options = [False, True]
max_hold_options = [24, 48, 72]  # 최대 보유 시간

optimization_results = []

total_combinations = len(tp1_ratios) * len(tp2_ratios) * len(split_ratios) * len(sl_buffers) * len(trailing_options) * len(max_hold_options)
print(f"총 테스트 조합: {total_combinations}")

count = 0
for tp1_r in tp1_ratios:
    for tp2_r in tp2_ratios:
        if tp2_r <= tp1_r:  # TP2는 TP1보다 커야 함
            continue
        for split_r in split_ratios:
            for sl_buf in sl_buffers:
                for trailing in trailing_options:
                    for max_hold in max_hold_options:
                        count += 1
                        if count % 50 == 0:
                            print(f"  진행: {count}/{total_combinations}")
                        
                        result_df = run_backtest(
                            signals, df_15m, tp1_r, tp2_r, split_r, sl_buf, trailing, max_hold
                        )
                        
                        if len(result_df) == 0:
                            continue
                        
                        # 성과 계산
                        tp1_rate = result_df['tp1_done'].sum() / len(result_df) * 100
                        tp2_rate = result_df['tp2_done'].sum() / len(result_df) * 100
                        sl_rate = result_df['sl_done'].sum() / len(result_df) * 100
                        win_rate = (result_df['total_pnl'] > 0).sum() / len(result_df) * 100
                        avg_pnl = result_df['total_pnl'].mean()
                        total_pnl = result_df['total_pnl'].sum()
                        
                        optimization_results.append({
                            'tp1_ratio': tp1_r,
                            'tp2_ratio': tp2_r,
                            'split_ratio': split_r,
                            'sl_buffer_pct': sl_buf,
                            'use_trailing': trailing,
                            'max_hold_hours': max_hold,
                            'signals': len(result_df),
                            'tp1_rate': tp1_rate,
                            'tp2_rate': tp2_rate,
                            'sl_rate': sl_rate,
                            'win_rate': win_rate,
                            'avg_pnl': avg_pnl,
                            'total_pnl': total_pnl
                        })

opt_df = pd.DataFrame(optimization_results)

# 결과 정렬 (평균 수익 기준)
opt_df = opt_df.sort_values('avg_pnl', ascending=False)

print("\n" + "="*80)
print("최적화 결과 TOP 20 (평균 수익 기준)")
print("="*80)

top_20 = opt_df.head(20)
for i, row in top_20.iterrows():
    print(f"\nTP1: {row['tp1_ratio']*100:.0f}% | TP2: {row['tp2_ratio']*100:.0f}% | "
          f"Split: {row['split_ratio']*100:.0f}% | SL버퍼: {row['sl_buffer_pct']:.1f}% | "
          f"Trailing: {row['use_trailing']} | 보유: {row['max_hold_hours']}h")
    print(f"  TP1달성: {row['tp1_rate']:.1f}% | TP2달성: {row['tp2_rate']:.1f}% | SL: {row['sl_rate']:.1f}%")
    print(f"  승률: {row['win_rate']:.1f}% | 평균수익: {row['avg_pnl']:+.2f}% | 총수익: {row['total_pnl']:+.1f}%")

# 승률 기준 Top 10
print("\n" + "="*80)
print("최적화 결과 TOP 10 (승률 기준)")
print("="*80)

top_winrate = opt_df.sort_values('win_rate', ascending=False).head(10)
for i, row in top_winrate.iterrows():
    print(f"\nTP1: {row['tp1_ratio']*100:.0f}% | TP2: {row['tp2_ratio']*100:.0f}% | "
          f"Split: {row['split_ratio']*100:.0f}% | SL버퍼: {row['sl_buffer_pct']:.1f}% | "
          f"Trailing: {row['use_trailing']} | 보유: {row['max_hold_hours']}h")
    print(f"  TP1달성: {row['tp1_rate']:.1f}% | TP2달성: {row['tp2_rate']:.1f}% | SL: {row['sl_rate']:.1f}%")
    print(f"  승률: {row['win_rate']:.1f}% | 평균수익: {row['avg_pnl']:+.2f}% | 총수익: {row['total_pnl']:+.1f}%")

# 최적 추천
print("\n" + "="*80)
print("🏆 최종 추천 파라미터")
print("="*80)

# 조건: 승률 60% 이상, 평균 수익 1% 이상
filtered = opt_df[(opt_df['win_rate'] >= 60) & (opt_df['avg_pnl'] >= 1.0)]
if len(filtered) > 0:
    best = filtered.iloc[0]
    print(f"\n✅ 균형 잡힌 최적 조건 (승률 60%+, 평균수익 1%+)")
else:
    best = opt_df.iloc[0]
    print(f"\n✅ 평균 수익 최고 조건")

print(f"\n   TP1 목표: 트렌드라인 높이의 {best['tp1_ratio']*100:.0f}%")
print(f"   TP2 목표: 트렌드라인 높이의 {best['tp2_ratio']*100:.0f}%")
print(f"   분할 비율: 1차 익절 시 {best['split_ratio']*100:.0f}% 매도")
print(f"   SL 버퍼: HL 가격 -{best['sl_buffer_pct']:.1f}%")
print(f"   Trailing SL: {best['use_trailing']}")
print(f"   최대 보유: {best['max_hold_hours']}시간")
print(f"\n   결과:")
print(f"   - TP1 달성률: {best['tp1_rate']:.1f}%")
print(f"   - TP2 달성률: {best['tp2_rate']:.1f}%")
print(f"   - 손절률: {best['sl_rate']:.1f}%")
print(f"   - 승률: {best['win_rate']:.1f}%")
print(f"   - 평균 수익: {best['avg_pnl']:+.2f}%")
print(f"   - 총 수익: {best['total_pnl']:+.1f}%")

# 결과 저장
opt_df.to_csv('partial_tp_optimization.csv', index=False)
print(f"\n최적화 결과 저장: partial_tp_optimization.csv ({len(opt_df)}개 조합)")

# 최적 파라미터로 상세 백테스트 실행
print("\n" + "="*80)
print("최적 파라미터로 상세 백테스트")
print("="*80)

best_result = run_backtest(
    signals, df_15m, 
    best['tp1_ratio'], best['tp2_ratio'], best['split_ratio'],
    best['sl_buffer_pct'], best['use_trailing'], int(best['max_hold_hours'])
)

best_result.to_csv('optimal_partial_tp_results.csv', index=False)
print(f"\n상세 결과 저장: optimal_partial_tp_results.csv ({len(best_result)}건)")

# 연도별 성과
best_result['year'] = pd.to_datetime(best_result['entry_time']).dt.year
yearly = best_result.groupby('year').agg({
    'total_pnl': ['count', 'sum', 'mean'],
    'tp1_done': 'sum',
    'tp2_done': 'sum',
    'sl_done': 'sum'
}).round(2)

print("\n연도별 성과:")
print(yearly)

