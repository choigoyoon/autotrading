"""
MTF 높은 타임프레임 검증
- 기존 전략 (4H+15M) vs 높은 타임프레임 (1D+1H 또는 1D+4H)
- FVG + Order Block 전략 적용
- 동일한 파라미터로 성과 비교
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("="*70)
print("MTF 높은 타임프레임 검증")
print("="*70)

# =====================================================================
# 데이터 로드
# =====================================================================
print("\n[1] 데이터 로드 중...")

# 15분봉 (기존 진입 타임프레임)
df_15m = pd.read_csv('btc_15m_ohlcv.csv')
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
df_15m = df_15m.sort_values('datetime').reset_index(drop=True)

# 4시간봉 (기존 시그널 타임프레임)
df_4h = pd.read_csv('btc_4h_ohlcv.csv')
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])
df_4h = df_4h.sort_values('datetime').reset_index(drop=True)

# 1시간봉 (새로운 진입 타임프레임)
df_1h = pd.read_csv('btc_1h_ohlcv.csv')
df_1h['datetime'] = pd.to_datetime(df_1h['datetime'])
df_1h = df_1h.sort_values('datetime').reset_index(drop=True)

# 1일봉 생성 (4시간봉에서)
print("\n[2] 1일봉 데이터 생성 중...")
df_4h['date'] = df_4h['datetime'].dt.date
df_1d = df_4h.groupby('date').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).reset_index()
df_1d.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
df_1d['datetime'] = pd.to_datetime(df_1d['date'])
df_1d = df_1d.sort_values('datetime').reset_index(drop=True)

print(f"  15분봉: {len(df_15m):,}개 ({df_15m['datetime'].min()} ~ {df_15m['datetime'].max()})")
print(f"  1시간봉: {len(df_1h):,}개 ({df_1h['datetime'].min()} ~ {df_1h['datetime'].max()})")
print(f"  4시간봉: {len(df_4h):,}개 ({df_4h['datetime'].min()} ~ {df_4h['datetime'].max()})")
print(f"  1일봉: {len(df_1d):,}개 ({df_1d['datetime'].min()} ~ {df_1d['datetime'].max()})")

# =====================================================================
# 시그널 탐지 함수
# =====================================================================

def detect_fvg_signals(df, min_gap_pct=0.3):
    """FVG 시그널 탐지"""
    signals = []
    
    for i in range(2, len(df)):
        current = df.iloc[i]
        prev2 = df.iloc[i-2]
        
        # Bullish FVG (롱)
        if current['low'] > prev2['high']:
            gap_pct = (current['low'] - prev2['high']) / prev2['high'] * 100
            if gap_pct >= min_gap_pct:
                signals.append({
                    'datetime': current['datetime'],
                    'idx': i,
                    'type': 'FVG_bull',
                    'direction': 'long',
                    'entry_zone': current['low'],  # 갭 상단
                    'gap_pct': gap_pct
                })
        
        # Bearish FVG (숏)
        if current['high'] < prev2['low']:
            gap_pct = (prev2['low'] - current['high']) / prev2['low'] * 100
            if gap_pct >= min_gap_pct:
                signals.append({
                    'datetime': current['datetime'],
                    'idx': i,
                    'type': 'FVG_bear',
                    'direction': 'short',
                    'entry_zone': current['high'],  # 갭 하단
                    'gap_pct': gap_pct
                })
    
    return pd.DataFrame(signals)

def detect_orderblock_signals(df):
    """Order Block 시그널 탐지 (Bullish만)"""
    signals = []
    
    for i in range(2, len(df)):
        prev2 = df.iloc[i-2]
        prev1 = df.iloc[i-1]
        current = df.iloc[i]
        
        # Bullish Order Block (롱)
        if (prev2['close'] < prev2['open'] and  # 2봉 전 음봉
            prev1['close'] > prev1['open'] and  # 1봉 전 양봉
            prev1['close'] > prev2['high']):     # 돌파
            
            signals.append({
                'datetime': current['datetime'],
                'idx': i,
                'type': 'orderblock',
                'direction': 'long',
                'entry_zone': prev2['high']
            })
    
    return pd.DataFrame(signals)

# =====================================================================
# 진입 및 백테스트 함수
# =====================================================================

def execute_trades(signals_df, entry_df, params):
    """
    시그널 발생 후 진입 타임프레임에서 실제 진입 실행
    
    Parameters:
    - signals_df: 시그널 데이터프레임 (높은 타임프레임)
    - entry_df: 진입 타임프레임 데이터프레임 (낮은 타임프레임)
    - params: 매매 파라미터 딕셔너리
    """
    
    tp1_pct = params['tp1']
    tp2_pct = params['tp2']
    sl_pct = params['sl']
    time_stop_bars = params['time_stop']  # 진입 타임프레임 기준
    
    trades = []
    
    for idx, signal in signals_df.iterrows():
        signal_time = signal['datetime']
        entry_zone = signal['entry_zone']
        direction = signal['direction']
        signal_type = signal['type']
        
        # 진입 유효기간 (시그널 발생 후 일정 시간)
        if 'valid_hours' in params:
            valid_until = signal_time + timedelta(hours=params['valid_hours'])
        else:
            valid_until = signal_time + timedelta(days=999)  # 무제한
        
        # 진입 타임프레임에서 진입 찾기
        entry_window = entry_df[
            (entry_df['datetime'] >= signal_time) &
            (entry_df['datetime'] <= valid_until)
        ]
        
        if len(entry_window) == 0:
            continue
        
        # 진입 조건
        entry_bar = None
        entry_price = None
        
        for i, bar in entry_window.iterrows():
            if direction == 'long':
                # 저가가 진입존 터치
                if bar['low'] <= entry_zone:
                    entry_bar = i
                    entry_price = entry_zone
                    break
            else:  # short
                # 고가가 진입존 터치
                if bar['high'] >= entry_zone:
                    entry_bar = i
                    entry_price = entry_zone
                    break
        
        if entry_bar is None:
            continue
        
        # 진입 이후 데이터
        entry_idx = entry_df.index.get_loc(entry_bar)
        max_idx = min(entry_idx + time_stop_bars, len(entry_df) - 1)
        
        if max_idx <= entry_idx:
            continue
        
        position_window = entry_df.iloc[entry_idx:max_idx+1]
        entry_datetime = position_window.iloc[0]['datetime']
        
        # TP/SL 레벨 계산
        if direction == 'long':
            tp1_level = entry_price * (1 + tp1_pct / 100)
            tp2_level = entry_price * (1 + tp2_pct / 100)
            sl_level = entry_price * (1 - sl_pct / 100)
            
            # TP1 도달 여부
            tp1_hit = (position_window['high'] >= tp1_level).any()
            tp1_idx = None
            if tp1_hit:
                tp1_idx = position_window[position_window['high'] >= tp1_level].index[0]
            
            # TP1 도달 전 SL/TIME 체크
            if tp1_hit:
                before_tp1 = position_window.loc[:tp1_idx]
                sl_before_tp1 = (before_tp1['low'] <= sl_level).any()
                
                if sl_before_tp1:
                    # TP1 전에 SL
                    pnl = -sl_pct
                    result = 'SL'
                else:
                    # TP1 도달, SL을 본절로 이동
                    after_tp1 = position_window.loc[tp1_idx:]
                    
                    # TP2 체크
                    tp2_hit = (after_tp1['high'] >= tp2_level).any()
                    be_hit = (after_tp1['low'] <= entry_price).any()
                    
                    if tp2_hit and be_hit:
                        tp2_idx = after_tp1[after_tp1['high'] >= tp2_level].index[0]
                        be_idx = after_tp1[after_tp1['low'] <= entry_price].index[0]
                        
                        if tp2_idx < be_idx:
                            pnl = tp2_pct
                            result = 'TP2'
                        else:
                            pnl = 0.0
                            result = 'BE'
                    elif tp2_hit:
                        pnl = tp2_pct
                        result = 'TP2'
                    elif be_hit:
                        pnl = 0.0
                        result = 'BE'
                    else:
                        # 시간 스탑
                        final_price = after_tp1.iloc[-1]['close']
                        pnl = (final_price - entry_price) / entry_price * 100
                        result = 'TIME'
            else:
                # TP1 미도달, SL 또는 TIME
                sl_hit = (position_window['low'] <= sl_level).any()
                
                if sl_hit:
                    pnl = -sl_pct
                    result = 'SL'
                else:
                    # 시간 스탑
                    final_price = position_window.iloc[-1]['close']
                    pnl = (final_price - entry_price) / entry_price * 100
                    result = 'TIME'
        
        else:  # short
            tp1_level = entry_price * (1 - tp1_pct / 100)
            tp2_level = entry_price * (1 - tp2_pct / 100)
            sl_level = entry_price * (1 + sl_pct / 100)
            
            # TP1 도달 여부
            tp1_hit = (position_window['low'] <= tp1_level).any()
            tp1_idx = None
            if tp1_hit:
                tp1_idx = position_window[position_window['low'] <= tp1_level].index[0]
            
            # TP1 도달 전 SL/TIME 체크
            if tp1_hit:
                before_tp1 = position_window.loc[:tp1_idx]
                sl_before_tp1 = (before_tp1['high'] >= sl_level).any()
                
                if sl_before_tp1:
                    # TP1 전에 SL
                    pnl = -sl_pct
                    result = 'SL'
                else:
                    # TP1 도달, SL을 본절로 이동
                    after_tp1 = position_window.loc[tp1_idx:]
                    
                    # TP2 체크
                    tp2_hit = (after_tp1['low'] <= tp2_level).any()
                    be_hit = (after_tp1['high'] >= entry_price).any()
                    
                    if tp2_hit and be_hit:
                        tp2_idx = after_tp1[after_tp1['low'] <= tp2_level].index[0]
                        be_idx = after_tp1[after_tp1['high'] >= entry_price].index[0]
                        
                        if tp2_idx < be_idx:
                            pnl = tp2_pct
                            result = 'TP2'
                        else:
                            pnl = 0.0
                            result = 'BE'
                    elif tp2_hit:
                        pnl = tp2_pct
                        result = 'TP2'
                    elif be_hit:
                        pnl = 0.0
                        result = 'BE'
                    else:
                        # 시간 스탑
                        final_price = after_tp1.iloc[-1]['close']
                        pnl = (entry_price - final_price) / entry_price * 100
                        result = 'TIME'
            else:
                # TP1 미도달, SL 또는 TIME
                sl_hit = (position_window['high'] >= sl_level).any()
                
                if sl_hit:
                    pnl = -sl_pct
                    result = 'SL'
                else:
                    # 시간 스탑
                    final_price = position_window.iloc[-1]['close']
                    pnl = (entry_price - final_price) / entry_price * 100
                    result = 'TIME'
        
        trades.append({
            'signal_datetime': signal_time,
            'entry_datetime': entry_datetime,
            'signal_type': signal_type,
            'direction': direction,
            'entry_price': entry_price,
            'pnl': pnl,
            'result': result
        })
    
    return pd.DataFrame(trades)

# =====================================================================
# 매매 파라미터 (기존 최적 설정)
# =====================================================================

params = {
    'tp1': 1.5,
    'tp2': 3.5,
    'sl': 1.5,
    'time_stop': 48,  # 바 수 (진입 타임프레임 기준)
    'valid_hours': 5   # 진입 유효기간 (시간)
}

print("\n[3] 매매 파라미터:")
print(f"  TP1: {params['tp1']}%")
print(f"  TP2: {params['tp2']}%")
print(f"  SL: {params['sl']}%")
print(f"  시간 스탑: {params['time_stop']}바")
print(f"  진입 유효기간: {params['valid_hours']}시간")

# =====================================================================
# 백테스트 #1: 기존 전략 (4H 시그널 + 15M 진입)
# =====================================================================

print("\n" + "="*70)
print("백테스트 #1: 기존 전략 (4H 시그널 + 15M 진입)")
print("="*70)

print("\n시그널 탐지 중 (4H)...")
signals_fvg_4h = detect_fvg_signals(df_4h, min_gap_pct=0.3)
signals_ob_4h = detect_orderblock_signals(df_4h)
signals_4h = pd.concat([signals_fvg_4h, signals_ob_4h], ignore_index=True)
signals_4h = signals_4h.sort_values('datetime').reset_index(drop=True)

print(f"  FVG: {len(signals_fvg_4h)}개")
print(f"  Order Block: {len(signals_ob_4h)}개")
print(f"  총 시그널: {len(signals_4h)}개")

print("\n거래 실행 중 (15M 진입)...")
params_15m = params.copy()
params_15m['time_stop'] = 48  # 15분 기준 48바 = 12시간
trades_4h_15m = execute_trades(signals_4h, df_15m, params_15m)

print(f"\n총 거래: {len(trades_4h_15m)}건")

if len(trades_4h_15m) > 0:
    total_pnl = trades_4h_15m['pnl'].sum()
    avg_pnl = trades_4h_15m['pnl'].mean()
    win_rate = (trades_4h_15m['pnl'] > 0).sum() / len(trades_4h_15m) * 100
    
    # 결과 분포
    result_dist = trades_4h_15m['result'].value_counts()
    
    # 연도별 데이터 기간
    trades_4h_15m['year'] = pd.to_datetime(trades_4h_15m['entry_datetime']).dt.year
    years = trades_4h_15m['year'].max() - trades_4h_15m['year'].min() + 1
    months = years * 12
    monthly_avg = total_pnl / months if months > 0 else 0
    
    # MDD 계산
    cumsum = trades_4h_15m['pnl'].cumsum()
    running_max = cumsum.cummax()
    drawdown = cumsum - running_max
    mdd = drawdown.min()
    
    print(f"\n📊 성과:")
    print(f"  총 수익: {total_pnl:.1f}%")
    print(f"  월평균 수익: {monthly_avg:.2f}%")
    print(f"  평균 수익/건: {avg_pnl:.3f}%")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  MDD: {mdd:.1f}%")
    
    print(f"\n📈 결과 분포:")
    for result, count in result_dist.items():
        pct = count / len(trades_4h_15m) * 100
        print(f"  {result}: {count}건 ({pct:.1f}%)")
    
    # 전략별 성과
    print(f"\n📊 전략별 성과:")
    for strat in trades_4h_15m['signal_type'].unique():
        strat_trades = trades_4h_15m[trades_4h_15m['signal_type'] == strat]
        strat_win = (strat_trades['pnl'] > 0).sum() / len(strat_trades) * 100
        strat_avg = strat_trades['pnl'].mean()
        print(f"  {strat}: {len(strat_trades)}건, 승률 {strat_win:.1f}%, 평균 {strat_avg:.3f}%")

# =====================================================================
# 백테스트 #2: 1D 시그널 + 1H 진입
# =====================================================================

print("\n" + "="*70)
print("백테스트 #2: 높은 타임프레임 (1D 시그널 + 1H 진입)")
print("="*70)

print("\n시그널 탐지 중 (1D)...")
signals_fvg_1d = detect_fvg_signals(df_1d, min_gap_pct=0.3)
signals_ob_1d = detect_orderblock_signals(df_1d)
signals_1d = pd.concat([signals_fvg_1d, signals_ob_1d], ignore_index=True)
signals_1d = signals_1d.sort_values('datetime').reset_index(drop=True)

print(f"  FVG: {len(signals_fvg_1d)}개")
print(f"  Order Block: {len(signals_ob_1d)}개")
print(f"  총 시그널: {len(signals_1d)}개")

print("\n거래 실행 중 (1H 진입)...")
params_1h = params.copy()
params_1h['time_stop'] = 12  # 1시간 기준 12바 = 12시간
params_1h['valid_hours'] = 120  # 1일봉 시그널은 더 긴 유효기간
trades_1d_1h = execute_trades(signals_1d, df_1h, params_1h)

print(f"\n총 거래: {len(trades_1d_1h)}건")

if len(trades_1d_1h) > 0:
    total_pnl = trades_1d_1h['pnl'].sum()
    avg_pnl = trades_1d_1h['pnl'].mean()
    win_rate = (trades_1d_1h['pnl'] > 0).sum() / len(trades_1d_1h) * 100
    
    # 결과 분포
    result_dist = trades_1d_1h['result'].value_counts()
    
    # 연도별 데이터 기간
    trades_1d_1h['year'] = pd.to_datetime(trades_1d_1h['entry_datetime']).dt.year
    years = trades_1d_1h['year'].max() - trades_1d_1h['year'].min() + 1
    months = years * 12
    monthly_avg = total_pnl / months if months > 0 else 0
    
    # MDD 계산
    cumsum = trades_1d_1h['pnl'].cumsum()
    running_max = cumsum.cummax()
    drawdown = cumsum - running_max
    mdd = drawdown.min()
    
    print(f"\n📊 성과:")
    print(f"  총 수익: {total_pnl:.1f}%")
    print(f"  월평균 수익: {monthly_avg:.2f}%")
    print(f"  평균 수익/건: {avg_pnl:.3f}%")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  MDD: {mdd:.1f}%")
    
    print(f"\n📈 결과 분포:")
    for result, count in result_dist.items():
        pct = count / len(trades_1d_1h) * 100
        print(f"  {result}: {count}건 ({pct:.1f}%)")
    
    # 전략별 성과
    print(f"\n📊 전략별 성과:")
    for strat in trades_1d_1h['signal_type'].unique():
        strat_trades = trades_1d_1h[trades_1d_1h['signal_type'] == strat]
        strat_win = (strat_trades['pnl'] > 0).sum() / len(strat_trades) * 100
        strat_avg = strat_trades['pnl'].mean()
        print(f"  {strat}: {len(strat_trades)}건, 승률 {strat_win:.1f}%, 평균 {strat_avg:.3f}%")

# =====================================================================
# 백테스트 #3: 1D 시그널 + 4H 진입
# =====================================================================

print("\n" + "="*70)
print("백테스트 #3: 높은 타임프레임 (1D 시그널 + 4H 진입)")
print("="*70)

print("\n거래 실행 중 (4H 진입)...")
params_4h = params.copy()
params_4h['time_stop'] = 3  # 4시간 기준 3바 = 12시간
params_4h['valid_hours'] = 120  # 1일봉 시그널은 더 긴 유효기간
trades_1d_4h = execute_trades(signals_1d, df_4h, params_4h)

print(f"\n총 거래: {len(trades_1d_4h)}건")

if len(trades_1d_4h) > 0:
    total_pnl = trades_1d_4h['pnl'].sum()
    avg_pnl = trades_1d_4h['pnl'].mean()
    win_rate = (trades_1d_4h['pnl'] > 0).sum() / len(trades_1d_4h) * 100
    
    # 결과 분포
    result_dist = trades_1d_4h['result'].value_counts()
    
    # 연도별 데이터 기간
    trades_1d_4h['year'] = pd.to_datetime(trades_1d_4h['entry_datetime']).dt.year
    years = trades_1d_4h['year'].max() - trades_1d_4h['year'].min() + 1
    months = years * 12
    monthly_avg = total_pnl / months if months > 0 else 0
    
    # MDD 계산
    cumsum = trades_1d_4h['pnl'].cumsum()
    running_max = cumsum.cummax()
    drawdown = cumsum - running_max
    mdd = drawdown.min()
    
    print(f"\n📊 성과:")
    print(f"  총 수익: {total_pnl:.1f}%")
    print(f"  월평균 수익: {monthly_avg:.2f}%")
    print(f"  평균 수익/건: {avg_pnl:.3f}%")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  MDD: {mdd:.1f}%")
    
    print(f"\n📈 결과 분포:")
    for result, count in result_dist.items():
        pct = count / len(trades_1d_4h) * 100
        print(f"  {result}: {count}건 ({pct:.1f}%)")
    
    # 전략별 성과
    print(f"\n📊 전략별 성과:")
    for strat in trades_1d_4h['signal_type'].unique():
        strat_trades = trades_1d_4h[trades_1d_4h['signal_type'] == strat]
        strat_win = (strat_trades['pnl'] > 0).sum() / len(strat_trades) * 100
        strat_avg = strat_trades['pnl'].mean()
        print(f"  {strat}: {len(strat_trades)}건, 승률 {strat_win:.1f}%, 평균 {strat_avg:.3f}%")

# =====================================================================
# 최종 비교
# =====================================================================

print("\n" + "="*70)
print("📊 최종 비교")
print("="*70)

strategies = []

if len(trades_4h_15m) > 0:
    strategies.append({
        'name': '기존 (4H+15M)',
        'trades': len(trades_4h_15m),
        'total_pnl': trades_4h_15m['pnl'].sum(),
        'monthly_avg': trades_4h_15m['pnl'].sum() / ((trades_4h_15m['year'].max() - trades_4h_15m['year'].min() + 1) * 12),
        'win_rate': (trades_4h_15m['pnl'] > 0).sum() / len(trades_4h_15m) * 100,
        'avg_pnl': trades_4h_15m['pnl'].mean(),
        'mdd': (trades_4h_15m['pnl'].cumsum() - trades_4h_15m['pnl'].cumsum().cummax()).min()
    })

if len(trades_1d_1h) > 0:
    strategies.append({
        'name': '높은TF (1D+1H)',
        'trades': len(trades_1d_1h),
        'total_pnl': trades_1d_1h['pnl'].sum(),
        'monthly_avg': trades_1d_1h['pnl'].sum() / ((trades_1d_1h['year'].max() - trades_1d_1h['year'].min() + 1) * 12),
        'win_rate': (trades_1d_1h['pnl'] > 0).sum() / len(trades_1d_1h) * 100,
        'avg_pnl': trades_1d_1h['pnl'].mean(),
        'mdd': (trades_1d_1h['pnl'].cumsum() - trades_1d_1h['pnl'].cumsum().cummax()).min()
    })

if len(trades_1d_4h) > 0:
    strategies.append({
        'name': '높은TF (1D+4H)',
        'trades': len(trades_1d_4h),
        'total_pnl': trades_1d_4h['pnl'].sum(),
        'monthly_avg': trades_1d_4h['pnl'].sum() / ((trades_1d_4h['year'].max() - trades_1d_4h['year'].min() + 1) * 12),
        'win_rate': (trades_1d_4h['pnl'] > 0).sum() / len(trades_1d_4h) * 100,
        'avg_pnl': trades_1d_4h['pnl'].mean(),
        'mdd': (trades_1d_4h['pnl'].cumsum() - trades_1d_4h['pnl'].cumsum().cummax()).min()
    })

comparison_df = pd.DataFrame(strategies)

print("\n")
print(comparison_df.to_string(index=False))

print("\n" + "="*70)
print("✅ MTF 높은 타임프레임 검증 완료")
print("="*70)

# 결과 저장
if len(trades_4h_15m) > 0:
    trades_4h_15m.to_csv('trades_4h_15m.csv', index=False)
    print("\n저장: trades_4h_15m.csv")

if len(trades_1d_1h) > 0:
    trades_1d_1h.to_csv('trades_1d_1h.csv', index=False)
    print("저장: trades_1d_1h.csv")

if len(trades_1d_4h) > 0:
    trades_1d_4h.to_csv('trades_1d_4h.csv', index=False)
    print("저장: trades_1d_4h.csv")

comparison_df.to_csv('mtf_comparison.csv', index=False)
print("저장: mtf_comparison.csv")

print("\n✨ 분석 완료!")
