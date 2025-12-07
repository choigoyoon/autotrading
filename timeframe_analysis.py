import pandas as pd
import numpy as np

print("="*80)
print("다중 타임프레임 전략 검증")
print("="*80)

# 데이터 로드
btc_4h = pd.read_csv('btc_4h_ohlcv.csv')
btc_1d = pd.read_csv('btc_1d_ohlcv.csv')

# BB 계산
def calculate_bb(df, window=30, num_std=2):
    df['bb_mid'] = df['close'].rolling(window=window).mean()
    df['bb_std'] = df['close'].rolling(window=window).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * num_std)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * num_std)
    df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_mid']) * 100
    return df

def detect_squeeze(df, window=50):
    df['bb_width_ma'] = df['bb_width'].rolling(window=window).mean()
    df['is_squeeze'] = df['bb_width'] < df['bb_width_ma']
    
    # 수축 종료 (squeeze -> expansion)
    df['squeeze_end'] = (df['is_squeeze'].shift(1) == True) & (df['is_squeeze'] == False)
    return df

# 각 타임프레임별 분석
results_summary = []

for data, timeframe, candle_name in [(btc_4h, '4H', '4시간봉'), (btc_1d, '1D', '일봉')]:
    print(f"\n{'='*80}")
    print(f"{candle_name} 분석")
    print(f"{'='*80}")
    
    df = data.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 타임프레임별 모멘텀 기간 설정
    if timeframe == '4H':
        # 4시간봉: 25봉 = 약 4일, 50봉 = 약 8일
        m_periods = [12, 25, 50]
        ema_periods = [5, 12, 50]
        hh_window = 25
        holding_periods = [42, 84]  # 1주(42*4h=168h), 2주(84*4h=336h)
        period_names = ['1주', '2주']
    else:  # 1D
        # 일봉: 4봉 = 4일, 8봉 = 8일
        m_periods = [4, 8, 16]
        ema_periods = [2, 5, 20]
        hh_window = 10
        holding_periods = [7, 14]  # 1주, 2주
        period_names = ['1주', '2주']
    
    # 지표 계산
    df = calculate_bb(df)
    df = detect_squeeze(df)
    
    # 모멘텀
    for p in m_periods:
        df[f'momentum_{p}'] = ((df['close'] - df['close'].shift(p)) / df['close'].shift(p)) * 100
    
    # EMA 정배열
    for p in ema_periods:
        df[f'ema{p}'] = df['close'].ewm(span=p, adjust=False).mean()
    df['ema_bull'] = (df[f'ema{ema_periods[0]}'] > df[f'ema{ema_periods[1]}']) & \
                     (df[f'ema{ema_periods[1]}'] > df[f'ema{ema_periods[2]}'])
    
    # HH
    df['rolling_high'] = df['high'].rolling(window=hh_window).max()
    df['prev_rolling_high'] = df['rolling_high'].shift(hh_window)
    df['HH'] = df['rolling_high'] > df['prev_rolling_high']
    
    # 수축 종료 시점들
    squeeze_indices = df[df['squeeze_end'] == True].index.tolist()
    print(f"\n수축→확장 포인트: {len(squeeze_indices)}개")
    
    if len(squeeze_indices) == 0:
        print("분석 가능한 포인트 없음")
        continue
    
    # 전략 테스트
    m_short = m_periods[0]
    m_mid = m_periods[1]
    m_long = m_periods[2]
    
    strategies = [
        (f'M{m_long}≥20%', lambda row: row[f'momentum_{m_long}'] >= 20),
        (f'M{m_long}≥15% + 정배열', 
         lambda row: (row[f'momentum_{m_long}'] >= 15) and row['ema_bull']),
        (f'M{m_long}≥10% + M{m_mid}≥10%',
         lambda row: (row[f'momentum_{m_long}'] >= 10) and (row[f'momentum_{m_mid}'] >= 10)),
        (f'M{m_mid}≥20%', lambda row: row[f'momentum_{m_mid}'] >= 20),
        (f'M{m_mid}≥15% + HH',
         lambda row: (row[f'momentum_{m_mid}'] >= 15) and row['HH']),
    ]
    
    print(f"\n{candle_name} 전략 테스트 결과:")
    print("-" * 80)
    
    for strategy_name, strategy_func in strategies:
        # 조건 충족하는 엔트리 찾기
        valid_entries = []
        for idx in squeeze_indices:
            row = df.loc[idx]
            try:
                if strategy_func(row):
                    valid_entries.append(idx)
            except:
                continue
        
        if len(valid_entries) == 0:
            continue
        
        # 각 홀딩 기간별로 계산
        for hp, hp_name in zip(holding_periods, period_names):
            profits = []
            for entry_idx in valid_entries:
                if entry_idx + hp < len(df):
                    entry_price = df.loc[entry_idx, 'close']
                    exit_price = df.loc[entry_idx + hp, 'close']
                    profit = ((exit_price - entry_price) / entry_price) * 100
                    profits.append(profit)
            
            if len(profits) == 0:
                continue
            
            wins = sum(1 for p in profits if p > 0)
            win_rate = wins / len(profits) * 100
            avg_profit = np.mean(profits)
            total_profit = sum(profits)
            
            results_summary.append({
                '타임프레임': candle_name,
                '전략': strategy_name,
                '건수': len(profits),
                '연간': f"{len(profits) / 5:.1f}",
                '홀딩': hp_name,
                '승률': f"{win_rate:.1f}%",
                '평균수익': f"{avg_profit:.2f}%",
                '총수익': f"{total_profit:.1f}%"
            })

# 결과 요약
if results_summary:
    print("\n" + "="*80)
    print("타임프레임별 전략 성과 종합")
    print("="*80)
    
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv('timeframe_strategy_results.csv', index=False, encoding='utf-8-sig')
    
    # 타임프레임별로 정렬
    for tf in ['4시간봉', '일봉']:
        tf_data = summary_df[summary_df['타임프레임'] == tf]
        if len(tf_data) > 0:
            print(f"\n[{tf}]")
            print(tf_data.to_string(index=False))
    
    print("\n✅ timeframe_strategy_results.csv 저장 완료")
    
    # 최고 성과
    print("\n" + "="*80)
    print("타임프레임별 최고 성과 전략 TOP 15")
    print("="*80)
    summary_df['평균수익_val'] = summary_df['평균수익'].str.rstrip('%').astype(float)
    top_strategies = summary_df.nlargest(15, '평균수익_val').drop('평균수익_val', axis=1)
    print(top_strategies.to_string(index=False))
else:
    print("\n분석 가능한 결과 없음")

print("\n" + "="*80)
print("분석 완료")
print("="*80)
