import pandas as pd
import numpy as np

# 데이터 로드
btc_df = pd.read_csv('btc_1h_ohlcv.csv')
btc_df['datetime'] = pd.to_datetime(btc_df['datetime'])
btc_df = btc_df.sort_values('datetime').reset_index(drop=True)

squeeze_df = pd.read_csv('expanded_squeeze_analysis.csv')
squeeze_df['datetime'] = pd.to_datetime(squeeze_df['datetime'])

print("="*80)
print("출구 전략 정교화 분석")
print("="*80)
print(f"\n총 엔트리 포인트: {len(squeeze_df)}건")

# 최고 성과 전략들만 테스트
test_strategies = [
    ('M100≥20%', squeeze_df['momentum_100'] >= 20),
    ('M100≥15% + HH', (squeeze_df['momentum_100'] >= 15) & (squeeze_df['HH'] == True)),
    ('M200≥15% + 정배열', (squeeze_df['momentum_200'] >= 15) & (squeeze_df['ema_bull'] == True)),
    ('M200≥10% + M100≥10%', (squeeze_df['momentum_200'] >= 10) & (squeeze_df['momentum_100'] >= 10)),
]

# BB 계산 함수
def calculate_bb(df, window=30, num_std=2):
    df['bb_mid'] = df['close'].rolling(window=window).mean()
    df['bb_std'] = df['close'].rolling(window=window).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * num_std)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * num_std)
    return df

btc_df = calculate_bb(btc_df)

# 모멘텀 계산
btc_df['momentum_50'] = ((btc_df['close'] - btc_df['close'].shift(50)) / btc_df['close'].shift(50)) * 100
btc_df['momentum_100'] = ((btc_df['close'] - btc_df['close'].shift(100)) / btc_df['close'].shift(100)) * 100

# 출구 전략 함수들
def exit_trailing_stop(entry_idx, entry_price, max_hours=336, trail_pct=5):
    """트레일링 스탑: 최고가 대비 N% 하락 시 청산"""
    max_price = entry_price
    for i in range(1, max_hours+1):
        if entry_idx + i >= len(btc_df):
            return i-1, btc_df.iloc[entry_idx + i - 1]['close']
        
        current_price = btc_df.iloc[entry_idx + i]['close']
        max_price = max(max_price, current_price)
        
        # 최고가 대비 trail_pct% 하락 시 청산
        if current_price < max_price * (1 - trail_pct/100):
            return i, current_price
    
    # 최대 기간 도달
    return max_hours, btc_df.iloc[entry_idx + max_hours]['close']

def exit_momentum_reversal(entry_idx, max_hours=336, momentum_threshold=-5):
    """모멘텀 반전: M50 < -5% 시 청산"""
    for i in range(1, max_hours+1):
        if entry_idx + i >= len(btc_df):
            return i-1, btc_df.iloc[entry_idx + i - 1]['close']
        
        m50 = btc_df.iloc[entry_idx + i]['momentum_50']
        if pd.notna(m50) and m50 < momentum_threshold:
            return i, btc_df.iloc[entry_idx + i]['close']
    
    return max_hours, btc_df.iloc[entry_idx + max_hours]['close']

def exit_bb_upper_touch(entry_idx, max_hours=336):
    """BB 상단 밴드 터치 시 청산"""
    for i in range(1, max_hours+1):
        if entry_idx + i >= len(btc_df):
            return i-1, btc_df.iloc[entry_idx + i - 1]['close']
        
        current_price = btc_df.iloc[entry_idx + i]['close']
        bb_upper = btc_df.iloc[entry_idx + i]['bb_upper']
        
        if pd.notna(bb_upper) and current_price >= bb_upper:
            return i, current_price
    
    return max_hours, btc_df.iloc[entry_idx + max_hours]['close']

def exit_composite(entry_idx, entry_price, max_hours=336):
    """복합 출구: 트레일링(7%) OR BB상단터치 OR M50<-5"""
    max_price = entry_price
    
    for i in range(1, max_hours+1):
        if entry_idx + i >= len(btc_df):
            return i-1, btc_df.iloc[entry_idx + i - 1]['close'], 'MAX_TIME'
        
        current_price = btc_df.iloc[entry_idx + i]['close']
        max_price = max(max_price, current_price)
        
        # 조건1: 트레일링 스탑 (7%)
        if current_price < max_price * 0.93:
            return i, current_price, 'TRAILING'
        
        # 조건2: BB 상단 터치
        bb_upper = btc_df.iloc[entry_idx + i]['bb_upper']
        if pd.notna(bb_upper) and current_price >= bb_upper:
            return i, current_price, 'BB_UPPER'
        
        # 조건3: 모멘텀 반전
        m50 = btc_df.iloc[entry_idx + i]['momentum_50']
        if pd.notna(m50) and m50 < -5:
            return i, current_price, 'MOMENTUM'
    
    return max_hours, btc_df.iloc[entry_idx + max_hours]['close'], 'MAX_TIME'

# 각 전략별로 출구 전략 테스트
all_results = []

for strategy_name, strategy_cond in test_strategies:
    subset = squeeze_df[strategy_cond].copy()
    if len(subset) == 0:
        continue
    
    print(f"\n{'='*80}")
    print(f"전략: {strategy_name} ({len(subset)}건)")
    print(f"{'='*80}")
    
    # 기존 방식 (고정 홀딩)
    win_168 = (subset['long_168h'] > 0).sum()
    win_336 = (subset['long_336h'] > 0).sum()
    
    print(f"\n[기존] 고정 홀딩:")
    print(f"  168h: 승률 {win_168/len(subset)*100:.1f}%, 평균 {subset['long_168h'].mean():.2f}%")
    print(f"  336h: 승률 {win_336/len(subset)*100:.1f}%, 평균 {subset['long_336h'].mean():.2f}%")
    
    # 동적 출구 전략들
    exit_strategies = {
        'Trailing_5%': lambda idx, price: exit_trailing_stop(idx, price, 336, 5),
        'Trailing_7%': lambda idx, price: exit_trailing_stop(idx, price, 336, 7),
        'Trailing_10%': lambda idx, price: exit_trailing_stop(idx, price, 336, 10),
        'Momentum_-5': lambda idx, price: exit_momentum_reversal(idx, 336, -5),
        'Momentum_-10': lambda idx, price: exit_momentum_reversal(idx, 336, -10),
        'BB_Upper': lambda idx, price: exit_bb_upper_touch(idx, 336),
    }
    
    exit_results = {}
    
    for exit_name, exit_func in exit_strategies.items():
        profits = []
        holding_times = []
        
        for _, entry in subset.iterrows():
            entry_dt = entry['datetime']
            entry_idx = btc_df[btc_df['datetime'] == entry_dt].index[0]
            entry_price = btc_df.iloc[entry_idx]['close']
            
            if exit_name in ['Trailing_5%', 'Trailing_7%', 'Trailing_10%']:
                hours, exit_price = exit_func(entry_idx, entry_price)
            else:
                hours, exit_price = exit_func(entry_idx, entry_price)
            
            profit = ((exit_price - entry_price) / entry_price) * 100
            profits.append(profit)
            holding_times.append(hours)
        
        wins = sum(1 for p in profits if p > 0)
        exit_results[exit_name] = {
            'win_rate': wins / len(profits) * 100,
            'avg_profit': np.mean(profits),
            'avg_holding': np.mean(holding_times),
            'total_profit': sum(profits)
        }
    
    # 복합 전략
    composite_profits = []
    composite_holding = []
    composite_reasons = []
    
    for _, entry in subset.iterrows():
        entry_dt = entry['datetime']
        entry_idx = btc_df[btc_df['datetime'] == entry_dt].index[0]
        entry_price = btc_df.iloc[entry_idx]['close']
        
        hours, exit_price, reason = exit_composite(entry_idx, entry_price, 336)
        profit = ((exit_price - entry_price) / entry_price) * 100
        composite_profits.append(profit)
        composite_holding.append(hours)
        composite_reasons.append(reason)
    
    comp_wins = sum(1 for p in composite_profits if p > 0)
    exit_results['Composite'] = {
        'win_rate': comp_wins / len(composite_profits) * 100,
        'avg_profit': np.mean(composite_profits),
        'avg_holding': np.mean(composite_holding),
        'total_profit': sum(composite_profits)
    }
    
    # 결과 출력
    print(f"\n[동적 출구 전략]:")
    results_list = []
    for name, res in exit_results.items():
        results_list.append({
            '출구전략': name,
            '승률': f"{res['win_rate']:.1f}%",
            '평균수익': f"{res['avg_profit']:.2f}%",
            '평균홀딩': f"{res['avg_holding']:.0f}h",
            '총수익': f"{res['total_profit']:.1f}%"
        })
    
    results_df = pd.DataFrame(results_list)
    print(results_df.to_string(index=False))
    
    # 전략별 결과 저장
    for _, row in results_df.iterrows():
        all_results.append({
            '전략': strategy_name,
            '건수': len(subset),
            '출구전략': row['출구전략'],
            '승률': row['승률'],
            '평균수익': row['평균수익'],
            '평균홀딩': row['평균홀딩'],
            '총수익': row['총수익']
        })

# 전체 결과 저장
final_df = pd.DataFrame(all_results)
final_df.to_csv('exit_strategy_results.csv', index=False, encoding='utf-8-sig')

print("\n" + "="*80)
print("종합 분석 완료")
print("="*80)
print("\n✅ exit_strategy_results.csv 저장 완료")

# 최고 성과 출구 전략
print("\n" + "="*80)
print("최고 성과 출구 전략 TOP 10")
print("="*80)
final_df['평균수익_val'] = final_df['평균수익'].str.rstrip('%').astype(float)
top10 = final_df.nlargest(10, '평균수익_val').drop('평균수익_val', axis=1)
print(top10.to_string(index=False))

