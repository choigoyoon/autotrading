import pandas as pd
import numpy as np
from datetime import timedelta

# CSV 읽기
df_15m = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df_15m = df_15m.rename(columns={'datetime': 'timestamp'})
df_15m = df_15m.sort_values('timestamp').reset_index(drop=True)

df_1h = pd.read_csv('btc_1h_ohlcv.csv', parse_dates=['datetime'])
df_1h = df_1h.rename(columns={'datetime': 'timestamp'})
df_1h = df_1h.sort_values('timestamp').reset_index(drop=True)

df_4h = pd.read_csv('btc_4h_ohlcv.csv', parse_dates=['datetime'])
df_4h = df_4h.rename(columns={'datetime': 'timestamp'})
df_4h = df_4h.sort_values('timestamp').reset_index(drop=True)

print(f"📊 데이터 로드 완료")
print(f"   15분봉: {len(df_15m):,}개 캔들")
print(f"   1시간봉: {len(df_1h):,}개 캔들")
print(f"   4시간봉: {len(df_4h):,}개 캔들\n")

# Bollinger Bands 계산
def calculate_bb(df, window=20, num_std=2):
    df['bb_middle'] = df['close'].rolling(window=window).mean()
    df['bb_std'] = df['close'].rolling(window=window).std()
    df['bb_upper'] = df['bb_middle'] + (num_std * df['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (num_std * df['bb_std'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
    return df

df_15m = calculate_bb(df_15m)
df_1h = calculate_bb(df_1h)
df_4h = calculate_bb(df_4h)

print(f"✅ Bollinger Bands 계산 완료\n")

# Swing High 찾기
def find_swing_highs(df, window=10):
    highs = []
    for i in range(window, len(df) - window):
        if df.loc[i, 'high'] == df.loc[i-window:i+window+1, 'high'].max():
            highs.append({
                'index': i,
                'time': df.loc[i, 'timestamp'],
                'price': df.loc[i, 'high']
            })
    return highs

highs = find_swing_highs(df_15m)
print(f"📈 총 {len(highs):,}개 H값 발견\n")

# LH (Lower High) 찾기
lh_values = []
for i in range(1, len(highs)):
    prev_h = highs[i-1]
    curr_h = highs[i]
    
    if curr_h['price'] < prev_h['price']:
        lh_pct = (curr_h['price'] / prev_h['price'] - 1) * 100
        lh_values.append({
            'prev_h_price': prev_h['price'],
            'prev_h_time': prev_h['time'],
            'lh_price': curr_h['price'],
            'lh_time': curr_h['time'],
            'lh_pct': lh_pct,
            'index': curr_h['index']
        })

print(f"🔴 총 {len(lh_values):,}개 LH값 발견\n")

# MTF 추세 확인 함수
def get_mtf_trend(timestamp, df_1h, df_4h):
    # 1시간봉 추세
    df_1h_sub = df_1h[df_1h['timestamp'] <= timestamp].tail(20)
    if len(df_1h_sub) < 20:
        trend_1h = 'unknown'
    else:
        recent_highs_1h = [df_1h_sub.iloc[i]['high'] for i in range(-3, 0)]
        if all(recent_highs_1h[i] > recent_highs_1h[i-1] for i in range(1, len(recent_highs_1h))):
            trend_1h = 'uptrend'
        elif all(recent_highs_1h[i] < recent_highs_1h[i-1] for i in range(1, len(recent_highs_1h))):
            trend_1h = 'downtrend'
        else:
            trend_1h = 'sideways'
    
    # 4시간봉 추세
    df_4h_sub = df_4h[df_4h['timestamp'] <= timestamp].tail(20)
    if len(df_4h_sub) < 20:
        trend_4h = 'unknown'
    else:
        recent_highs_4h = [df_4h_sub.iloc[i]['high'] for i in range(-3, 0)]
        if all(recent_highs_4h[i] > recent_highs_4h[i-1] for i in range(1, len(recent_highs_4h))):
            trend_4h = 'uptrend'
        elif all(recent_highs_4h[i] < recent_highs_4h[i-1] for i in range(1, len(recent_highs_4h))):
            trend_4h = 'downtrend'
        else:
            trend_4h = 'sideways'
    
    return trend_1h, trend_4h

# 각 LH값 분석
lh_analysis = []

for lh_idx, lh in enumerate(lh_values[:2000], 1):  # 전체 분석 (시간 고려하여 2000개)
    if lh_idx % 500 == 0:
        print(f"⏳ LH 분석 중... {lh_idx}/{len(lh_values[:2000])}")
    
    lh_time = lh['lh_time']
    lh_price = lh['lh_price']
    lh_index = lh['index']
    
    # BB 상태
    bb_row = df_15m.loc[lh_index]
    bb_upper = bb_row['bb_upper']
    bb_lower = bb_row['bb_lower']
    bb_middle = bb_row['bb_middle']
    bb_width = bb_row['bb_width']
    
    if pd.isna(bb_upper):
        bb_position = 'unknown'
    elif lh_price > bb_upper:
        bb_position = 'above_upper'
    elif lh_price < bb_lower:
        bb_position = 'below_lower'
    elif lh_price > bb_middle:
        bb_position = 'upper_half'
    else:
        bb_position = 'lower_half'
    
    # BB 찢김 여부 (폭이 좁으면 squeeze)
    if pd.isna(bb_width):
        bb_squeeze = 'unknown'
    elif bb_width < 2:
        bb_squeeze = 'tight'
    elif bb_width > 5:
        bb_squeeze = 'wide'
    else:
        bb_squeeze = 'normal'
    
    # MTF 추세
    trend_1h, trend_4h = get_mtf_trend(lh_time, df_1h, df_4h)
    
    # LH 이후 가격 움직임 (다음 50캔들 = 12.5시간)
    next_50_candles = df_15m.loc[lh_index+1:lh_index+51]
    if len(next_50_candles) < 50:
        result = 'insufficient_data'
        max_gain = 0
        max_drop = 0
    else:
        max_price = next_50_candles['high'].max()
        min_price = next_50_candles['low'].min()
        
        max_gain = (max_price / lh_price - 1) * 100
        max_drop = (min_price / lh_price - 1) * 100
        
        if max_drop < -2:  # 2% 이상 하락
            result = 'drop'
        elif max_gain > 2:  # 2% 이상 상승
            result = 'rise'
        else:
            result = 'sideways'
    
    # 롱/숏 비중 계산
    # 기본 점수: LH = 숏 편향 (40점)
    short_score = 40
    long_score = 0
    
    # BB 점수
    if bb_position == 'above_upper':
        short_score += 20  # BB 상단 = 과매수 = 숏
    elif bb_position == 'below_lower':
        long_score += 10  # BB 하단 = 과매도 = 롱 (하지만 LH이므로 약하게)
    elif bb_position == 'upper_half':
        short_score += 10
    
    if bb_squeeze == 'tight':
        short_score += 5  # 찢김 = 큰 움직임 준비
    
    # MTF 점수
    if trend_1h == 'downtrend':
        short_score += 15
    elif trend_1h == 'uptrend':
        long_score += 10
    
    if trend_4h == 'downtrend':
        short_score += 20
    elif trend_4h == 'uptrend':
        long_score += 15
    
    # 총점 계산
    total_score = short_score + long_score
    short_ratio = short_score / total_score * 100 if total_score > 0 else 0
    long_ratio = long_score / total_score * 100 if total_score > 0 else 0
    
    # 최종 판단
    if short_ratio >= 70:
        judgment = '강력 숏'
    elif short_ratio >= 60:
        judgment = '숏 유리'
    elif short_ratio >= 50:
        judgment = '약 숏'
    elif long_ratio >= 60:
        judgment = '롱 유리'
    else:
        judgment = '중립'
    
    lh_analysis.append({
        'lh_num': lh_idx,
        'lh_time': lh_time,
        'lh_price': lh_price,
        'lh_pct': lh['lh_pct'],
        'bb_position': bb_position,
        'bb_squeeze': bb_squeeze,
        'bb_width': bb_width,
        'trend_1h': trend_1h,
        'trend_4h': trend_4h,
        'short_score': short_score,
        'long_score': long_score,
        'short_ratio': short_ratio,
        'long_ratio': long_ratio,
        'judgment': judgment,
        'result': result,
        'max_gain': max_gain,
        'max_drop': max_drop
    })

print(f"\n✅ LH 분석 완료!\n")

# DataFrame 생성
df_lh = pd.DataFrame(lh_analysis)

# 통계
print(f"{'='*80}")
print(f"📊 LH값 롱/숏 비중 종합 분석 리포트")
print(f"{'='*80}\n")

print(f"총 LH값: {len(df_lh):,}개\n")

# 판단별 통계
print(f"💡 판단별 분포:")
for judgment in ['강력 숏', '숏 유리', '약 숏', '중립', '롱 유리']:
    count = len(df_lh[df_lh['judgment'] == judgment])
    pct = count / len(df_lh) * 100
    print(f"   {judgment}: {count:,}개 ({pct:.1f}%)")

print(f"\n📈 실제 결과 분포:")
result_counts = df_lh['result'].value_counts()
for result, count in result_counts.items():
    pct = count / len(df_lh) * 100
    print(f"   {result}: {count:,}개 ({pct:.1f}%)")

# 판단별 승률
print(f"\n🎯 판단별 승률:")
for judgment in ['강력 숏', '숏 유리', '약 숏']:
    df_sub = df_lh[df_lh['judgment'] == judgment]
    if len(df_sub) > 0:
        win_count = len(df_sub[df_sub['result'] == 'drop'])
        win_rate = win_count / len(df_sub) * 100
        avg_drop = df_sub[df_sub['result'] == 'drop']['max_drop'].mean()
        print(f"   {judgment}: {win_count}/{len(df_sub)} = {win_rate:.1f}% (평균 하락: {avg_drop:.2f}%)")

# MTF 조합별 승률
print(f"\n🌐 MTF 조합별 승률 (숏 기준):")
mtf_combinations = df_lh.groupby(['trend_1h', 'trend_4h'])
for (trend_1h, trend_4h), group in mtf_combinations:
    if len(group) > 10:  # 최소 10개 이상
        win_count = len(group[group['result'] == 'drop'])
        win_rate = win_count / len(group) * 100
        avg_short_ratio = group['short_ratio'].mean()
        print(f"   1H: {trend_1h:10s} | 4H: {trend_4h:10s} → 숏 승률: {win_rate:5.1f}% (평균 숏 비중: {avg_short_ratio:.1f}%)")

# BB 위치별 승률
print(f"\n📊 BB 위치별 승률 (숏 기준):")
bb_groups = df_lh.groupby('bb_position')
for bb_pos, group in bb_groups:
    if len(group) > 10:
        win_count = len(group[group['result'] == 'drop'])
        win_rate = win_count / len(group) * 100
        avg_short_ratio = group['short_ratio'].mean()
        print(f"   {bb_pos:15s} → 숏 승률: {win_rate:5.1f}% (평균 숏 비중: {avg_short_ratio:.1f}%)")

# Top 10 강력 숏 사례
print(f"\n🔥 Top 10 강력 숏 사례:")
top_short = df_lh.nlargest(10, 'short_ratio')
for idx, row in top_short.iterrows():
    print(f"   {row['lh_time']} | LH ${row['lh_price']:,.0f} | 숏비중 {row['short_ratio']:.1f}% | {row['judgment']}")
    print(f"      BB: {row['bb_position']}, 1H: {row['trend_1h']}, 4H: {row['trend_4h']} → 결과: {row['result']} (하락 {row['max_drop']:.2f}%)")

# CSV 저장
df_lh.to_csv('lh_long_short_ratio_analysis.csv', index=False)
print(f"\n✅ 결과 저장: lh_long_short_ratio_analysis.csv")

# 요약 통계
print(f"\n{'='*80}")
print(f"📌 핵심 요약")
print(f"{'='*80}")
strong_short = df_lh[df_lh['judgment'] == '강력 숏']
if len(strong_short) > 0:
    strong_short_win = len(strong_short[strong_short['result'] == 'drop'])
    strong_short_winrate = strong_short_win / len(strong_short) * 100
    print(f"✅ '강력 숏' 신호: {len(strong_short)}개 | 승률: {strong_short_winrate:.1f}%")
    print(f"   평균 숏 비중: {strong_short['short_ratio'].mean():.1f}%")
    print(f"   평균 하락폭: {strong_short[strong_short['result']=='drop']['max_drop'].mean():.2f}%")

downtrend_both = df_lh[(df_lh['trend_1h'] == 'downtrend') & (df_lh['trend_4h'] == 'downtrend')]
if len(downtrend_both) > 0:
    dt_win = len(downtrend_both[downtrend_both['result'] == 'drop'])
    dt_winrate = dt_win / len(downtrend_both) * 100
    print(f"\n✅ MTF 둘 다 하락: {len(downtrend_both)}개 | 승률: {dt_winrate:.1f}%")
    print(f"   평균 숏 비중: {downtrend_both['short_ratio'].mean():.1f}%")

