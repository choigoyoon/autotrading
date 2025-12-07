import pandas as pd
import numpy as np

# CSV 읽기
df = pd.read_csv('btc_15m_ohlcv.csv', parse_dates=['datetime'])
df = df.rename(columns={'datetime': 'timestamp'})
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"📊 데이터 로드: {len(df):,}개 캔들\n")

# Bollinger Bands 계산
def calculate_bb(df, window=20, num_std=2):
    df['bb_middle'] = df['close'].rolling(window=window).mean()
    df['bb_std'] = df['close'].rolling(window=window).std()
    df['bb_upper'] = df['bb_middle'] + (num_std * df['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (num_std * df['bb_std'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
    return df

df = calculate_bb(df)

# Swing High/Low 찾기
def find_swing_highs(df, window=10):
    highs = []
    for i in range(window, len(df) - window):
        if df.loc[i, 'high'] == df.loc[i-window:i+window+1, 'high'].max():
            highs.append({
                'index': i,
                'time': df.loc[i, 'timestamp'],
                'price': df.loc[i, 'high'],
                'type': 'H'
            })
    return highs

def find_swing_lows(df, window=10):
    lows = []
    for i in range(window, len(df) - window):
        if df.loc[i, 'low'] == df.loc[i-window:i+window+1, 'low'].min():
            lows.append({
                'index': i,
                'time': df.loc[i, 'timestamp'],
                'price': df.loc[i, 'low'],
                'type': 'L'
            })
    return lows

highs = find_swing_highs(df)
lows = find_swing_lows(df)

print(f"📈 H값: {len(highs):,}개")
print(f"📉 L값: {len(lows):,}개\n")

# HLHLHL 전체 라벨링 (시간순 정렬)
all_points = highs + lows
all_points = sorted(all_points, key=lambda x: x['index'])

print(f"🔢 전체 HL 포인트: {len(all_points):,}개\n")

# HL/LL, HH/LH 라벨링
labeled_points = []

for i, point in enumerate(all_points):
    if i == 0:
        labeled_points.append({
            **point,
            'label': point['type'],
            'pattern': 'start'
        })
        continue
    
    # 이전 같은 타입 찾기
    prev_same_type = None
    for j in range(i-1, -1, -1):
        if all_points[j]['type'] == point['type']:
            prev_same_type = all_points[j]
            break
    
    if prev_same_type:
        if point['type'] == 'H':
            # H값 패턴
            if point['price'] > prev_same_type['price']:
                pattern = 'HH'  # Higher High
            elif point['price'] < prev_same_type['price']:
                pattern = 'LH'  # Lower High
            else:
                pattern = 'EH'  # Equal High
        else:  # L값
            # L값 패턴
            if point['price'] > prev_same_type['price']:
                pattern = 'HL'  # Higher Low
            elif point['price'] < prev_same_type['price']:
                pattern = 'LL'  # Lower Low
            else:
                pattern = 'EL'  # Equal Low
    else:
        pattern = 'first'
    
    labeled_points.append({
        **point,
        'label': f"{point['type']}{i+1}",
        'pattern': pattern
    })

print(f"✅ HL 라벨링 완료!\n")

# 3단계 역추세 분석
counter_trend_opportunities = []

for i in range(len(labeled_points)):
    point = labeled_points[i]
    
    # 저점(L값)에서만 분석
    if point['type'] != 'L':
        continue
    
    # 1단계: 저점 확인
    # LL 패턴이면 저점 후보
    if point['pattern'] != 'LL':
        continue
    
    point_idx = point['index']
    point_time = point['time']
    point_price = point['price']
    
    # 2단계: 추세선 돌파 확인
    # 이 저점 이전 3개 H값으로 하락 추세선 만들기
    prev_highs = [p for p in labeled_points[:i] if p['type'] == 'H'][-3:]
    
    if len(prev_highs) < 3:
        continue
    
    # 하락 추세선인지 확인 (LH-LH-LH)
    is_downtrend = (
        prev_highs[0]['price'] > prev_highs[1]['price'] > prev_highs[2]['price']
    )
    
    if not is_downtrend:
        continue
    
    h1, h2, h3 = prev_highs[0], prev_highs[1], prev_highs[2]
    
    # 추세선 돌파 찾기
    trendline_break_idx = None
    trendline_break_price = None
    
    for idx in range(point_idx + 1, min(point_idx + 100, len(df))):
        if df.loc[idx, 'close'] > h3['price']:
            trendline_break_idx = idx
            trendline_break_price = df.loc[idx, 'close']
            break
    
    if not trendline_break_idx:
        continue
    
    trendline_break_time = df.loc[trendline_break_idx, 'timestamp']
    
    # 3단계: 저항선 돌파 확인 (H3 → H2 → H1 순서대로)
    h3_break = False
    h3_break_idx = None
    h2_break = False
    h2_break_idx = None
    h1_break = False
    h1_break_idx = None
    
    for idx in range(trendline_break_idx, min(trendline_break_idx + 200, len(df))):
        if not h3_break and df.loc[idx, 'high'] > h3['price']:
            h3_break = True
            h3_break_idx = idx
        if h3_break and not h2_break and df.loc[idx, 'high'] > h2['price']:
            h2_break = True
            h2_break_idx = idx
        if h2_break and not h1_break and df.loc[idx, 'high'] > h1['price']:
            h1_break = True
            h1_break_idx = idx
            break
    
    # 돌파할 힘 분석 (추세선 돌파 시점)
    tb_idx = trendline_break_idx
    
    # 1) BB 상단 찢기 확인
    bb_tear = False
    if not pd.isna(df.loc[tb_idx, 'bb_upper']):
        if df.loc[tb_idx, 'high'] > df.loc[tb_idx, 'bb_upper']:
            bb_tear = True
    
    # 2) FVG (Fair Value Gap) 확인
    # 3캔들 패턴: 1번 고점 < 3번 저점 (갭)
    fvg = False
    if tb_idx >= 2:
        candle_1_high = df.loc[tb_idx-2, 'high']
        candle_3_low = df.loc[tb_idx, 'low']
        if candle_1_high < candle_3_low:
            fvg = True
    
    # 3) OB (Order Block) 확인
    # 강한 양봉 (body > 1%)
    ob = False
    open_price = df.loc[tb_idx, 'open']
    close_price = df.loc[tb_idx, 'close']
    body_pct = (close_price / open_price - 1) * 100
    if body_pct > 1:
        ob = True
    
    # 4) 캔들 패턴 (강한 양봉)
    strong_bullish = body_pct > 2
    
    # 힘의 총점
    power_score = 0
    power_details = []
    
    if bb_tear:
        power_score += 3
        power_details.append('BB상단찢기')
    if fvg:
        power_score += 2
        power_details.append('FVG')
    if ob:
        power_score += 2
        power_details.append('OB')
    if strong_bullish:
        power_score += 3
        power_details.append('강양봉')
    
    # 판단: 할 수 있다 / 못한다
    if power_score >= 5 and h3_break:
        judgment = '매매 가능'
    elif power_score >= 3:
        judgment = '조건부 가능'
    else:
        judgment = '매매 불가'
    
    counter_trend_opportunities.append({
        'l_label': point['label'],
        'l_time': point_time,
        'l_price': point_price,
        'h1_price': h1['price'],
        'h2_price': h2['price'],
        'h3_price': h3['price'],
        'trendline_break_time': trendline_break_time,
        'trendline_break_price': trendline_break_price,
        'h3_break': h3_break,
        'h2_break': h2_break,
        'h1_break': h1_break,
        'power_score': power_score,
        'power_details': ', '.join(power_details),
        'bb_tear': bb_tear,
        'fvg': fvg,
        'ob': ob,
        'strong_bullish': strong_bullish,
        'judgment': judgment
    })

print(f"✅ 3단계 역추세 분석 완료!\n")

# DataFrame 생성
df_result = pd.DataFrame(counter_trend_opportunities)

if len(df_result) == 0:
    print("❌ 역추세 기회 없음!")
    exit()

# 통계
print(f"{'='*80}")
print(f"🎯 HLHLHL 전체 라벨링 + 3단계 역추세 분석")
print(f"{'='*80}\n")

print(f"총 역추세 기회: {len(df_result)}개\n")

# 판단별 분포
print(f"💡 판단별 분포:")
judgment_counts = df_result['judgment'].value_counts()
for judgment, count in judgment_counts.items():
    pct = count / len(df_result) * 100
    print(f"   {judgment}: {count}개 ({pct:.1f}%)")

# 저항선 돌파 통계
h3_break_count = df_result['h3_break'].sum()
h2_break_count = df_result['h2_break'].sum()
h1_break_count = df_result['h1_break'].sum()

print(f"\n📊 저항선 돌파 통계:")
print(f"   H3 돌파: {h3_break_count}개 ({h3_break_count/len(df_result)*100:.1f}%)")
print(f"   H2 돌파: {h2_break_count}개 ({h2_break_count/len(df_result)*100:.1f}%)")
print(f"   H1 돌파: {h1_break_count}개 ({h1_break_count/len(df_result)*100:.1f}%)")

# 힘의 요소별 통계
print(f"\n💪 힘의 요소별 통계:")
print(f"   BB 상단 찢기: {df_result['bb_tear'].sum()}개 ({df_result['bb_tear'].sum()/len(df_result)*100:.1f}%)")
print(f"   FVG: {df_result['fvg'].sum()}개 ({df_result['fvg'].sum()/len(df_result)*100:.1f}%)")
print(f"   OB: {df_result['ob'].sum()}개 ({df_result['ob'].sum()/len(df_result)*100:.1f}%)")
print(f"   강한 양봉: {df_result['strong_bullish'].sum()}개 ({df_result['strong_bullish'].sum()/len(df_result)*100:.1f}%)")

# 평균 힘 점수
avg_power = df_result['power_score'].mean()
print(f"\n⚡ 평균 힘 점수: {avg_power:.2f}점")

# 힘 점수별 성공률 (H2 돌파 기준)
print(f"\n🎯 힘 점수별 H2 돌파 성공률:")
for score in range(0, 11):
    subset = df_result[df_result['power_score'] == score]
    if len(subset) > 0:
        success_rate = subset['h2_break'].sum() / len(subset) * 100
        print(f"   점수 {score}점: {len(subset)}개 → H2 돌파 {success_rate:.1f}%")

# Top 10 매매 가능 사례
print(f"\n🔥 Top 10 '매매 가능' 사례:")
tradable = df_result[df_result['judgment'] == '매매 가능'].nlargest(10, 'power_score')
for idx, row in tradable.iterrows():
    print(f"\n{row['l_label']}: {row['l_time']}")
    print(f"   저점: ${row['l_price']:,.0f}")
    print(f"   추세선: H1 ${row['h1_price']:,.0f} → H2 ${row['h2_price']:,.0f} → H3 ${row['h3_price']:,.0f}")
    print(f"   돌파: {row['trendline_break_time']} @ ${row['trendline_break_price']:,.0f}")
    print(f"   힘: {row['power_score']}점 ({row['power_details']})")
    print(f"   저항 돌파: H3 {'✅' if row['h3_break'] else '❌'} | H2 {'✅' if row['h2_break'] else '❌'} | H1 {'✅' if row['h1_break'] else '❌'}")

# CSV 저장
df_result.to_csv('hlhlhl_full_labeling_analysis.csv', index=False)
print(f"\n✅ 결과 저장: hlhlhl_full_labeling_analysis.csv")

# 핵심 인사이트
print(f"\n{'='*80}")
print(f"💡 핵심 인사이트")
print(f"{'='*80}")

print(f"\n역추세 3단계:")
print(f"1️⃣ 저점 확인 (LL 패턴)")
print(f"2️⃣ 추세선 돌파 (H3 돌파)")
print(f"3️⃣ 저항선 돌파 (H3 → H2 → H1)")

print(f"\n돌파할 힘:")
print(f"✅ BB 상단 찢기: +3점")
print(f"✅ FVG (Fair Value Gap): +2점")
print(f"✅ OB (Order Block): +2점")
print(f"✅ 강한 양봉 (>2%): +3점")

print(f"\n판단 기준:")
print(f"✅ 5점 이상 + H3 돌파 = 매매 가능")
print(f"⚠️ 3~4점 = 조건부 가능")
print(f"❌ 3점 미만 = 매매 불가")

tradable_count = len(df_result[df_result['judgment'] == '매매 가능'])
print(f"\n🎯 결론:")
print(f"   매매 가능 기회: {tradable_count}개 ({tradable_count/len(df_result)*100:.1f}%)")
print(f"   평균 힘 점수: {avg_power:.2f}점")

