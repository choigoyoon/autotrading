"""
MTF Zone 추출
- 1D/4H/1H L/H 라벨의 꼬리 범위를 지지/저항 구간으로 활용
- L 라벨: 저점 꼬리 = 지지 구간
- H 라벨: 고점 꼬리 = 저항 구간
- 15분 진입 시 MTF Zone과의 관계 분석
"""

import pandas as pd
import numpy as np

print("="*60)
print("MTF Zone 추출")
print("="*60)

# 데이터 로드
print("\n데이터 로드 중...")
df_15m = pd.read_csv('output_phase1_labeled.csv')
df_1h = pd.read_csv('btcusdt_1h_labeled.csv')
df_4h = pd.read_csv('btcusdt_4h_labeled.csv')
df_1d = pd.read_csv('btcusdt_1d_labeled.csv')
df_classified = pd.read_csv('output_mtf_situation_classified.csv')

for df in [df_15m, df_1h, df_4h, df_1d, df_classified]:
    df['datetime'] = pd.to_datetime(df['datetime'])

print(f"15분: {len(df_15m):,}개")
print(f"1시간: {len(df_1h):,}개")
print(f"4시간: {len(df_4h):,}개")
print(f"1일: {len(df_1d):,}개")

# Zone 정의: L/H 라벨의 꼬리 범위
def extract_zones_from_labels(df, timeframe_name):
    """L/H 라벨에서 Zone 추출"""

    zones = []

    # L 라벨: low ~ close (지지 구간)
    l_labels = df[df['label'] == 'L']
    for idx, row in l_labels.iterrows():
        zones.append({
            'datetime': row['datetime'],
            'type': 'support',
            'timeframe': timeframe_name,
            'zone_low': row['low'],
            'zone_high': row['close'],
            'pivot_price': row['low'],  # 핵심 가격
        })

    # H 라벨: close ~ high (저항 구간)
    h_labels = df[df['label'] == 'H']
    for idx, row in h_labels.iterrows():
        zones.append({
            'datetime': row['datetime'],
            'type': 'resistance',
            'timeframe': timeframe_name,
            'zone_low': row['close'],
            'zone_high': row['high'],
            'pivot_price': row['high'],  # 핵심 가격
        })

    return pd.DataFrame(zones)

# 각 타임프레임별 Zone 추출
print("\n" + "="*60)
print("타임프레임별 Zone 추출")
print("="*60)

zones_1h = extract_zones_from_labels(df_1h, '1H')
zones_4h = extract_zones_from_labels(df_4h, '4H')
zones_1d = extract_zones_from_labels(df_1d, '1D')

print(f"\n1시간 Zones: {len(zones_1h):,}개")
print(f"  지지: {(zones_1h['type'] == 'support').sum():,}개")
print(f"  저항: {(zones_1h['type'] == 'resistance').sum():,}개")

print(f"\n4시간 Zones: {len(zones_4h):,}개")
print(f"  지지: {(zones_4h['type'] == 'support').sum():,}개")
print(f"  저항: {(zones_4h['type'] == 'resistance').sum():,}개")

print(f"\n1일 Zones: {len(zones_1d):,}개")
print(f"  지지: {(zones_1d['type'] == 'support').sum():,}개")
print(f"  저항: {(zones_1d['type'] == 'resistance').sum():,}개")

# Zone 샘플
print("\n1일 Zone 샘플:")
print(zones_1d.head(3).to_string(index=False))

# Zone 폭 통계
for zones, name in [(zones_1h, '1시간'), (zones_4h, '4시간'), (zones_1d, '1일')]:
    zones['zone_width'] = (zones['zone_high'] - zones['zone_low']) / zones['pivot_price'] * 100

    print(f"\n{name} Zone 폭 통계:")
    print(f"  평균: {zones['zone_width'].mean():.3f}%")
    print(f"  중간값: {zones['zone_width'].median():.3f}%")
    print(f"  최대: {zones['zone_width'].max():.3f}%")
    print(f"  최소: {zones['zone_width'].min():.3f}%")

# 15분봉에 가장 가까운 MTF Zone 매칭
print("\n" + "="*60)
print("15분봉에 MTF Zone 매칭")
print("="*60)

def find_nearest_zones(dt, price, zones_df, max_distance_pct=10.0):
    """
    특정 시점/가격에서 가장 가까운 Zone 찾기
    나우캐스트: dt 이전의 Zone만 사용
    """

    # dt 이전의 Zone만
    available_zones = zones_df[zones_df['datetime'] < dt].copy()

    if len(available_zones) == 0:
        return None, None

    # 가격과의 거리 계산
    available_zones['distance'] = available_zones.apply(
        lambda row: abs(row['pivot_price'] - price) / price * 100,
        axis=1
    )

    # max_distance 이내의 Zone만
    nearby_zones = available_zones[available_zones['distance'] <= max_distance_pct]

    if len(nearby_zones) == 0:
        return None, None

    # 지지/저항 분리
    support_zones = nearby_zones[nearby_zones['type'] == 'support']
    resistance_zones = nearby_zones[nearby_zones['type'] == 'resistance']

    nearest_support = support_zones.loc[support_zones['distance'].idxmin()] if len(support_zones) > 0 else None
    nearest_resistance = resistance_zones.loc[resistance_zones['distance'].idxmin()] if len(resistance_zones) > 0 else None

    return nearest_support, nearest_resistance

# 15분봉 일부 샘플에 Zone 매칭
print("\n15분봉 샘플에 MTF Zone 매칭 (최근 1000개):")

sample_15m = df_classified.tail(1000)
zone_matches = []

for i, row in sample_15m.iterrows():
    if i % 200 == 0:
        print(f"  진행: {len(zone_matches)}/{len(sample_15m)}...", end='\r')

    dt = row['datetime']
    price = row['close']

    # 각 타임프레임별 가장 가까운 Zone
    sup_1h, res_1h = find_nearest_zones(dt, price, zones_1h, max_distance_pct=5.0)
    sup_4h, res_4h = find_nearest_zones(dt, price, zones_4h, max_distance_pct=10.0)
    sup_1d, res_1d = find_nearest_zones(dt, price, zones_1d, max_distance_pct=15.0)

    zone_matches.append({
        'datetime': dt,
        'price': price,
        'situation': row['situation'],
        # 1H
        'nearest_support_1h': sup_1h['pivot_price'] if sup_1h is not None else None,
        'nearest_resistance_1h': res_1h['pivot_price'] if res_1h is not None else None,
        'support_dist_1h': sup_1h['distance'] if sup_1h is not None else None,
        'resistance_dist_1h': res_1h['distance'] if res_1h is not None else None,
        # 4H
        'nearest_support_4h': sup_4h['pivot_price'] if sup_4h is not None else None,
        'nearest_resistance_4h': res_4h['pivot_price'] if res_4h is not None else None,
        'support_dist_4h': sup_4h['distance'] if sup_4h is not None else None,
        'resistance_dist_4h': res_4h['distance'] if res_4h is not None else None,
        # 1D
        'nearest_support_1d': sup_1d['pivot_price'] if sup_1d is not None else None,
        'nearest_resistance_1d': res_1d['pivot_price'] if res_1d is not None else None,
        'support_dist_1d': sup_1d['distance'] if sup_1d is not None else None,
        'resistance_dist_1d': res_1d['distance'] if res_1d is not None else None,
    })

print(" " * 50, end='\r')

zone_matches_df = pd.DataFrame(zone_matches)

# Zone 매칭률
print("\nZone 매칭률 (5%, 10%, 15% 이내):")
print(f"  1H 지지: {zone_matches_df['nearest_support_1h'].notna().sum() / len(zone_matches_df) * 100:.1f}%")
print(f"  1H 저항: {zone_matches_df['nearest_resistance_1h'].notna().sum() / len(zone_matches_df) * 100:.1f}%")
print(f"  4H 지지: {zone_matches_df['nearest_support_4h'].notna().sum() / len(zone_matches_df) * 100:.1f}%")
print(f"  4H 저항: {zone_matches_df['nearest_resistance_4h'].notna().sum() / len(zone_matches_df) * 100:.1f}%")
print(f"  1D 지지: {zone_matches_df['nearest_support_1d'].notna().sum() / len(zone_matches_df) * 100:.1f}%")
print(f"  1D 저항: {zone_matches_df['nearest_resistance_1d'].notna().sum() / len(zone_matches_df) * 100:.1f}%")

# Zone 활용 아이디어
print("\n" + "="*60)
print("Zone 활용 전략")
print("="*60)

print("\n1. 진입 필터:")
print("   - 롱: 현재가가 1H/4H 지지선 근처 (1% 이내)")
print("   - 숏: 현재가가 1H/4H 저항선 근처 (1% 이내)")
print("   → Zone에서 이탈 시 강한 모멘텀 기대")

print("\n2. 부분 익절:")
print("   - TP1 (50%): 1H 저항/지지 도달")
print("   - TP2 (25%): 4H 저항/지지 도달")
print("   - TP3 (15%): 1D 저항/지지 도달")
print("   - TP4 (10%): 트레일링")

print("\n3. 손절 조정:")
print("   - Zone 이탈 실패 시 빠른 손절")
print("   - Zone 통과 후 손절 이동 (브레이크이븐)")

# 샘플 출력
print("\n" + "="*60)
print("Zone 매칭 샘플 (최근 5개)")
print("="*60)

for idx, row in zone_matches_df.tail(5).iterrows():
    print(f"\n시간: {row['datetime']} (상황: {row['situation']})")
    print(f"  가격: {row['price']:.2f}")

    if pd.notna(row['nearest_support_1h']):
        print(f"  1H 지지: {row['nearest_support_1h']:.2f} ({row['support_dist_1h']:.2f}% 아래)")
    if pd.notna(row['nearest_resistance_1h']):
        print(f"  1H 저항: {row['nearest_resistance_1h']:.2f} ({row['resistance_dist_1h']:.2f}% 위)")

    if pd.notna(row['nearest_support_4h']):
        print(f"  4H 지지: {row['nearest_support_4h']:.2f} ({row['support_dist_4h']:.2f}% 아래)")
    if pd.notna(row['nearest_resistance_4h']):
        print(f"  4H 저항: {row['nearest_resistance_4h']:.2f} ({row['resistance_dist_4h']:.2f}% 위)")

# 저장
zones_1h.to_csv('mtf_zones_1h.csv', index=False)
zones_4h.to_csv('mtf_zones_4h.csv', index=False)
zones_1d.to_csv('mtf_zones_1d.csv', index=False)
zone_matches_df.to_csv('sample_zone_matches.csv', index=False)

print("\n저장 완료:")
print("  - mtf_zones_1h.csv")
print("  - mtf_zones_4h.csv")
print("  - mtf_zones_1d.csv")
print("  - sample_zone_matches.csv")

print("\n" + "="*60)
print("Zone 추출 완료")
print("="*60)

print("\n다음 단계:")
print("  1. 방향 필터 (상황별 롱/숏 제한)")
print("  2. Zone 기반 진입 필터 추가")
print("  3. 부분 익절 시스템 구현")
print("  4. 통합 백테스트")
