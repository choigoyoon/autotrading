import pandas as pd
import numpy as np

feat_df = pd.read_csv('bb30_pattern_features.csv')
print(f"총 케이스: {len(feat_df)}개")
print(f"기본 승률: {feat_df['win'].mean()*100:.1f}%, 평균 수익: {feat_df['pnl'].mean():+.2f}%")

print("\n" + "="*80)
print("🔍 주목할 패턴 심층 분석")
print("="*80)

# 발견된 좋은 패턴들
print("\n### 1. DOWN + 중간(30-70) = 승률 59.8%, +0.44%")
sub = feat_df[(feat_df['last_1_dir'] == -1) & 
              (feat_df['last_bb_pos'] >= 30) & 
              (feat_df['last_bb_pos'] < 70)]
print(f"   케이스: {len(sub)}건")
print(f"   돌파방향 분포: UP {(sub['break_dir']==1).sum()}, DOWN {(sub['break_dir']==-1).sum()}")

# 이 패턴에서 돌파 방향별 수익
up_break = sub[sub['break_dir'] == 1]
down_break = sub[sub['break_dir'] == -1]
print(f"   UP돌파 시: {len(up_break)}건, 승률 {up_break['win'].mean()*100:.1f}%, 평균 {up_break['pnl'].mean():+.2f}%")
print(f"   DOWN돌파 시: {len(down_break)}건, 승률 {down_break['win'].mean()*100:.1f}%, 평균 {down_break['pnl'].mean():+.2f}%")

print("\n### 2. 중하단(20-40) = 승률 55.1%, +0.22%")
sub = feat_df[(feat_df['last_bb_pos'] >= 20) & (feat_df['last_bb_pos'] < 40)]
print(f"   케이스: {len(sub)}건")
up_break = sub[sub['break_dir'] == 1]
down_break = sub[sub['break_dir'] == -1]
print(f"   UP돌파 시: {len(up_break)}건, 승률 {up_break['win'].mean()*100:.1f}%, 평균 {up_break['pnl'].mean():+.2f}%")
print(f"   DOWN돌파 시: {len(down_break)}건, 승률 {down_break['win'].mean()*100:.1f}%, 평균 {down_break['pnl'].mean():+.2f}%")

# 추가 조건 탐색
print("\n" + "="*80)
print("🎯 3중 조건 조합 탐색")
print("="*80)

best_patterns = []

# 방향 + 위치 + 수축강도
width_med = feat_df['min_width'].median()
for d in [1, -1]:
    dir_label = 'UP' if d == 1 else 'DOWN'
    for pos_low, pos_high, pos_label in [(0, 30, '하단'), (30, 50, '중하단'), (50, 70, '중상단'), (70, 100, '상단')]:
        for width_cond, width_label in [('strong', '강한수축'), ('weak', '약한수축')]:
            if width_cond == 'strong':
                sub = feat_df[(feat_df['last_1_dir'] == d) & 
                              (feat_df['last_bb_pos'] >= pos_low) & 
                              (feat_df['last_bb_pos'] < pos_high) &
                              (feat_df['min_width'] < width_med)]
            else:
                sub = feat_df[(feat_df['last_1_dir'] == d) & 
                              (feat_df['last_bb_pos'] >= pos_low) & 
                              (feat_df['last_bb_pos'] < pos_high) &
                              (feat_df['min_width'] >= width_med)]
            
            if len(sub) >= 20:
                win_rate = sub['win'].mean() * 100
                avg_pnl = sub['pnl'].mean()
                if win_rate >= 55 or avg_pnl >= 0.3:
                    best_patterns.append({
                        'pattern': f"{dir_label} + {pos_label} + {width_label}",
                        'count': len(sub),
                        'win_rate': win_rate,
                        'avg_pnl': avg_pnl
                    })
                    print(f"  ✓ {dir_label} + {pos_label} + {width_label}: {len(sub)}건, 승률 {win_rate:.1f}%, 평균 {avg_pnl:+.2f}%")

# 방향 + 위치 + 기울기
print("\n### 방향 + 위치 + 기울기")
slope_med = feat_df['avg_slope'].median()
for d in [1, -1]:
    dir_label = 'UP' if d == 1 else 'DOWN'
    for pos_low, pos_high, pos_label in [(0, 30, '하단'), (30, 50, '중하단'), (50, 70, '중상단'), (70, 100, '상단')]:
        for slope_cond, slope_label in [('up', '상승기울기'), ('down', '하락기울기')]:
            if slope_cond == 'up':
                sub = feat_df[(feat_df['last_1_dir'] == d) & 
                              (feat_df['last_bb_pos'] >= pos_low) & 
                              (feat_df['last_bb_pos'] < pos_high) &
                              (feat_df['avg_slope'] > slope_med)]
            else:
                sub = feat_df[(feat_df['last_1_dir'] == d) & 
                              (feat_df['last_bb_pos'] >= pos_low) & 
                              (feat_df['last_bb_pos'] < pos_high) &
                              (feat_df['avg_slope'] <= slope_med)]
            
            if len(sub) >= 20:
                win_rate = sub['win'].mean() * 100
                avg_pnl = sub['pnl'].mean()
                if win_rate >= 55 or avg_pnl >= 0.3:
                    best_patterns.append({
                        'pattern': f"{dir_label} + {pos_label} + {slope_label}",
                        'count': len(sub),
                        'win_rate': win_rate,
                        'avg_pnl': avg_pnl
                    })
                    print(f"  ✓ {dir_label} + {pos_label} + {slope_label}: {len(sub)}건, 승률 {win_rate:.1f}%, 평균 {avg_pnl:+.2f}%")

# 수축 길이 + 위치 이동 + 마지막 방향
print("\n### 수축길이 + 위치이동 + 마지막방향")
for len_low, len_high, len_label in [(6, 15, '중간수축(6-15h)'), (15, 50, '긴수축(15h+)')]:
    for drift_low, drift_high, drift_label in [(10, 100, '상승이동'), (-100, -10, '하락이동')]:
        for d in [1, -1]:
            dir_label = 'UP' if d == 1 else 'DOWN'
            sub = feat_df[(feat_df['length'] >= len_low) & 
                          (feat_df['length'] < len_high) &
                          (feat_df['pos_drift'] >= drift_low) & 
                          (feat_df['pos_drift'] < drift_high) &
                          (feat_df['last_1_dir'] == d)]
            
            if len(sub) >= 15:
                win_rate = sub['win'].mean() * 100
                avg_pnl = sub['pnl'].mean()
                if win_rate >= 55 or avg_pnl >= 0.3:
                    print(f"  ✓ {len_label} + {drift_label} + {dir_label}: {len(sub)}건, 승률 {win_rate:.1f}%, 평균 {avg_pnl:+.2f}%")

# 마지막 3봉 연속 + 위치
print("\n### 마지막 3봉 연속 + BB 위치")
for sum_val, sum_label in [(-3, '3봉연속DOWN'), (3, '3봉연속UP')]:
    for pos_low, pos_high, pos_label in [(0, 30, '하단'), (30, 50, '중하단'), (50, 70, '중상단'), (70, 100, '상단')]:
        sub = feat_df[(feat_df['last_3_sum'] == sum_val) & 
                      (feat_df['last_bb_pos'] >= pos_low) & 
                      (feat_df['last_bb_pos'] < pos_high)]
        
        if len(sub) >= 10:
            win_rate = sub['win'].mean() * 100
            avg_pnl = sub['pnl'].mean()
            print(f"  {sum_label} + {pos_label}: {len(sub)}건, 승률 {win_rate:.1f}%, 평균 {avg_pnl:+.2f}%")

# 돌파 방향 예측 - 어떤 조건에서 UP/DOWN 돌파가 예측 가능한가?
print("\n" + "="*80)
print("🔮 돌파 방향 예측 패턴")
print("="*80)

print("\n### 조건별 돌파 방향 비율")
# 마지막 봉 방향 → 돌파 방향
for d in [1, -1]:
    dir_label = 'UP' if d == 1 else 'DOWN'
    sub = feat_df[feat_df['last_1_dir'] == d]
    up_break = (sub['break_dir'] == 1).sum()
    down_break = (sub['break_dir'] == -1).sum()
    print(f"  마지막봉 {dir_label} → UP돌파 {up_break/len(sub)*100:.0f}% / DOWN돌파 {down_break/len(sub)*100:.0f}%")

# BB 위치 → 돌파 방향  
print("\n### BB 위치별 돌파 방향")
for pos_low, pos_high, pos_label in [(0, 20, '하단(0-20)'), (20, 40, '중하단'), (40, 60, '중간'), (60, 80, '중상단'), (80, 100, '상단(80-100)')]:
    sub = feat_df[(feat_df['last_bb_pos'] >= pos_low) & (feat_df['last_bb_pos'] < pos_high)]
    if len(sub) >= 20:
        up_break = (sub['break_dir'] == 1).sum()
        down_break = (sub['break_dir'] == -1).sum()
        
        # 각 방향 수익
        up_pnl = sub[sub['break_dir'] == 1]['pnl'].mean() if up_break > 0 else 0
        down_pnl = sub[sub['break_dir'] == -1]['pnl'].mean() if down_break > 0 else 0
        
        print(f"  {pos_label}: UP {up_break/len(sub)*100:.0f}% ({up_pnl:+.2f}%) / DOWN {down_break/len(sub)*100:.0f}% ({down_pnl:+.2f}%)")

print("\n" + "="*80)
print("📊 최종 유의미한 패턴 정리")
print("="*80)

# 수익 기준 상위 패턴 재탐색
print("\n승률 55% 이상 또는 평균수익 0.3% 이상인 패턴:")
if best_patterns:
    for p in sorted(best_patterns, key=lambda x: x['avg_pnl'], reverse=True):
        print(f"  • {p['pattern']}: {p['count']}건, 승률 {p['win_rate']:.1f}%, 수익 {p['avg_pnl']:+.2f}%")
