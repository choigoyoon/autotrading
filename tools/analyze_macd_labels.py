import pandas as pd
from pathlib import Path

def analyze_label_quality():
    """라벨링 품질 분석"""
    
    labels_dir = Path('data/labels_macd/btc_usdt_kst_macd/btc_usdt_kst')
    results_dir = Path('results/label_analysis')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timeframes = [
        '1min', '3min', '5min', '10min', '15min', '30min',
        '1h', '2h', '4h', '6h', '8h', '12h',
        '1D', '2D', '3D', '1W'
    ]
    
    summary_data = []
    total_all = 0
    label_1_all = 0
    label_minus_1_all = 0
    label_0_all = 0
    
    for tf in timeframes:
        label_file = labels_dir / f"macd_labels_{tf}.parquet"
        if not label_file.exists():
            continue
            
        df = pd.read_parquet(label_file)
        
        # 기본 통계
        total = len(df)
        label_1_count = (df['label'] == 1).sum()
        label_minus_1_count = (df['label'] == -1).sum()  
        label_0_count = (df['label'] == 0).sum()

        # 전체 합계 누적
        total_all += total
        label_1_all += label_1_count
        label_minus_1_all += label_minus_1_count
        label_0_all += label_0_count

        # 타임프레임별 분포 CSV 저장
        dist_df = pd.DataFrame({
            'label': [1, -1, 0],
            'count': [label_1_count, label_minus_1_count, label_0_count],
            'pct': [label_1_count/total*100, label_minus_1_count/total*100, label_0_count/total*100]
        })
        dist_df.to_csv(results_dir / f'labels_dist_{tf}.csv', index=False, encoding='utf-8-sig')
        
        summary_data.append({
            'timeframe': tf,
            'total_candles': total,
            'label_1_count': label_1_count,
            'label_-1_count': label_minus_1_count,
            'label_0_count': label_0_count,
            'label_1_pct': label_1_count / total * 100,
            'label_-1_pct': label_minus_1_count / total * 100,
            'label_0_pct': label_0_count / total * 100,
        })
    
    # 결과 DataFrame 생성
    summary_df = pd.DataFrame(summary_data)

    # 전체 요약(토탈) 행 추가
    if total_all > 0:
        total_row = pd.DataFrame([{
            'timeframe': 'TOTAL',
            'total_candles': total_all,
            'label_1_count': label_1_all,
            'label_-1_count': label_minus_1_all,
            'label_0_count': label_0_all,
            'label_1_pct': label_1_all/total_all*100,
            'label_-1_pct': label_minus_1_all/total_all*100,
            'label_0_pct': label_0_all/total_all*100,
        }])
        summary_df = pd.concat([summary_df, total_row], ignore_index=True)

    # 전체 요약 CSV 저장
    summary_df.to_csv(results_dir / 'macd_labels_summary.csv', index=False, encoding='utf-8-sig')
    
    # 요약 출력
    print("📊 MACD 라벨링 결과 요약 (타임프레임별 + 전체)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    return summary_df

# ================= 추가 분석 함수 =================

def analyze_label_reversals(df):
    """변곡점(라벨 변화) 분석"""
    # 첫 행은 변화로 치지 않음
    label_shifts = df['label'].diff() != 0
    reversal_count = label_shifts.sum()
    return reversal_count

def analyze_yearly_distribution(df):
    """연도별 라벨 분포 분석"""
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        return None  # 인덱스가 날짜형이 아니면 분석 불가
    df_year = df.copy()
    df_year['year'] = df_year.index.year
    yearly_stats = df_year.groupby('year')['label'].value_counts().unstack(fill_value=0)
    return yearly_stats

def analyze_label_sequences(df):
    """라벨 연속성(동일 라벨 구간 길이) 분석"""
    runs = []
    labels = df['label'].values
    if len(labels) == 0:
        return runs
    current_label = labels[0]
    current_length = 1
    for label in labels[1:]:
        if label == current_label:
            current_length += 1
        else:
            runs.append((current_label, current_length))
            current_label = label
            current_length = 1
    runs.append((current_label, current_length))
    return runs

# ================= 메인 함수 끝에 추가 분석 =================

if __name__ == "__main__":
    summary_df = analyze_label_quality()

    # 각 타임프레임별 추가 분석 결과 저장
    labels_dir = Path('data/labels_macd/btc_usdt_kst_macd/btc_usdt_kst')
    results_dir = Path('results/label_analysis')
    timeframes = [
        '1min', '3min', '5min', '10min', '15min', '30min',
        '1h', '2h', '4h', '6h', '8h', '12h',
        '1D', '2D', '3D', '1W'
    ]
    for tf in timeframes:
        label_file = labels_dir / f"macd_labels_{tf}.parquet"
        if not label_file.exists():
            continue
        df = pd.read_parquet(label_file)
        # 변곡점 분석
        reversals = analyze_label_reversals(df)
        print(f"[{tf}] 변곡점(라벨 변화) 개수: {reversals}")
        with open(results_dir / f'reversals_{tf}.txt', 'w', encoding='utf-8') as f:
            f.write(f"{tf} 변곡점(라벨 변화) 개수: {reversals}\n")
        # 연도별 분포
        yearly_stats = analyze_yearly_distribution(df)
        if yearly_stats is not None:
            print(f"[{tf}] 연도별 라벨 분포:")
            print(yearly_stats)
            yearly_stats.to_csv(results_dir / f'yearly_label_dist_{tf}.csv', encoding='utf-8-sig')
        # 라벨 연속성 분석
        runs = analyze_label_sequences(df)
        runs_df = pd.DataFrame(runs, columns=['label', 'length'])
        runs_df.to_csv(results_dir / f'label_runs_{tf}.csv', index=False, encoding='utf-8-sig')
        print(f"[{tf}] 라벨 연속 구간(동일값 지속) 통계: 전체 {len(runs)}구간, 평균 길이 {runs_df['length'].mean():.2f}")