import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# 분석할 타임프레임 목록을 상수로 정의
SUPPORTED_TIMEFRAMES: List[str] = [
    '1min', '3min', '5min', '10min', '15min', '30min',
    '1h', '2h', '4h', '6h', '8h', '12h',
    '1D', '2D', '3D', '1W'
]

def analyze_label_quality() -> Optional[pd.DataFrame]:
    """
    모든 타임프레임에 대한 MACD 라벨 데이터의 품질을 분석하고 요약합니다.

    Returns:
        Optional[pd.DataFrame]: 분석 성공 시 요약 데이터프레임, 아니면 None.
    """
    labels_dir = Path('data/labels_macd/btc_usdt_kst_macd/btc_usdt_kst')
    results_dir = Path('results/label_analysis')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary_data: List[Dict[str, Any]] = []
    total_all, label_1_all, label_minus_1_all, label_0_all = 0, 0, 0, 0
    
    for tf in SUPPORTED_TIMEFRAMES:
        label_file = labels_dir / f"macd_labels_{tf}.parquet"
        if not label_file.exists():
            continue
            
        df = pd.read_parquet(label_file)
        
        total = len(df)
        if total == 0:
            continue

        label_1_count = (df['label'] == 1).sum()
        label_minus_1_count = (df['label'] == -1).sum()  
        label_0_count = total - label_1_count - label_minus_1_count

        total_all += total
        label_1_all += label_1_count
        label_minus_1_all += label_minus_1_count
        label_0_all += label_0_count

        dist_df = pd.DataFrame({
            'label': [1, -1, 0],
            'count': [label_1_count, label_minus_1_count, label_0_count],
            'pct': [
                (label_1_count / total) * 100, 
                (label_minus_1_count / total) * 100, 
                (label_0_count / total) * 100
            ]
        })
        dist_df.to_csv(results_dir / f'labels_dist_{tf}.csv', index=False, encoding='utf-8-sig')
        
        summary_data.append({
            'timeframe': tf,
            'total_candles': total,
            'label_1_count': label_1_count,
            'label_-1_count': label_minus_1_count,
            'label_0_count': label_0_count,
            'label_1_pct': (label_1_count / total) * 100,
            'label_-1_pct': (label_minus_1_count / total) * 100,
            'label_0_pct': (label_0_count / total) * 100,
        })
    
    if not summary_data:
        print("분석할 라벨 데이터가 없습니다.")
        return None

    summary_df = pd.DataFrame(summary_data)

    if total_all > 0:
        total_row = pd.DataFrame([{
            'timeframe': 'TOTAL',
            'total_candles': total_all,
            'label_1_count': label_1_all,
            'label_-1_count': label_minus_1_all,
            'label_0_count': label_0_all,
            'label_1_pct': (label_1_all / total_all) * 100,
            'label_-1_pct': (label_minus_1_all / total_all) * 100,
            'label_0_pct': (label_0_all / total_all) * 100,
        }])
        summary_df = pd.concat([summary_df, total_row], ignore_index=True)

    summary_df.to_csv(results_dir / 'macd_labels_summary.csv', index=False, encoding='utf-8-sig')
    
    print("📊 MACD 라벨링 결과 요약 (타임프레임별 + 전체)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    return summary_df

# ================= 추가 분석 함수 =================

def analyze_label_reversals(df: pd.DataFrame) -> int:
    """변곡점(라벨 변화) 횟수를 분석합니다."""
    if df.empty:
        return 0
    label_shifts = df['label'].diff() != 0
    return int(label_shifts.sum())

def analyze_yearly_distribution(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """연도별 라벨 분포를 분석합니다."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return None
    
    df_year = df.copy()
    # pyright가 DatetimeIndex의 'year' 속성을 추론하지 못하는 경우가 있어, 이를 무시하도록 처리합니다.
    df_year['year'] = df.index.year # type: ignore
    yearly_stats = df_year.groupby('year')['label'].value_counts().unstack(fill_value=0)
    return yearly_stats

def analyze_label_sequences(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """라벨 연속성(동일 라벨 구간 길이)을 분석합니다."""
    if df.empty:
        return []
    
    runs: List[Tuple[int, int]] = []
    labels = df['label'].to_numpy()
    
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

def run_full_analysis() -> None:
    """전체 라벨 품질 및 상세 분석을 실행하고 결과를 저장합니다."""
    summary_df = analyze_label_quality()
    if summary_df is None:
        return

    labels_dir = Path('data/labels_macd/btc_usdt_kst_macd/btc_usdt_kst')
    results_dir = Path('results/label_analysis')

    for tf in SUPPORTED_TIMEFRAMES:
        label_file = labels_dir / f"macd_labels_{tf}.parquet"
        if not label_file.exists():
            continue
        df = pd.read_parquet(label_file)
        if df.empty:
            continue

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
        if runs:
            runs_df = pd.DataFrame(runs, columns=['label', 'length'])
            runs_df.to_csv(results_dir / f'label_runs_{tf}.csv', index=False, encoding='utf-8-sig')
            mean_length = runs_df['length'].mean()
            print(f"[{tf}] 라벨 연속 구간(동일값 지속) 통계: 전체 {len(runs)}구간, 평균 길이 {mean_length:.2f}")

# ================= 메인 함수 =================

if __name__ == "__main__":
    run_full_analysis()