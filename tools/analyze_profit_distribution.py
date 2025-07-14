import pandas as pd
from pathlib import Path
from enum import Enum
from typing import List, Dict, Any, Optional

class ProfitCategory(Enum):
    """수익률 등급을 정의하는 열거형 클래스"""
    GOD = "신급 (1000%+)"
    SUPER_HIT = "초대박 (500-1000%)"
    BIG_HIT = "대박 (300-500%)"
    EXCELLENT = "훌륭 (100-300%)"
    GOOD = "좋음 (50-100%)"
    NORMAL = "보통 (0-50%)"

def categorize_profit(profit_pct: float) -> str:
    """수익률(%)을 기반으로 카테고리 등급을 반환합니다."""
    if profit_pct < 50:
        return ProfitCategory.NORMAL.value
    elif profit_pct < 100:
        return ProfitCategory.GOOD.value
    elif profit_pct < 300:
        return ProfitCategory.EXCELLENT.value
    elif profit_pct < 500:
        return ProfitCategory.BIG_HIT.value
    elif profit_pct < 1000:
        return ProfitCategory.SUPER_HIT.value
    else:
        return ProfitCategory.GOD.value

def analyze_top_profit_labels() -> Optional[pd.DataFrame]:
    """
    라벨 성능 데이터를 분석하여 수익률 등급을 나누고, 상위 라벨을 추출합니다.
    
    Returns:
        Optional[pd.DataFrame]: 분석에 실패하면 None, 성공하면 Top 1000 라벨 데이터프레임.
    """
    # 🔥 올바른 경로
    input_file = Path("results/label_performance/performance_analysis_parallel.csv")
    output_dir = Path("results/profit_analysis")
    output_dir.mkdir(exist_ok=True)
    
    if not input_file.exists():
        print(f"❌ 파일 없음: {input_file}")
        return None
    
    df = pd.read_csv(input_file)
    print(f"📊 총 {len(df):,}개 라벨 분석")
    
    df['수익률_등급'] = df['최대수익률(%)'].apply(categorize_profit)
    
    # 타임프레임별 등급 분포
    print("\n🎯 타임프레임별 수익률 등급 분포:")
    result_summary: List[Dict[str, Any]] = []
    
    # np.ndarray에서 unique()를 호출하면 Series가 아닌 numpy 배열을 반환할 수 있으므로 list로 변환
    unique_timeframes: list[str] = sorted(list(df['타임프레임'].unique()))

    for tf in unique_timeframes:
        tf_data = df[df['타임프레임'] == tf]
        
        # 등급별 개수
        grade_counts = tf_data['수익률_등급'].value_counts()
        total = len(tf_data)
        
        # Top 10% 라벨들
        top_10_pct_count = int(total * 0.1)
        if top_10_pct_count == 0:
            continue # 분석할 데이터가 없는 경우 건너뛰기

        top_10_pct = tf_data.nlargest(top_10_pct_count, '최대수익률(%)')
        
        summary: Dict[str, Any] = {
            '타임프레임': tf,
            '총라벨수': total,
            '평균수익률': tf_data['최대수익률(%)'].mean(),
            '최대수익률': tf_data['최대수익률(%)'].max(),
            '신급_1000%+': grade_counts.get(ProfitCategory.GOD.value, 0),
            '초대박_500-1000%': grade_counts.get(ProfitCategory.SUPER_HIT.value, 0),
            '대박_300-500%': grade_counts.get(ProfitCategory.BIG_HIT.value, 0),
            'Top10%_평균수익률': top_10_pct['최대수익률(%)'].mean() if not top_10_pct.empty else 0
        }
        result_summary.append(summary)
        
        print(f"\n📈 {tf}:")
        print(f"   평균: {summary['평균수익률']:.1f}%")
        print(f"   신급: {summary['신급_1000%+']}개")
        print(f"   초대박: {summary['초대박_500-1000%']}개")
        print(f"   대박: {summary['대박_300-500%']}개")
        print(f"   Top10% 평균: {summary['Top10%_평균수익률']:.1f}%")
    
    # 결과 저장
    summary_df = pd.DataFrame(result_summary)
    summary_df.to_csv(output_dir / 'profit_grade_summary.csv', index=False, encoding='utf-8-sig')
    
    # 🔥 Top 1000 라벨 추출
    top_1000 = df.nlargest(1000, '최대수익률(%)')
    top_1000.to_csv(output_dir / 'top_1000_labels.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n🏆 Top 1000 라벨 (AI 학습용):")
    print(f"   평균 수익률: {top_1000['최대수익률(%)'].mean():.1f}%")
    print(f"   최대 수익률: {top_1000['최대수익률(%)'].max():.1f}%")
    print(f"   타임프레임 분포:")
    # value_counts()는 Series를 반환하며, items()로 루프를 돌 수 있습니다.
    for tf, count in top_1000['타임프레임'].value_counts().head().items():
        print(f"     {tf}: {count}개")
    
    print(f"\n✅ 결과 저장:")
    print(f"   - 등급별 요약: {output_dir}/profit_grade_summary.csv")
    print(f"   - Top 1000: {output_dir}/top_1000_labels.csv")
    
    return top_1000

if __name__ == "__main__":
    analyze_top_profit_labels()