import pandas as pd
from pathlib import Path

def analyze_top_profit_labels():
    """실용적인 수익률 분석"""
    
    # 🔥 올바른 경로
    input_file = Path("results/label_performance/performance_analysis_parallel.csv")
    output_dir = Path("results/profit_analysis")
    output_dir.mkdir(exist_ok=True)
    
    if not input_file.exists():
        print(f"❌ 파일 없음: {input_file}")
        return
    
    df = pd.read_csv(input_file)
    print(f"📊 총 {len(df):,}개 라벨 분석")
    
    # 🔥 의미 있는 구간 분석
    def categorize_profit(profit_pct):
        if profit_pct < 50:
            return "보통 (0-50%)"
        elif profit_pct < 100:
            return "좋음 (50-100%)"
        elif profit_pct < 300:
            return "훌륭 (100-300%)"
        elif profit_pct < 500:
            return "대박 (300-500%)"
        elif profit_pct < 1000:
            return "초대박 (500-1000%)"
        else:
            return "신급 (1000%+)"
    
    df['수익률_등급'] = df['최대수익률(%)'].apply(categorize_profit)
    
    # 타임프레임별 등급 분포
    print("\n🎯 타임프레임별 수익률 등급 분포:")
    result_summary = []
    
    for tf in sorted(df['타임프레임'].unique()):
        tf_data = df[df['타임프레임'] == tf]
        
        # 등급별 개수
        grade_counts = tf_data['수익률_등급'].value_counts()
        total = len(tf_data)
        
        # Top 10% 라벨들
        top_10_pct = tf_data.nlargest(int(total * 0.1), '최대수익률(%)')
        
        summary = {
            '타임프레임': tf,
            '총라벨수': total,
            '평균수익률': tf_data['최대수익률(%)'].mean(),
            '최대수익률': tf_data['최대수익률(%)'].max(),
            '신급_1000%+': grade_counts.get('신급 (1000%+)', 0),
            '초대박_500-1000%': grade_counts.get('초대박 (500-1000%)', 0),
            '대박_300-500%': grade_counts.get('대박 (300-500%)', 0),
            'Top10%_평균수익률': top_10_pct['최대수익률(%)'].mean()
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
    summary_df.to_csv(output_dir / 'profit_grade_summary.csv', index=False)
    
    # 🔥 Top 1000 라벨 추출
    top_1000 = df.nlargest(1000, '최대수익률(%)')
    top_1000.to_csv(output_dir / 'top_1000_labels.csv', index=False)
    
    print(f"\n🏆 Top 1000 라벨 (AI 학습용):")
    print(f"   평균 수익률: {top_1000['최대수익률(%)'].mean():.1f}%")
    print(f"   최대 수익률: {top_1000['최대수익률(%)'].max():.1f}%")
    print(f"   타임프레임 분포:")
    for tf, count in top_1000['타임프레임'].value_counts().head().items():
        print(f"     {tf}: {count}개")
    
    print(f"\n✅ 결과 저장:")
    print(f"   - 등급별 요약: {output_dir}/profit_grade_summary.csv")
    print(f"   - Top 1000: {output_dir}/top_1000_labels.csv")
    
    return top_1000

if __name__ == "__main__":
    analyze_top_profit_labels()