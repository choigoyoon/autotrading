import pandas as pd
from pathlib import Path

def show_top_100_trades():
    """
    분석 결과를 바탕으로 '본절 회귀'와 '미회귀' 거래 Top 100을 각각 출력합니다.
    """
    input_file = Path("results") / "label_performance" / "performance_analysis_fast.csv"
    output_dir = Path("results") / "top_trades"
    output_dir.mkdir(exist_ok=True)

    if not input_file.exists():
        print(f"오류: 분석 파일 '{input_file}'을 찾을 수 없습니다.")
        return

    print(f"분석 파일 로드: {input_file}")
    df = pd.read_csv(input_file, parse_dates=['진입시점', '최고(저)점시점', '청산시점'])
    print(f"   총 {len(df):,}개의 거래 내역을 분석합니다.")

    # 소수점 2자리까지만 표시하도록 설정
    pd.set_option('display.float_format', '{:.2f}'.format)

    # --- 1. 본절 회귀 (Breakeven Exit) Top 100 ---
    breakeven_df = df[df['최종상태'] == 'Breakeven Exit'].copy()
    
    # 연도별로 Top 15개씩 추출
    breakeven_df['year'] = breakeven_df['진입시점'].dt.year
    breakeven_top_by_year = breakeven_df.groupby('year').apply(
        lambda x: x.nlargest(15, '최대수익률(%)')
    ).reset_index(drop=True)
    
    # 전체에서 Top 100 선정
    breakeven_top_100 = breakeven_top_by_year.nlargest(100, '최대수익률(%)')

    # 콘솔에 출력
    print("\n" + "="*80)
    print("[본절 회귀 (Breakeven Exit)] 최대수익률 Top 100")
    print("   (L/H 지점 진입 후, 최고/최저점을 찍고 다시 본절로 돌아온 거래들)")
    print("="*80)
    print(breakeven_top_100.to_string())

    # --- 2. 미회귀 (Holding) Top 100 ---
    holding_df = df[df['최종상태'] == 'Holding'].copy()

    # 연도별로 Top 15개씩 추출
    holding_df['year'] = holding_df['진입시점'].dt.year
    holding_top_by_year = holding_df.groupby('year').apply(
        lambda x: x.nlargest(15, '최대수익률(%)')
    ).reset_index(drop=True)

    # 전체에서 Top 100 선정
    holding_top_100 = holding_top_by_year.nlargest(100, '최대수익률(%)')

    # 콘솔에 출력
    print("\n" + "="*80)
    print("[미회귀 (Holding)] 최대수익률 Top 100")
    print("   (L/H 지점 진입 후, 아직 본절로 돌아오지 않은 거래들)")
    print("="*80)
    print(holding_top_100.to_string())

    # 파일로 저장
    breakeven_file = output_dir / "top_100_breakeven_trades.csv"
    holding_file = output_dir / "top_100_holding_trades.csv"
    
    breakeven_top_100.to_csv(breakeven_file, index=False, encoding='utf-8-sig')
    holding_top_100.to_csv(holding_file, index=False, encoding='utf-8-sig')

    print("\n" + "="*80)
    print("Top 100 거래 목록 파일 저장 완료")
    print(f"  - 본절 회귀: {breakeven_file}")
    print(f"  - 미회귀: {holding_file}")
    print("="*80)

if __name__ == "__main__":
    show_top_100_trades() 