from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

class BounceStatisticsAnalyzer:
    """
    라벨의 반등 패턴을 분석하고, 등급별 통계를 계산하여 리포트를 생성합니다.
    - 다중 타임프레임 합의에 따라 라벨 등급(A, B)을 분류합니다.
    - 수익률 기준 Top 100 라벨을 식별합니다.
    - 등급별 반등률, 성공률, 지속 기간 등 상세 통계를 계산합니다.
    """
    def __init__(self, base_path: Path):
        """
        분석기 초기화

        Args:
            base_path (Path): 프로젝트 루트 경로.
        """
        self.base_path = base_path
        self.results_path = self.base_path / "results"
        self.data_path = self.base_path / "data"
        self.performance_df = None
        self.graded_df = None
        self.top_100_indices = None
        self.stats = {}
        print("BounceStatisticsAnalyzer가 초기화되었습니다.")

    def _load_data(self):
        """분석에 필요한 데이터 파일을 로드하고 병합합니다."""
        print("데이터 로딩 중...")
        perf_file = self.results_path / "label_performance" / "performance_analysis_fast.csv"
        merged_labels_file = self.data_path / "labels" / "btc_usdt_kst_merged_labels.parquet"
        
        if not perf_file.exists():
            raise FileNotFoundError(f"성능 분석 파일({perf_file})을 찾을 수 없습니다. 'analyze_label_performance_fast.py'를 먼저 실행하세요.")
        if not merged_labels_file.exists():
            raise FileNotFoundError(f"병합된 라벨 파일({merged_labels_file})을 찾을 수 없습니다. 'merge_labels.py'를 먼저 실행하세요.")
            
        # CSV 파일을 읽을 때 첫 번째 컬럼(인덱스)을 timestamp로 지정
        self.performance_df = pd.read_csv(perf_file, index_col=0, parse_dates=True)
        merged_labels = pd.read_parquet(merged_labels_file)
        
        # 타임프레임 합의(동일 시간 라벨 개수) 계산
        agreement_counts = merged_labels.groupby(merged_labels.index).size().rename('agreement_count')
        
        # 인덱스를 기준으로 합의 개수를 성능 데이터프레임에 병합
        self.performance_df = self.performance_df.join(agreement_counts, how='left')
        self.performance_df['agreement_count'].fillna(1, inplace=True) # 병합 안된 경우 1로 처리
        
        # 'timestamp' 컬럼이 필요하다면 인덱스를 리셋하여 생성
        self.performance_df.reset_index(inplace=True)
        # 인덱스 이름이 'index' 또는 'Unnamed: 0'일 수 있으므로 'timestamp'로 명확히 변경
        self.performance_df.rename(columns={'index': 'timestamp', 'Unnamed: 0': 'timestamp'}, inplace=True)
        print("데이터 로딩 및 전처리 완료.")

    def _grade_labels(self):
        """라벨을 A급, B급으로 분류하고 수익률 Top 100을 식별합니다."""
        if self.performance_df is None:
            raise RuntimeError("데이터가 로드되지 않았습니다. _load_data()를 먼저 실행해야 합니다.")
        
        print("라벨 등급 분류 중...")
        df = self.performance_df.copy()
        
        # 합의 개수에 따라 A/B 등급 부여
        conditions = [
            (df['agreement_count'] >= 5),
            (df['agreement_count'].between(3, 4))
        ]
        choices = ['A_Grade', 'B_Grade']
        df['grade'] = np.select(conditions, choices, default='C_Grade')

        # 수익률 기준 Top 100 라벨 인덱스 식별 (매수/매도 별도)
        top_100_buy_indices = df[df['label_type'] == 1].nlargest(100, 'max_profit_pct').index
        top_100_sell_indices = df[df['label_type'] == -1].nlargest(100, 'max_profit_pct').index
        self.top_100_indices = top_100_buy_indices.union(top_100_sell_indices)

        self.graded_df = df
        print("라벨 등급 분류 완료.")

    def _calculate_stats(self, df: pd.DataFrame, label_type: int) -> pd.Series:
        """주어진 데이터프레임에 대해 통계를 계산합니다."""
        data = df[df['label_type'] == label_type]
        if data.empty:
            return pd.Series([0] * 9, index=[
                'count', 'avg_bounce_pct', 'max_bounce_pct', 'success_rate_5pct',
                'avg_duration_tomax_min', 'p25_bounce_pct', 'p50_bounce_pct', 'p75_bounce_pct', 'p90_bounce_pct'
            ], dtype=float)

        stats = {
            'count': len(data),
            'avg_bounce_pct': data['max_profit_pct'].mean() * 100,
            'max_bounce_pct': data['max_profit_pct'].max() * 100,
            'success_rate_5pct': (data['max_profit_pct'] > 0.05).mean() * 100,
            'avg_duration_tomax_min': data['time_to_max_profit'].mean(),
            'p25_bounce_pct': data['max_profit_pct'].quantile(0.25) * 100,
            'p50_bounce_pct': data['max_profit_pct'].quantile(0.50) * 100,
            'p75_bounce_pct': data['max_profit_pct'].quantile(0.75) * 100,
            'p90_bounce_pct': data['max_profit_pct'].quantile(0.90) * 100,
        }
        return pd.Series(stats)

    def analyze_bounce_patterns(self):
        """라벨별 전체 반등 통계를 분석합니다."""
        print("전체 반등 패턴 분석 시작...")
        self._load_data()
        if self.performance_df is None:
            raise RuntimeError("데이터가 로드되지 않았습니다. _load_data()가 None을 반환했습니다.")

        overall_buy_stats = self._calculate_stats(self.performance_df, 1)
        overall_sell_stats = self._calculate_stats(self.performance_df, -1)

        stats_df = pd.DataFrame({'Buy_Label (L)': overall_buy_stats, 'Sell_Label (H)': overall_sell_stats})
        self.stats['Overall'] = stats_df
        print("전체 분석 완료.")

    def get_graded_labels(self) -> pd.DataFrame:
        """
        등급이 부여된 라벨 데이터프레임을 반환합니다.
        데이터가 준비되지 않았다면 준비 과정을 트리거합니다.
        """
        if self.graded_df is None:
            print("'graded_df'가 존재하지 않아 등급 계산을 시작합니다.")
            self.calculate_grade_statistics()
        
        if self.graded_df is None:
            # calculate_grade_statistics 실행 후에도 None이면 빈 데이터프레임 반환 또는 예외 발생
            print("경고: 등급 계산 후에도 'graded_df'를 생성할 수 없습니다.")
            return pd.DataFrame()
            
        return self.graded_df

    def calculate_grade_statistics(self):
        """A급, B급, Top100 라벨별 통계를 계산합니다."""
        if self.performance_df is None:
            self.analyze_bounce_patterns() # 전체 분석이 선행되어야 함
        
        print("등급별 통계 계산 시작...")
        self._grade_labels()

        if self.graded_df is None:
            print("경고: 라벨 등급 데이터(graded_df)가 생성되지 않았습니다. 등급별 통계 계산을 건너뜁니다.")
            return
        
        # A급/B급 통계
        for grade in ['A_Grade', 'B_Grade']:
            grade_df = self.graded_df[self.graded_df['grade'] == grade]
            buy_stats = self._calculate_stats(grade_df, 1)
            sell_stats = self._calculate_stats(grade_df, -1)
            stats_df = pd.DataFrame({'Buy_Label (L)': buy_stats, 'Sell_Label (H)': sell_stats})
            self.stats[grade] = stats_df

        # Top 100 수익률 통계
        if self.top_100_indices is not None and not self.top_100_indices.empty:
            top100_df = self.graded_df.loc[self.top_100_indices]
            if isinstance(top100_df, pd.Series):
                # 단일 row만 선택된 경우 DataFrame으로 변환
                top100_df = top100_df.to_frame().T
            buy_stats = self._calculate_stats(top100_df, 1)
            sell_stats = self._calculate_stats(top100_df, -1)
            stats_df = pd.DataFrame({'Buy_Label (L)': buy_stats, 'Sell_Label (H)': sell_stats})
            self.stats['Top100_Profit'] = stats_df
        else:
            print("Top 100 수익률 라벨이 없습니다. 'Top100_Profit' 통계를 건너뜁니다.")
        print("등급별 통계 계산 완료.")

    def generate_bounce_report(self):
        """계산된 통계를 바탕으로 HTML 리포트를 생성합니다."""
        if not self.stats:
            raise RuntimeError("통계가 계산되지 않았습니다. 분석 메소드를 먼저 실행하세요.")
            
        print("HTML 리포트 생성 중...")
        report_path = self.results_path / "label_analysis" / "bounce_statistics_report.html"
        report_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            _ = f.write("<html><head><title>라벨 반등 통계 리포트</title>")
            _ = f.write("""
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 20px; background-color: #f4f7f6; color: #333; }
                h1, h2 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                table { border-collapse: collapse; width: 80%; margin-top: 20px; margin-bottom: 40px; box-shadow: 0 2px 3px rgba(0,0,0,0.1); }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: right; }
                th { background-color: #3498db; color: white; text-align: center; }
                tr:nth-child(even) { background-color: #ecf0f1; }
                tr:hover { background-color: #bdc3c7; }
                .container { max-width: 1200px; margin: auto; }
            </style>
            """)
            _ = f.write("</head><body><div class='container'>")
            _ = f.write("<h1>라벨 반등 통계 리포트</h1>")
            
            for name, df in self.stats.items():
                title = name.replace('_', ' ').title()
                _ = f.write(f"<h2>{title} Statistics</h2>")
                _ = f.write(df.to_html(float_format=lambda x: f'{x:,.2f}'))
                
            _ = f.write("</div></body></html>")
            
        print(f"리포트가 성공적으로 생성되었습니다: {report_path}")

# 메인 실행 블록
if __name__ == '__main__':
    try:
        # 프로젝트 루트 경로 설정 (현재 파일 위치 기준)
        project_root = Path(__file__).resolve().parents[2]
        analyzer = BounceStatisticsAnalyzer(base_path=project_root)
        
        # 분석 실행
        analyzer.analyze_bounce_patterns()
        analyzer.calculate_grade_statistics()
        
        # 리포트 생성
        analyzer.generate_bounce_report()
        
    except (FileNotFoundError, RuntimeError) as e:
        print(f"오류가 발생했습니다: {e}")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")
