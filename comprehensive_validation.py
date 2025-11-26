"""
종합 검증 리포트 시스템
MACD H/L 나우캐스트 시스템의 모든 검증을 실행하고 종합 리포트를 생성합니다.
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from macd_hl_labeling import process_all_timeframes
from hierarchical_validation import HierarchicalValidator
from nowcast_simulation import NowcastSimulator


class ComprehensiveValidator:
    """종합 검증 시스템"""

    def __init__(self, data_path):
        """
        Args:
            data_path: BTC 15분봉 CSV 파일 경로
        """
        self.data_path = data_path
        self.df = None
        self.hl_results = None
        self.validation_results = {}
        self.report_dir = "validation_reports"

        os.makedirs(self.report_dir, exist_ok=True)

    def load_data(self):
        """데이터 로드 및 전처리"""
        print("=" * 60)
        print("1. 데이터 로드")
        print("=" * 60)

        self.df = pd.read_csv(self.data_path)
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])

        print(f"\n파일: {self.data_path}")
        print(f"데이터 shape: {self.df.shape}")
        print(f"기간: {self.df['datetime'].min()} ~ {self.df['datetime'].max()}")
        print(f"총 일수: {(self.df['datetime'].max() - self.df['datetime'].min()).days}")

        # 결측치 확인
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n결측치:")
            print(missing[missing > 0])
        else:
            print("\n결측치 없음")

    def run_hl_labeling(self):
        """H/L 라벨링 수행"""
        print("\n" + "=" * 60)
        print("2. MACD H/L 라벨링")
        print("=" * 60)

        self.hl_results = process_all_timeframes(self.df)

        # 결과 요약
        print("\n타임프레임별 H/L 요약:")
        summary = []
        for tf, result in self.hl_results.items():
            stats = result['stats']
            if stats:
                summary.append({
                    'Timeframe': tf,
                    'Total H/L': stats['total_count'],
                    'High': stats['high_count'],
                    'Low': stats['low_count'],
                    'Avg Interval (h)': stats.get('avg_interval_hours', 0)
                })

        summary_df = pd.DataFrame(summary)
        print("\n" + summary_df.to_string(index=False))

        # CSV 저장
        summary_df.to_csv(f"{self.report_dir}/hl_summary.csv", index=False)

    def run_hierarchical_validation(self):
        """계층적 전파 검증"""
        print("\n" + "=" * 60)
        print("3. 계층적 전파 검증")
        print("=" * 60)

        validator = HierarchicalValidator()
        validations = validator.validate_all_layers(self.hl_results)

        metrics_df = validator.calculate_metrics(validations)

        print("\n검증 메트릭:")
        print(metrics_df.to_string(index=False))

        # 저장
        metrics_df.to_csv(f"{self.report_dir}/hierarchical_metrics.csv", index=False)
        validator.plot_validation_results(
            metrics_df,
            f"{self.report_dir}/hierarchical_validation.png"
        )

        self.validation_results['hierarchical'] = {
            'validations': validations,
            'metrics': metrics_df
        }

    def run_nowcast_simulation(self, target_tf='1H'):
        """나우캐스트 시뮬레이션"""
        print("\n" + "=" * 60)
        print(f"4. 나우캐스트 시뮬레이션 (15T → {target_tf})")
        print("=" * 60)

        simulator = NowcastSimulator()

        # 시뮬레이션 실행
        results = simulator.simulate_realtime_detection(self.df, target_tf)

        if len(simulator.predictions) > 0:
            predictions_df = pd.DataFrame(simulator.predictions)
            matched_results = simulator.match_predictions_with_actual(
                predictions_df,
                results['hl_upper_final']
            )

            # 성능 메트릭
            metrics = simulator.calculate_performance_metrics(matched_results)

            print("\n나우캐스트 성능:")
            print(f"  정확도: {metrics['accuracy']*100:.2f}%")
            print(f"  정밀도: {metrics['precision']*100:.2f}%")
            print(f"  재현율: {metrics['recall']*100:.2f}%")
            print(f"  F1 Score: {metrics['f1_score']:.3f}")

            # 저장
            simulator.plot_confusion_matrix(
                metrics,
                f"{self.report_dir}/nowcast_confusion_matrix.png"
            )

            # 메트릭을 DataFrame으로 저장
            metrics_df = pd.DataFrame([{
                'Target_TF': target_tf,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score'],
                'True_Positives': metrics['true_positives'],
                'True_Negatives': metrics['true_negatives'],
                'False_Positives': metrics['false_positives'],
                'False_Negatives': metrics['false_negatives']
            }])

            metrics_df.to_csv(f"{self.report_dir}/nowcast_metrics.csv", index=False)

            self.validation_results['nowcast'] = {
                'metrics': metrics,
                'matched_results': matched_results
            }
        else:
            print("\n예측 없음")

    def analyze_major_inflection_points(self):
        """주요 변곡점 분석"""
        print("\n" + "=" * 60)
        print("5. 주요 변곡점 분석")
        print("=" * 60)

        # 주요 시점 정의
        inflection_points = [
            {'date': '2020-03-12', 'event': '코로나 폭락'},
            {'date': '2021-04-14', 'event': '2021 ATH (64k)'},
            {'date': '2021-11-10', 'event': '2021 ATH (69k)'},
            {'date': '2022-06-18', 'event': '루나 사태'},
            {'date': '2022-11-09', 'event': 'FTX 사태'},
            {'date': '2024-03-14', 'event': '2024 신고점 (73k)'},
        ]

        analysis_results = []

        for point in inflection_points:
            date_str = point['date']
            event = point['event']

            try:
                target_date = pd.to_datetime(date_str)

                # 해당 시점 주변 데이터
                week_before = target_date - pd.Timedelta(days=7)
                week_after = target_date + pd.Timedelta(days=7)

                period_df = self.df[
                    (self.df['datetime'] >= week_before) &
                    (self.df['datetime'] <= week_after)
                ]

                if len(period_df) > 0:
                    price_change = (
                        period_df['close'].iloc[-1] - period_df['close'].iloc[0]
                    ) / period_df['close'].iloc[0] * 100

                    volatility = period_df['close'].pct_change().std() * 100

                    analysis_results.append({
                        'Date': date_str,
                        'Event': event,
                        'Price_Change_Pct': price_change,
                        'Volatility': volatility,
                        'Data_Points': len(period_df)
                    })

                    print(f"\n{date_str} - {event}")
                    print(f"  가격 변화: {price_change:+.2f}%")
                    print(f"  변동성: {volatility:.2f}%")

            except Exception as e:
                print(f"\n{date_str} - {event}: 데이터 없음")

        if analysis_results:
            inflection_df = pd.DataFrame(analysis_results)
            inflection_df.to_csv(f"{self.report_dir}/inflection_points.csv", index=False)

    def generate_final_report(self):
        """최종 리포트 생성"""
        print("\n" + "=" * 60)
        print("6. 최종 리포트 생성")
        print("=" * 60)

        report_path = f"{self.report_dir}/comprehensive_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BTC MACD H/L 나우캐스트 시스템 종합 검증 리포트\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"데이터 파일: {self.data_path}\n\n")

            # 데이터 요약
            f.write("1. 데이터 요약\n")
            f.write("-" * 80 + "\n")
            f.write(f"총 데이터 포인트: {len(self.df):,}\n")
            f.write(f"기간: {self.df['datetime'].min()} ~ {self.df['datetime'].max()}\n")
            f.write(f"총 일수: {(self.df['datetime'].max() - self.df['datetime'].min()).days:,}\n\n")

            # H/L 라벨링 요약
            if os.path.exists(f"{self.report_dir}/hl_summary.csv"):
                f.write("\n2. H/L 라벨링 요약\n")
                f.write("-" * 80 + "\n")
                hl_summary = pd.read_csv(f"{self.report_dir}/hl_summary.csv")
                f.write(hl_summary.to_string(index=False) + "\n\n")

            # 계층적 검증
            if os.path.exists(f"{self.report_dir}/hierarchical_metrics.csv"):
                f.write("\n3. 계층적 전파 검증\n")
                f.write("-" * 80 + "\n")
                hier_metrics = pd.read_csv(f"{self.report_dir}/hierarchical_metrics.csv")
                f.write(hier_metrics.to_string(index=False) + "\n\n")

            # 나우캐스트 성능
            if os.path.exists(f"{self.report_dir}/nowcast_metrics.csv"):
                f.write("\n4. 나우캐스트 성능\n")
                f.write("-" * 80 + "\n")
                nowcast_metrics = pd.read_csv(f"{self.report_dir}/nowcast_metrics.csv")
                f.write(nowcast_metrics.to_string(index=False) + "\n\n")

            # 결론
            f.write("\n5. 결론 및 권고사항\n")
            f.write("-" * 80 + "\n")
            f.write("- 시스템 유효성: [데이터 기반 평가]\n")
            f.write("- 주요 발견사항: [분석 결과 요약]\n")
            f.write("- 개선 방향: [권고사항]\n")

        print(f"\n최종 리포트 저장: {report_path}")

        # 모든 결과 파일 목록
        print("\n생성된 파일:")
        for file in sorted(os.listdir(self.report_dir)):
            print(f"  - {self.report_dir}/{file}")

    def run_complete_validation(self):
        """전체 검증 실행"""
        print("\n" + "=" * 80)
        print("BTC MACD H/L 나우캐스트 시스템 종합 검증")
        print("=" * 80)

        try:
            self.load_data()
            self.run_hl_labeling()
            self.run_hierarchical_validation()
            self.run_nowcast_simulation(target_tf='1H')
            self.analyze_major_inflection_points()
            self.generate_final_report()

            print("\n" + "=" * 80)
            print("검증 완료!")
            print("=" * 80)
            print(f"\n결과 디렉토리: {self.report_dir}/")

        except Exception as e:
            print(f"\n에러 발생: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # 데이터 파일 찾기
    data_dir = "data"
    pattern = os.path.join(data_dir, "BTC_USDT_USDT_15m_*.csv")
    files = glob.glob(pattern)

    if not files:
        print("BTC 데이터 파일이 없습니다.")
        print("먼저 bybit_collector_ccxt.py를 실행하여 데이터를 수집하세요.")
    else:
        latest_file = max(files, key=os.path.getmtime)

        # 종합 검증 실행
        validator = ComprehensiveValidator(latest_file)
        validator.run_complete_validation()
