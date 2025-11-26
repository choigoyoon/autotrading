"""
계층적 H/L 전파 검증 시스템
하위 타임프레임의 H/L이 상위 타임프레임의 H/L로 승격되는지 검증합니다.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns


class HierarchicalValidator:
    """계층적 H/L 전파 검증"""

    def __init__(self):
        self.timeframe_hierarchy = ['15T', '1H', '4H', '1D', '3D', '1W']
        self.timeframe_hours = {
            '15T': 0.25,
            '1H': 1,
            '4H': 4,
            '1D': 24,
            '3D': 72,
            '1W': 168
        }

    def find_matching_hl(self, upper_hl, lower_hl_df, time_window_hours=None):
        """
        상위 TF H/L에 대응하는 하위 TF H/L 찾기

        Args:
            upper_hl: 상위 TF H/L 행 (Series)
            lower_hl_df: 하위 TF H/L DataFrame
            time_window_hours: 검색 시간 윈도우 (None이면 자동)

        Returns:
            매칭된 하위 H/L 리스트
        """
        upper_time = upper_hl['timestamp']
        upper_type = upper_hl['type']

        if time_window_hours is None:
            # 상위 TF 크기만큼 검색
            time_window_hours = self.timeframe_hours.get('1W', 168)

        # 검색 윈도우
        start_time = upper_time - timedelta(hours=time_window_hours)
        end_time = upper_time + timedelta(hours=time_window_hours)

        # 같은 타입(H or L)이면서 시간 범위 내
        matches = lower_hl_df[
            (lower_hl_df['type'] == upper_type) &
            (lower_hl_df['timestamp'] >= start_time) &
            (lower_hl_df['timestamp'] <= end_time)
        ].copy()

        if len(matches) > 0:
            # 시간 차이 계산
            matches['time_diff_hours'] = (matches['timestamp'] - upper_time).dt.total_seconds() / 3600
            matches['price_diff_pct'] = abs(matches['price'] - upper_hl['price']) / upper_hl['price'] * 100

        return matches

    def validate_propagation(self, upper_tf, lower_tf, upper_hl_df, lower_hl_df,
                            price_tolerance_pct=2.0, time_window_multiplier=1.5):
        """
        한 단계 전파 검증 (예: 1W → 3D)

        Args:
            upper_tf: 상위 타임프레임
            lower_tf: 하위 타임프레임
            upper_hl_df: 상위 H/L DataFrame
            lower_hl_df: 하위 H/L DataFrame
            price_tolerance_pct: 가격 허용 오차 (%)
            time_window_multiplier: 시간 윈도우 배수

        Returns:
            검증 결과 DataFrame
        """
        results = []

        upper_tf_hours = self.timeframe_hours[upper_tf]
        time_window = upper_tf_hours * time_window_multiplier

        for idx, upper_hl in upper_hl_df.iterrows():
            matches = self.find_matching_hl(upper_hl, lower_hl_df, time_window)

            # 가격 허용 오차 내의 매치만
            if len(matches) > 0:
                valid_matches = matches[matches['price_diff_pct'] <= price_tolerance_pct]
            else:
                valid_matches = pd.DataFrame()

            if len(valid_matches) > 0:
                # 가장 가까운 매치 선택
                best_match = valid_matches.loc[valid_matches['time_diff_hours'].abs().idxmin()]

                result = {
                    'upper_tf': upper_tf,
                    'lower_tf': lower_tf,
                    'upper_timestamp': upper_hl['timestamp'],
                    'lower_timestamp': best_match['timestamp'],
                    'type': upper_hl['type'],
                    'upper_price': upper_hl['price'],
                    'lower_price': best_match['price'],
                    'price_diff_pct': best_match['price_diff_pct'],
                    'time_diff_hours': best_match['time_diff_hours'],
                    'lead_time_hours': -best_match['time_diff_hours'],  # 음수면 하위가 먼저
                    'detected': True,
                    'num_candidates': len(matches)
                }
            else:
                result = {
                    'upper_tf': upper_tf,
                    'lower_tf': lower_tf,
                    'upper_timestamp': upper_hl['timestamp'],
                    'lower_timestamp': None,
                    'type': upper_hl['type'],
                    'upper_price': upper_hl['price'],
                    'lower_price': None,
                    'price_diff_pct': None,
                    'time_diff_hours': None,
                    'lead_time_hours': None,
                    'detected': False,
                    'num_candidates': len(matches)
                }

            results.append(result)

        return pd.DataFrame(results)

    def validate_all_layers(self, hl_results):
        """
        모든 계층 검증

        Args:
            hl_results: process_all_timeframes() 결과

        Returns:
            {(upper_tf, lower_tf): validation_df, ...}
        """
        validations = {}

        for i in range(len(self.timeframe_hierarchy) - 1):
            upper_tf = self.timeframe_hierarchy[i + 1]
            lower_tf = self.timeframe_hierarchy[i]

            upper_hl_df = hl_results[upper_tf]['hl_df']
            lower_hl_df = hl_results[lower_tf]['hl_df']

            if len(upper_hl_df) == 0 or len(lower_hl_df) == 0:
                continue

            print(f"\n검증: {upper_tf} ← {lower_tf}")

            validation_df = self.validate_propagation(
                upper_tf, lower_tf,
                upper_hl_df, lower_hl_df
            )

            validations[(upper_tf, lower_tf)] = validation_df

            # 요약 통계
            detected_rate = validation_df['detected'].sum() / len(validation_df) * 100
            print(f"  감지율: {detected_rate:.1f}% ({validation_df['detected'].sum()}/{len(validation_df)})")

            if validation_df['detected'].sum() > 0:
                lead_times = validation_df[validation_df['detected']]['lead_time_hours']
                print(f"  평균 선행 시간: {lead_times.mean():.1f}시간")
                print(f"  중앙값 선행 시간: {lead_times.median():.1f}시간")

        return validations

    def calculate_metrics(self, validations):
        """
        검증 메트릭 계산

        Args:
            validations: validate_all_layers() 결과

        Returns:
            메트릭 DataFrame
        """
        metrics = []

        for (upper_tf, lower_tf), val_df in validations.items():
            total = len(val_df)
            detected = val_df['detected'].sum()
            missed = total - detected

            if detected > 0:
                lead_times = val_df[val_df['detected']]['lead_time_hours']
                avg_lead = lead_times.mean()
                median_lead = lead_times.median()
                min_lead = lead_times.min()
                max_lead = lead_times.max()

                price_diffs = val_df[val_df['detected']]['price_diff_pct']
                avg_price_diff = price_diffs.mean()
            else:
                avg_lead = median_lead = min_lead = max_lead = None
                avg_price_diff = None

            metrics.append({
                'upper_tf': upper_tf,
                'lower_tf': lower_tf,
                'total_upper_hl': total,
                'detected': detected,
                'missed': missed,
                'detection_rate_pct': detected / total * 100 if total > 0 else 0,
                'avg_lead_time_hours': avg_lead,
                'median_lead_time_hours': median_lead,
                'min_lead_time_hours': min_lead,
                'max_lead_time_hours': max_lead,
                'avg_price_diff_pct': avg_price_diff
            })

        return pd.DataFrame(metrics)

    def plot_validation_results(self, metrics_df, save_path='validation_results.png'):
        """검증 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 감지율
        ax = axes[0, 0]
        x_labels = [f"{row['upper_tf']}←{row['lower_tf']}" for _, row in metrics_df.iterrows()]
        ax.bar(x_labels, metrics_df['detection_rate_pct'])
        ax.set_ylabel('Detection Rate (%)')
        ax.set_title('H/L Detection Rate by Layer')
        ax.set_ylim([0, 105])
        for i, v in enumerate(metrics_df['detection_rate_pct']):
            ax.text(i, v + 2, f'{v:.1f}%', ha='center')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 2. 평균 선행 시간
        ax = axes[0, 1]
        ax.bar(x_labels, metrics_df['avg_lead_time_hours'].fillna(0))
        ax.set_ylabel('Lead Time (hours)')
        ax.set_title('Average Lead Time')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 3. 감지/미감지 수
        ax = axes[1, 0]
        width = 0.35
        x = np.arange(len(x_labels))
        ax.bar(x - width/2, metrics_df['detected'], width, label='Detected')
        ax.bar(x + width/2, metrics_df['missed'], width, label='Missed')
        ax.set_ylabel('Count')
        ax.set_title('Detected vs Missed H/L')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 4. 가격 오차
        ax = axes[1, 1]
        ax.bar(x_labels, metrics_df['avg_price_diff_pct'].fillna(0))
        ax.set_ylabel('Price Difference (%)')
        ax.set_title('Average Price Difference')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n검증 결과 저장: {save_path}")
        plt.close()


if __name__ == "__main__":
    import os
    import glob
    from macd_hl_labeling import process_all_timeframes

    print("=" * 60)
    print("계층적 H/L 전파 검증")
    print("=" * 60)

    # 데이터 로드
    data_dir = "data"
    pattern = os.path.join(data_dir, "BTC_USDT_USDT_15m_*.csv")
    files = glob.glob(pattern)

    if not files:
        print("\nBTC 데이터 파일이 없습니다.")
    else:
        latest_file = max(files, key=os.path.getmtime)
        print(f"\n데이터 로드: {latest_file}")

        df = pd.read_csv(latest_file)
        df['datetime'] = pd.to_datetime(df['datetime'])

        # H/L 라벨링
        print("\nH/L 라벨링 수행 중...")
        hl_results = process_all_timeframes(df)

        # 계층적 검증
        print("\n" + "=" * 60)
        print("계층적 전파 검증 시작")
        print("=" * 60)

        validator = HierarchicalValidator()
        validations = validator.validate_all_layers(hl_results)

        # 메트릭 계산
        print("\n" + "=" * 60)
        print("검증 메트릭")
        print("=" * 60)

        metrics_df = validator.calculate_metrics(validations)
        print("\n" + metrics_df.to_string(index=False))

        # 시각화
        os.makedirs('models', exist_ok=True)
        validator.plot_validation_results(metrics_df, 'models/validation_results.png')

        print("\n검증 완료!")
