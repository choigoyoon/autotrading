import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict
import logging

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ComprehensiveSRDetector:
    """
    '진짜 스윙 포인트' (swing_high, swing_low)를 사용하여
    주요 지지/저항(S/R) 클러스터를 식별하는 클래스.
    """

    def __init__(self, cluster_eps_pct: float = 0.005, min_cluster_samples: int = 2):
        """
        Args:
            cluster_eps_pct (float): S/R 레벨을 클러스터링할 때 사용할 가격 백분율 기반 거리.
                                   현재 가격의 0.5% 내에 있는 포인트들을 같은 클러스터로 묶습니다.
            min_cluster_samples (int): 하나의 유효한 클러스터를 형성하기 위해 필요한 최소 S/R 포인트 수.
        """
        self.cluster_eps_pct = cluster_eps_pct
        self.min_cluster_samples = min_cluster_samples
        self.logger = logging.getLogger(__name__)

    def detect(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        데이터프레임에서 지지/저항 클러스터를 탐지합니다.

        Args:
            df (pd.DataFrame): 'high', 'low', 'swing_high', 'swing_low' 컬럼을 포함해야 함.

        Returns:
            Dict[str, List[Dict]]: 'support'와 'resistance' 클러스터 목록을 포함하는 딕셔너리.
        """
        if not all(col in df.columns for col in ['high', 'low', 'swing_high', 'swing_low']):
            raise ValueError("입력 DataFrame에 'high', 'low', 'swing_high', 'swing_low' 컬럼이 필요합니다.")

        # '진짜' 스윙 포인트를 S/R 후보로 추출
        supports = df[df['swing_low'] == 1]['low'].values
        resistances = df[df['swing_high'] == 1]['high'].values
        
        if len(supports) < self.min_cluster_samples or len(resistances) < self.min_cluster_samples:
            self.logger.warning("S/R 분석을 위한 스윙 포인트가 충분하지 않습니다.")
            return {'support_clusters': [], 'resistance_clusters': []}

        # 동적 DBSCAN `eps` 계산 (현재 가격 기준)
        last_price = df['close'].iloc[-1]
        dynamic_eps = last_price * self.cluster_eps_pct

        # 클러스터링 실행
        support_clusters = self._cluster_levels(supports, dynamic_eps)
        resistance_clusters = self._cluster_levels(resistances, dynamic_eps)

        self.logger.info(f"✅ S/R 분석 완료: 지지 클러스터 {len(support_clusters)}개, 저항 클러스터 {len(resistance_clusters)}개 탐지.")

        return {
            'support_clusters': support_clusters,
            'resistance_clusters': resistance_clusters
        }

    def _cluster_levels(self, levels: np.ndarray, eps: float) -> List[Dict]:
        """DBSCAN을 사용하여 S/R 레벨을 클러스터링합니다."""
        if len(levels) < self.min_cluster_samples:
            return []

        db = DBSCAN(eps=eps, min_samples=self.min_cluster_samples, metric='euclidean').fit(levels.reshape(-1, 1))
        
        clusters = []
        unique_labels = set(db.labels_)
        
        for k in unique_labels:
            if k == -1:  # 노이즈 포인트는 제외
                continue

            class_member_mask = (db.labels_ == k)
            cluster_points = levels[class_member_mask]
            
            if len(cluster_points) > 0:
                cluster_mean = cluster_points.mean()
                cluster_std = cluster_points.std()
                clusters.append({
                    'level': float(cluster_mean),
                    'strength': len(cluster_points),
                    'std_dev': float(cluster_std),
                    'points': cluster_points.tolist()
                })
        
        return sorted(clusters, key=lambda x: x['strength'], reverse=True)

if __name__ == '__main__':
    # --- 유닛 테스트를 위한 샘플 데이터 생성 ---
    print("ComprehensiveSRDetector 유닛 테스트 시작...")
    
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=500, freq='4H'))
    price_data = 40000 + np.random.randn(500).cumsum() * 20
    
    # 스윙 레벨을 만들기 위해 의도적으로 피크/밸리 생성
    price_data[100:110] = price_data[100:110] + 500
    price_data[250:260] = price_data[250:260] - 600
    price_data[400:410] = price_data[400:410] + 700

    sample_df = pd.DataFrame({
        'open': price_data - 10,
        'high': price_data + 100 + np.random.uniform(0, 50, 500),
        'low': price_data - 100 - np.random.uniform(0, 50, 500),
        'close': price_data,
        'volume': np.random.randint(100, 1000, 500)
    }, index=dates)

    # 볼륨 컨펌을 위한 데이터 조작
    sample_df.loc[sample_df.index[250:260], 'volume'] = 2000

    current_price = sample_df['close'].iloc[-1]
    
    # --- 탐지기 실행 ---
    detector = ComprehensiveSRDetector(cluster_eps_pct=0.005, min_cluster_samples=2)
    results = detector.detect(sample_df)
    
    print(f"\n현재 가격: {current_price:.2f}")
    
    print("\n--- 지지 레벨 (상위 5개) ---")
    if results['support_clusters']:
        for cluster in results['support_clusters'][:5]:
            print(f"가격: {cluster['level']:.2f}, 강도: {cluster['strength']}, 표준편차: {cluster['std_dev']:.2f}, 포인트: {cluster['points']}")
    else:
        print("탐지된 지지 레벨 없음")

    print("\n--- 저항 레벨 (상위 5개) ---")
    if results['resistance_clusters']:
        for cluster in results['resistance_clusters'][:5]:
            print(f"가격: {cluster['level']:.2f}, 강도: {cluster['strength']}, 표준편차: {cluster['std_dev']:.2f}, 포인트: {cluster['points']}")
    else:
        print("탐지된 저항 레벨 없음")
        
    print("\n유닛 테스트 완료.") 