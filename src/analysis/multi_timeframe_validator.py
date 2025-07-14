import pandas as pd
from pathlib import Path
import warnings
import sys

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# pandas-ta가 설치되어 있다고 가정합니다. 만약 없다면, 간단한 구현으로 대체합니다.
try:
    import pandas_ta as ta # type: ignore
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    class ta_mock: # 이름 충돌 방지를 위해 클래스 이름 변경
        @staticmethod
        def ema(*args, **kwargs): pass
        @staticmethod
        def macd(*args, **kwargs): pass
        @staticmethod
        def rsi(*args, **kwargs): pass
    ta = ta_mock() # 인스턴스 할당

warnings.filterwarnings('ignore', category=FutureWarning)

class MultiTimeframeValidator:
    """
    다중 타임프레임의 시장 상황을 계층적으로 분석하고 일관성을 검증하여
    현재 시장 환경이 매매에 유리한지 점수화합니다.
    """
    TIMEFRAME_HIERARCHY = {
        'macro': ['1w', '1d', '12h'],
        'meso': ['8h', '4h', '2h'],
        'micro': ['1h'],
        'execution': ['30m', '15m', '5m', '1m']
    }
    ALL_TIMEFRAMES = [tf for tfs in TIMEFRAME_HIERARCHY.values() for tf in tfs]

    def __init__(self, base_path: Path):
        """
        분석기 초기화

        Args:
            base_path (Path): 프로젝트 루트 경로.
        """
        self.base_path = base_path
        self.data_path = self.base_path / "data" / "processed" / "btc_usdt_kst" / "resampled_ohlcv"
        self.dataframes = {}
        self.analysis_results = {}
        self.is_prepared = False
        print("MultiTimeframeValidator가 초기화되었습니다.")

    def prepare_data(self):
        """데이터 로딩 및 지표 계산을 한 번만 수행합니다."""
        if self.is_prepared:
            print("데이터가 이미 준비되었습니다.")
            return
        
        self._load_data()
        self._calculate_indicators()
        self.is_prepared = True
        print("모든 타임프레임 데이터 및 지표 준비 완료.")

    def _load_data(self):
        """모든 타임프레임의 데이터를 로드합니다."""
        print("모든 타임프레임 데이터 로딩 중...")
        # '1w' -> '1week', '1d' -> '1day' 등 실제 파일명에 맞게 매핑
        tf_map = {'1w': '1week', '1d': '1day'}
        
        for tf_key in self.ALL_TIMEFRAMES:
            tf_filename = tf_map.get(tf_key, tf_key)
            file_path = self.data_path / f"{tf_filename}.parquet"
            if not file_path.exists():
                # 'm' -> 'min' 변형 시도
                file_path = self.data_path / f"{tf_filename.replace('m', 'min')}.parquet"
                if not file_path.exists():
                     raise FileNotFoundError(f"{tf_key} 데이터를 찾을 수 없습니다: {file_path}")
            
            self.dataframes[tf_key] = pd.read_parquet(file_path)
        print(f"{len(self.dataframes)}개 타임프레임 데이터 로딩 완료.")

    def _calculate_indicators(self):
        """모든 데이터프레임에 대해 기술 지표를 계산합니다."""
        if not self.dataframes:
            raise RuntimeError("데이터가 로드되지 않았습니다. _load_data()를 먼저 실행하세요.")

        print("기술 지표 계산 중...")
        
        for tf, df in self.dataframes.items():
            if PANDAS_TA_AVAILABLE:
                df.ta.ema(length=20, append=True, col_names=(f"EMA_20",))
                df.ta.ema(length=50, append=True, col_names=(f"EMA_50",))
                df.ta.macd(append=True)
                df.ta.rsi(append=True)
            else:
                # pandas-ta가 없을 경우, 간단한 EMA/MACD/RSI 계산
                print(f"경고: {tf}에 대해 'pandas-ta' 없이 기본 지표 계산 중...")
                df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
                df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
                ema_12 = df['close'].ewm(span=12, adjust=False).mean()
                ema_26 = df['close'].ewm(span=26, adjust=False).mean()
                df['MACD_12_26_9'] = ema_12 - ema_26
                df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
                df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
                if not loss.eq(0).all():
                    rs = gain / loss.where(loss != 0, 1e-9)
                    df['RSI_14'] = 100 - (100 / (1 + rs))
                else:
                    df['RSI_14'] = 100

    def analyze_single_timeframe(self, tf: str, at_timestamp=None) -> dict:
        """단일 타임프레임의 상태를 분석합니다."""
        df = self.dataframes[tf]
        if at_timestamp:
            try:
                data = df.loc[:at_timestamp].iloc[-1]
            except (IndexError, KeyError):
                return {"error": "해당 시간에 데이터 없음"}
        else:
            data = df.iloc[-1]

        # MACD 컬럼 이름이 pandas_ta 기본값(MACDh_12_26_9)과 일치하는지 확인
        macd_hist_col = 'MACDh_12_26_9'
        rsi_col = 'RSI_14'

        if macd_hist_col not in data.index or rsi_col not in data.index:
            return {"error": f"{tf}에서 필요한 지표({macd_hist_col}, {rsi_col})를 찾을 수 없음"}

        trend = "up" if data['EMA_20'] > data['EMA_50'] else "down"
        macd_signal = "bullish" if data[macd_hist_col] > 0 else "bearish"
        
        # RSI를 이용한 신호 강도 계산 (0~1 사이 값, 50을 기준으로 멀어질수록 강함)
        rsi_val = data[rsi_col]
        strength = abs(rsi_val - 50) / 50

        return {
            "timestamp": data.name,
            "trend": trend,
            "rsi": rsi_val,
            "macd_signal": macd_signal,
            "signal_strength": strength
        }

    def analyze_all_timeframes(self, at_timestamp=None):
        """모든 타임프레임의 상황을 분석합니다."""
        if not self.is_prepared:
            raise RuntimeError("데이터가 준비되지 않았습니다. prepare_data()를 먼저 호출하세요.")
        
        print(f"분석 기준 시간: {at_timestamp or '최신'}")
        for tf in self.ALL_TIMEFRAMES:
            self.analysis_results[tf] = self.analyze_single_timeframe(tf, at_timestamp)

    def check_hierarchy_consistency(self) -> dict:
        """타임프레임 계층 간의 추세 일관성을 검증합니다."""
        if not self.analysis_results:
            raise RuntimeError("분석 결과가 없습니다. analyze_all_timeframes()를 먼저 실행하세요.")
        
        consistency = {}
        for level, tfs in self.TIMEFRAME_HIERARCHY.items():
            trends = [self.analysis_results[tf].get('trend') for tf in tfs if 'error' not in self.analysis_results.get(tf, {})]
            if trends:
                is_consistent_up = all(t == 'up' for t in trends)
                is_consistent_down = all(t == 'down' for t in trends)
                
                if is_consistent_up:
                    consistency[level] = 'up'
                elif is_consistent_down:
                    consistency[level] = 'down'
                else:
                    consistency[level] = 'mixed'
        return consistency

    def score_environment(self) -> float:
        """분석 결과를 바탕으로 현재 시장 환경을 점수화합니다."""
        consistency = self.check_hierarchy_consistency()
        score = 0.0
        weights = {'macro': 40, 'meso': 30, 'micro': 20, 'execution': 10}
        
        for level, trend in consistency.items():
            if trend == 'up':
                score += weights[level]
            elif trend == 'down':
                score -= weights[level]
        
        return score

# 메인 실행 블록
if __name__ == '__main__':
    try:
        project_root = Path(__file__).resolve().parents[2]
        validator = MultiTimeframeValidator(base_path=project_root)
        
        validator.prepare_data()
        validator.analyze_all_timeframes()
        
        consistency_report = validator.check_hierarchy_consistency()
        print("\n--- 계층별 추세 일관성 ---")
        for level, trend in consistency_report.items():
            print(f"{level.title():>10s}: {trend.upper()}")
            
        environment_score = validator.score_environment()
        print("\n--- 최종 시장 환경 점수 ---")
        print(f"Score: {environment_score:.2f} (-100: Strong Bear, 100: Strong Bull)")

        if environment_score > 50:
            print("판단: 강세 신호가 여러 타임프레임에 걸쳐 일관되게 나타나고 있습니다. 매수 포지션에 유리한 환경입니다.")
        elif environment_score < -50:
            print("판단: 약세 신호가 여러 타임프레임에 걸쳐 일관되게 나타나고 있습니다. 매도 포지션에 유리한 환경입니다.")
        else:
            print("판단: 추세가 혼재되어 있어, 명확한 방향성을 기다리는 것이 좋습니다.")

    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n오류가 발생했습니다: {e}")
    except Exception as e:
        print(f"\n예상치 못한 오류가 발생했습니다: {e}")
