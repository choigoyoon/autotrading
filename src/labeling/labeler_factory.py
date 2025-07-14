from .enhanced_macd_labeler import EnhancedMACDLabeler
import logging

logger = logging.getLogger(__name__)

class LabelerFactory:
    @staticmethod
    def create_labeler(labeler_type: str, macd_params: dict, **other_labeler_params):
        """
        Creates a labeler instance based on type and parameters.
        Args:
            labeler_type (str): The type of labeler to create.
            macd_params (dict): Parameters for MACD calculation (fast, slow, signal, use_numba).
            **other_labeler_params: Other parameters specific to the labeler type (e.g., min_duration).
                                   These are typically NOT passed to constructor but to generate_labels method.
                                   However, if a labeler needs them in constructor, it can pick them up.
                                   For EnhancedMACDLabeler, constructor takes MACD params.
        """
        logger.info(f"Attempting to create labeler of type '{labeler_type}'")
        logger.debug(f"MACD params for factory: {macd_params}")
        logger.debug(f"Other labeler_specific_params for factory (passed to generate_labels): {other_labeler_params}")
        
        # EnhancedMACDLabeler 생성자에는 MACD 관련 파라미터만 전달
        constructor_params = {
            'fast': macd_params.get('fast', 12),
            'slow': macd_params.get('slow', 26),
            'signal': macd_params.get('signal', 9),
            'use_numba': macd_params.get('use_numba', True)
        }

        if labeler_type == "histogram_extremes_v2" or \
           labeler_type == "histogram_extremes" or \
           labeler_type == "histogram_threshold_cross" or \
           labeler_type == "macd":
            # 현재 EnhancedMACDLabeler는 config에 따라 다른 로직을 가질 수 있으므로, 
            # factory는 단순히 지정된 클래스를 생성하는 역할에 집중합니다.
            # EnhancedMACDLabeler가 "histogram_threshold_cross" 로직을 사용하도록 내부가 수정된 상태여야 합니다.
            logger.info(f"Creating EnhancedMACDLabeler with constructor_params: {constructor_params} for type '{labeler_type}'")
            return EnhancedMACDLabeler(**constructor_params)
        # Add other labeler types here
        # elif labeler_type == "crossover_basic":
        #     return CrossoverBasicLabeler(**macd_params, **other_labeler_params)
        else:
            logger.error(f"Unknown labeler type requested: {labeler_type}")
            raise ValueError(f"Unknown labeler type: {labeler_type}")

if __name__ == "__main__":
    # 이 블록은 python -m src.labeling.labeler_factory 또는 python src/labeling/labeler_factory.py (PYTHONPATH 설정 시)로 직접 실행될 때만 동작합니다.
    
    import pandas as pd
    from pathlib import Path
    import sys

    # --- 프로젝트 경로 설정 (labeler_factory.py가 src/labeling/ 내에 위치하므로, src 폴더를 sys.path에 추가) ---
    try:
        # 이 파일(labeler_factory.py)의 위치를 기준으로 프로젝트 루트를 추정합니다.
        # __file__ -> .../src/labeling/labeler_factory.py
        # .parent -> .../src/labeling/
        # .parent.parent -> .../src/
        # .parent.parent.parent -> 프로젝트 루트
        CURRENT_FILE_PATH = Path(__file__).resolve()
        SRC_DIR = CURRENT_FILE_PATH.parent.parent # .../src/
        PROJECT_ROOT = SRC_DIR.parent

        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        if str(PROJECT_ROOT) not in sys.path: # 혹시 모를 경우를 위해 프로젝트 루트도 추가
             sys.path.insert(0, str(PROJECT_ROOT)) # sys.path의 맨 앞에 추가
        
        # print(f"[Factory Direct Run] CURRENT_FILE_PATH: {CURRENT_FILE_PATH}")
        # print(f"[Factory Direct Run] SRC_DIR: {SRC_DIR}")
        # print(f"[Factory Direct Run] PROJECT_ROOT: {PROJECT_ROOT}")
        # print(f"[Factory Direct Run] sys.path: {sys.path}")

    except NameError:
        # __file__이 정의되지 않은 환경 (예: 일부 대화형 콘솔)
        # 이 경우, 현재 작업 디렉토리를 기준으로 src 폴더를 찾습니다.
        # 이 스크립트를 프로젝트 루트에서 python -m src.labeling.labeler_factory 형태로 실행한다고 가정합니다.
        PROJECT_ROOT = Path.cwd()
        SRC_DIR = PROJECT_ROOT / 'src'
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
    
    # EnhancedMACDLabeler 임포트는 LabelerFactory 클래스 정의 위에 이미 있음
    # from .enhanced_macd_labeler import EnhancedMACDLabeler # 이미 클래스 레벨에서 임포트됨

    # 로깅 설정 (이미 파일 상단에 logger = logging.getLogger(__name__)가 있음)
    # 여기서 추가로 basicConfig를 설정하면, 다른 모듈의 로깅에도 영향을 줄 수 있으므로 주의.
    # 테스트 목적상 루트 로거에 간단히 설정하거나, 기존 로거를 활용합니다.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    script_logger = logging.getLogger("labeler_factory_direct_run") # 이 스크립트 실행용 로거

    # fetch_data_for_factory_test 함수 제거 또는 주석 처리
    # def fetch_data_for_factory_test(symbol='BTC/USDT', timeframe='1m', limit=150):
    #     ...

    def load_data_from_local_parquet(file_path: Path) -> pd.DataFrame:
        script_logger.info(f"로컬 Parquet 파일에서 데이터 로드 시도: {file_path}")
        if not file_path.exists():
            script_logger.error(f"데이터 파일을 찾을 수 없습니다: {file_path}")
            return pd.DataFrame()
        try:
            df = pd.read_parquet(file_path)
            script_logger.info(f"성공적으로 {len(df)}개의 레코드를 로드했습니다. 파일: {file_path.name}")
            # 필요한 컬럼 확인 (최소 'close' 필요)
            if 'close' not in df.columns:
                script_logger.error(f"로드된 DataFrame에 'close' 컬럼이 없습니다. 컬럼: {df.columns.to_list()}")
                return pd.DataFrame()
            # 타임스탬프 인덱스 확인
            if not isinstance(df.index, pd.DatetimeIndex):
                script_logger.warning(f"로드된 DataFrame의 인덱스가 DatetimeIndex가 아닙니다. 타입: {type(df.index)}")
                # 필요시 변환 시도 (예: df.index = pd.to_datetime(df.index))
            return df
        except Exception as e:
            script_logger.error(f"Parquet 파일 로드 중 오류 발생 ({file_path.name}): {e}", exc_info=True)
            return pd.DataFrame()

    script_logger.info("LabelerFactory 직접 실행 테스트 시작 (로컬 파일 사용)...")
    
    # 1. 데이터 가져오기 (로컬 Parquet 파일에서)
    # 파일 경로는 PROJECT_ROOT를 기준으로 설정합니다.
    # PROJECT_ROOT는 이 스크립트 실행 시점에 올바르게 설정되어야 합니다.
    # (예: E:\trading)
    # data/raw/BTC_USDT/minutes/1m_basedata.parquet
    # 이전 경로 설정 로직에서 PROJECT_ROOT가 e:\trading 으로 잘 잡혔다고 가정
    parquet_file_path = PROJECT_ROOT / "data" / "raw" / "BTC_USDT" / "minutes" / "1m_basedata.parquet"
    
    script_logger.info(f"사용할 Parquet 파일 경로: {parquet_file_path}")
    test_df = load_data_from_local_parquet(parquet_file_path)

    if test_df.empty:
        script_logger.error("데이터를 로드하지 못해 테스트를 진행할 수 없습니다.")
        # 대체 파일 시도 (1min_basedata.parquet)
        script_logger.info("대체 파일 1min_basedata.parquet 로드 시도...")
        parquet_file_path_alt = PROJECT_ROOT / "data" / "raw" / "BTC_USDT" / "minutes" / "1min_basedata.parquet"
        test_df = load_data_from_local_parquet(parquet_file_path_alt)
        if test_df.empty:
            script_logger.error("대체 파일에서도 데이터를 로드하지 못했습니다.")
            sys.exit(1)
        else:
            script_logger.info(f"대체 파일 {parquet_file_path_alt.name} 사용.")

    # MACD 계산에 충분한 데이터가 있는지 확인 (선택 사항, 대용량 파일이므로 앞부분만 사용할 수도 있음)
    # if len(test_df) < 50:
    #     script_logger.error(f"로드된 데이터가 너무 적습니다 ({len(test_df)}개). MACD 계산 및 라벨링에 부적합합니다.")
    #     sys.exit(1)

    # 2. LabelerFactory를 통해 EnhancedMACDLabeler 인스턴스 생성
    # (이하 로직은 이전과 거의 동일)
    logger.info("테스트: LabelerFactory를 통해 라벨러를 생성합니다...")
    
    # MACD 파라미터 (LabelerFactory의 create_labeler 스펙에 맞게 제공)
    macd_config_params = {
        'fast': 12,
        'slow': 26,
        'signal': 9,
        'use_numba': True
    }
    # other_labeler_params는 EnhancedMACDLabeler 생성자에 직접 전달되지 않음 (필요 시 로직 수정)
    
    # 라벨러 타입은 'macd' 또는 수정된 EnhancedMACDLabeler가 처리하는 타입 중 하나여야 함.
    # 현재 EnhancedMACDLabeler는 히스토그램 부호 기반으로 수정되었음.
    try:
        # LabelerFactory.create_labeler는 staticmethod이므로 클래스명으로 직접 호출
        labeler_instance = LabelerFactory.create_labeler(
            labeler_type="macd", # 'macd'가 EnhancedMACDLabeler를 반환하도록 factory가 수정되었음
            macd_params=macd_config_params
        )
        script_logger.info(f"성공적으로 라벨러 인스턴스 생성: {type(labeler_instance)}")
    except Exception as e:
        script_logger.error(f"라벨러 인스턴스 생성 중 오류: {e}", exc_info=True)
        sys.exit(1)

    # 3. 라벨 생성
    script_logger.info("라벨 생성을 시작합니다...")
    labels = labeler_instance.generate_enhanced_labels(test_df.copy())

    # 4. 결과 출력
    if labels.empty:
        script_logger.warning("라벨이 생성되지 않았습니다.")
    else:
        total_labels = len(labels)
        buy_signals = (labels == 1).sum()
        sell_signals = (labels == -1).sum()
        hold_signals = (labels == 0).sum()

        script_logger.info("--- 라벨링 결과 (LabelerFactory 직접 실행) ---")
        script_logger.info(f"처리된 총 데이터 포인트: {total_labels}")
        script_logger.info(f"매수 신호 (+1): {buy_signals}")
        script_logger.info(f"매도 신호 (-1): {sell_signals}")
        script_logger.info(f"관망 신호 (0): {hold_signals}")
        if total_labels > 0:
            script_logger.info("라벨 분포 비율:")
            script_logger.info(f"  매수 (+1): {buy_signals / total_labels * 100:.2f}%")
            script_logger.info(f"  매도 (-1): {sell_signals / total_labels * 100:.2f}%")
            script_logger.info(f"  관망 (0): {hold_signals / total_labels * 100:.2f}%")
        script_logger.info("생성된 라벨 샘플 (앞 5개):")
        print(labels.head())
        script_logger.info("생성된 라벨 샘플 (뒤 5개):")
        print(labels.tail())
    
    script_logger.info("LabelerFactory 직접 실행 테스트 종료.") 