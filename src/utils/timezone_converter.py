import sys
from pathlib import Path
from typing import Literal, Union, Optional

import pandas as pd
from loguru import logger

SourceTimezone = Union[Literal["UTC"], Literal["Asia/Seoul"]]
TargetTimezone = Union[Literal["UTC"], Literal["Asia/Seoul"]]

class TimezoneConverter:
    """
    Pandas DataFrame의 타임스탬프 시간대를 변환하는 클래스.
    모든 타임스탬프는 변환 후에도 타임존 정보를 유지(aware)합니다.
    """
    def __init__(self, timestamp_col: str = "timestamp"):
        self.timestamp_col = timestamp_col

    def convert_timezone(
        self,
        df: pd.DataFrame,
        source_tz: SourceTimezone,
        target_tz: TargetTimezone,
    ) -> pd.DataFrame:
        """
        DataFrame의 타임스탬프 열 시간대를 변환합니다.

        Args:
            df (pd.DataFrame): 변환할 데이터프레임.
            source_tz (SourceTimezone): 원본 시간대.
            target_tz (TargetTimezone): 대상 시간대.

        Returns:
            pd.DataFrame: 시간대가 변환된 데이터프레임 (aware).
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("입력 데이터는 반드시 pandas DataFrame이어야 합니다.")
        
        df = df.copy()

        if self.timestamp_col not in df.columns:
            raise KeyError(f"타임스탬프 열 '{self.timestamp_col}'을 찾을 수 없습니다.")

        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], errors='coerce')
        
        # NaT 값이 있으면 변환 실패로 간주
        if df[self.timestamp_col].isnull().any():
            raise ValueError("타임스탬프 변환 중 유효하지 않은 값이 포함되어 있습니다.")

        # 원본 시간대 설정 (타임존 정보가 없는 경우, naive -> aware)
        if df[self.timestamp_col].dt.tz is None:
            df[self.timestamp_col] = df[self.timestamp_col].dt.tz_localize(source_tz)
        else: # 이미 aware 상태이면, 원본 시간대와 일치하는지 확인
            if str(df[self.timestamp_col].dt.tz) != source_tz:
                logger.warning(
                    f"입력 데이터의 시간대({df[self.timestamp_col].dt.tz})가 "
                    f"명시된 원본 시간대({source_tz})와 다릅니다. "
                    f"대상 시간대({target_tz})로 강제 변환합니다."
                )

        # 대상 시간대로 변환
        return df[self.timestamp_col].dt.tz_convert(target_tz).to_frame()


    def to_kst(self, df: pd.DataFrame) -> pd.DataFrame:
        """UTC 또는 naive 시간을 KST로 변환합니다."""
        return self.convert_timezone(df, source_tz="UTC", target_tz="Asia/Seoul")

    def to_utc(self, df: pd.DataFrame) -> pd.DataFrame:
        """KST 또는 naive 시간을 UTC로 변환합니다."""
        return self.convert_timezone(df, source_tz="Asia/Seoul", target_tz="UTC")


def get_project_root() -> Path:
    """프로젝트 루트 디렉토리를 반환합니다."""
    return Path(__file__).resolve().parent.parent.parent

def batch_convert_timezone(
    input_dir: Path,
    output_dir: Path,
    target_tz: TargetTimezone,
    timestamp_col: str = "timestamp",
    source_tz: Optional[SourceTimezone] = None,
) -> None:
    """
    특정 디렉토리의 모든 CSV/Parquet 파일의 시간대를 일괄 변환합니다.

    Args:
        input_dir (Path): 입력 파일 디렉토리.
        output_dir (Path): 출력 파일 디렉토리.
        target_tz (TargetTimezone): 변환할 목표 시간대.
        timestamp_col (str): 타임스탬프 열 이름.
        source_tz (Optional[SourceTimezone]): 원본 시간대. None이면 naive로 간주.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = TimezoneConverter(timestamp_col)
    
    files_to_process = list(input_dir.glob("*.csv")) + list(input_dir.glob("*.parquet"))

    if not files_to_process:
        logger.warning(f"'{input_dir}'에서 처리할 파일을 찾지 못했습니다.")
        return

    for file_path in files_to_process:
        try:
            logger.info(f"🔄 처리 중: {file_path.name}")
            
            df = pd.read_csv(file_path) if file_path.suffix == ".csv" else pd.read_parquet(file_path)

            # 결정된 소스 타임존. source_tz가 명시되지 않으면 KST를 기본값으로 사용
            determined_source_tz: SourceTimezone = source_tz or ("Asia/Seoul" if target_tz == "UTC" else "UTC")

            df_converted = converter.convert_timezone(df, determined_source_tz, target_tz)

            # 새 파일명 생성 (예: btc_data_kst.csv)
            new_file_name = f"{file_path.stem}_{target_tz.replace('/', '_').lower()}{file_path.suffix}"
            output_path = output_dir / new_file_name

            if output_path.exists():
                logger.info(f"⏩ 건너뛰기: '{output_path.name}' 파일이 이미 존재합니다.")
                continue

            if file_path.suffix == ".csv":
                df_converted.to_csv(output_path, index=False)
            else:
                df_converted.to_parquet(output_path, index=False)
            
            logger.success(f"✅ 완료: '{output_path.name}'")

        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"🚨 오류 발생 ({file_path.name}): {e}")
        except Exception as e:
            logger.error(f"🚨 예기치 않은 오류 발생 ({file_path.name}): {e}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    project_root = get_project_root()
    
    # --- 사용 예시 ---
    # 1. KST Parquet 파일을 UTC로 변환
    logger.info("\n--- KST -> UTC Parquet 파일 변환 시작 ---")
    batch_convert_timezone(
        input_dir=project_root / "data/rwa/parquet_kst",
        output_dir=project_root / "data/rwa/parquet_utc",
        target_tz="UTC",
        source_tz="Asia/Seoul"
    )

    # 2. UTC CSV 파일을 KST로 변환
    logger.info("\n--- UTC -> KST CSV 파일 변환 시작 ---")
    batch_convert_timezone(
        input_dir=project_root / "data/rwa/csv_utc",
        output_dir=project_root / "data/rwa/csv_kst",
        target_tz="Asia/Seoul",
        source_tz="UTC"
    )

    logger.info("\n🎉 모든 작업이 완료되었습니다.") 