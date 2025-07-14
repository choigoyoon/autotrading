import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

# --- Constants ---
TIMESTAMP_COLUMN = "timestamp"
EXPECTED_INTERVAL = pd.Timedelta(minutes=1)


def validate_timestamps(file_path: Path) -> None:
    """
    CSV 파일의 timestamp 열을 검증하여 리포트를 출력합니다.

    Args:
        file_path (Path): 검증할 CSV 파일 경로.
    """
    logger.info(f"'{file_path}' 파일 타임스탬프 검증을 시작합니다...")

    try:
        df: pd.DataFrame = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"파일을 찾을 수 없습니다: '{file_path}'")
        return
    except Exception as e:
        logger.error(f"'{file_path}' 파일 로딩 중 오류 발생: {e}")
        return

    if TIMESTAMP_COLUMN not in df.columns:
        logger.error(f"'{TIMESTAMP_COLUMN}' 열을 찾을 수 없습니다.")
        return

    try:
        # UTC 기준으로 타임스탬프 파싱
        df[TIMESTAMP_COLUMN] = pd.to_datetime(
            df[TIMESTAMP_COLUMN], unit="ms", utc=True, errors="raise"
        )
        logger.success("타임스탬프 파싱 완료 (UTC 기준).")
    except (ValueError, TypeError) as e:
        logger.error(f"타임스탬프 파싱 중 오류 발생: {e}")
        return

    # 1. 중복 검사
    duplicates: pd.DataFrame = df[df.duplicated(subset=[TIMESTAMP_COLUMN], keep=False)]
    if not duplicates.empty:
        logger.warning(f"{len(duplicates)}개의 중복된 타임스탬프를 발견했습니다:")
        logger.warning(duplicates.to_string())
    else:
        logger.success("중복된 타임스탬프가 없습니다.")

    # 2. 순서 검사
    if not df[TIMESTAMP_COLUMN].is_monotonic_increasing:
        logger.warning("타임스탬프가 시간순으로 정렬되어 있지 않습니다. 정렬 후 분석을 계속합니다.")
        df.sort_values(TIMESTAMP_COLUMN, inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        logger.success("타임스탬프가 시간순으로 올바르게 정렬되어 있습니다.")

    # 3. 누락 검사 (Gaps)
    time_diff: pd.Series = df[TIMESTAMP_COLUMN].diff()
    gaps: pd.DataFrame = df[time_diff > EXPECTED_INTERVAL]

    if not gaps.empty:
        logger.warning(f"{len(gaps)}개의 누락된 구간(gap)을 발견했습니다.")
        
        gaps_info = gaps[[TIMESTAMP_COLUMN]].copy()
        gaps_info['previous_timestamp'] = df.loc[gaps.index - 1, TIMESTAMP_COLUMN].values
        gaps_info['gap_duration'] = gaps_info[TIMESTAMP_COLUMN] - gaps_info['previous_timestamp']

        for _, row in gaps_info.head(10).iterrows():
            missing_duration = row['gap_duration'] - EXPECTED_INTERVAL
            logger.info(
                f"  - {row['previous_timestamp']} 부터 {missing_duration} 동안 데이터 누락 후 "
                f"{row[TIMESTAMP_COLUMN]} 에서 재개"
            )
        
        if len(gaps) > 10:
            logger.info(f"  ... (총 {len(gaps)}개 중 상위 10개만 표시)")
    else:
        logger.success(f"{EXPECTED_INTERVAL.total_seconds() / 60:.0f}분 간격의 누락된 타임스탬프가 없습니다.")
        
    logger.info("검증 완료.")


if __name__ == "__main__":
    # 로그 설정
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # 프로젝트 루트 경로를 기준으로 파일 경로 설정
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        csv_file = project_root / 'data/rwa/csv/btc_1min.csv'
        validate_timestamps(csv_file)
    except NameError:
        logger.error("프로젝트 루트를 찾을 수 없어 테스트를 실행할 수 없습니다.")
    except Exception as e:
        logger.error(f"스크립트 실행 중 예기치 않은 오류 발생: {e}") 