import re
import logging
from typing import Any, Union
from pathlib import Path
import pandas as pd
from loguru import logger

def get_timeframe_category(timeframe: str) -> str:
    """
    CCXT 타임프레임 문자열을 기반으로 카테고리를 반환합니다.
    예: '1m' -> 'minutes', '1h' -> 'hours', '1d' -> 'days', '1w' -> 'weeks', '1M' -> 'months'
    """
    if not isinstance(timeframe, str) or not timeframe:
        return "unknown"
    
    unit = timeframe[-1]
    if unit == 'm':
        return "minutes"
    elif unit == 'h':
        return "hours"
    elif unit == 'd':
        return "days"
    elif unit == 'w':
        return "weeks"
    elif unit == 'M':
        return "months"
    return "others" # 예: 연 단위(Y) 또는 기타 커스텀

def timeframe_to_filename_str(timeframe: str) -> str:
    """Converts CCXT timeframe to a case-sensitive OS friendly string for filenames.
       '1m' -> '1min', '1M' -> '1month'. Others like h, d, w can be kept as is or expanded.
    """
    if not isinstance(timeframe, str) or not timeframe:
        raise ValueError("Input timeframe must be a non-empty string")

    # 정규표현식으로 숫자 부분과 단위 부분을 분리 시도
    match = re.fullmatch(r'(\d+)([a-zA-Z]+)', timeframe)
    if not match:
        # 매치가 안되면 (예: '1', 'm', 'Unsupported') 일단 소문자화 후 반환 (또는 오류 처리)
        # 이 부분은 프로젝트의 타임프레임 명명 규칙에 따라 더 엄격하게 처리할 수 있음
        return timeframe.lower() 

    num_part = match.group(1)
    unit_part = match.group(2)

    if unit_part == 'm': # 분 (minute)
        return f"{num_part}min"
    elif unit_part == 'M': # 월 (Month)
        return f"{num_part}month"
    # 시간(h), 일(d), 주(w)는 현재 그대로 사용. 필요시 변환 추가.
    # elif unit_part.lower() == 'h':
    #     return f"{num_part}hour"
    # elif unit_part.lower() == 'd':
    #     return f"{num_part}day"
    # elif unit_part.lower() == 'w':
    #     return f"{num_part}week"
    
    # 위의 특정 단위에 해당하지 않으면 원래 문자열에서 단위 부분만 소문자화하여 반환
    return f"{num_part}{unit_part.lower()}"

def filename_str_to_timeframe(filename_tf_part: str) -> str:
    """Converts filename string part back to CCXT timeframe."""
    if not isinstance(filename_tf_part, str) or not filename_tf_part:
        raise ValueError("Input filename_tf_part must be a non-empty string")

    if filename_tf_part.endswith('min'):
        return filename_tf_part.replace('min', 'm')
    elif filename_tf_part.endswith('month'):
        return filename_tf_part.replace('month', 'M')
    # elif filename_tf_part.endswith('hour'):
    #     return filename_tf_part.replace('hour', 'h')
    # elif filename_tf_part.endswith('day'):
    #     return filename_tf_part.replace('day', 'd')
    # elif filename_tf_part.endswith('week'):
    #     return filename_tf_part.replace('week', 'w')
    return filename_tf_part

# 다른 헬퍼 함수들도 여기에 추가 가능
# 예시:
def safe_division(
    numerator: Union[int, float], 
    denominator: Union[int, float], 
    default_val: float = 0.0
) -> float:
    """
    0으로 나누는 것을 방지하는 안전한 나누기 함수.
    
    Args:
        numerator (Union[int, float]): 분자.
        denominator (Union[int, float]): 분모.
        default_val (float, optional): 분모가 0일 때 반환할 기본값. Defaults to 0.0.

    Returns:
        float: 나눗셈 결과.
    """
    if denominator == 0:
        return default_val
    return float(numerator / denominator)

def get_project_root() -> Path:
    """프로젝트 루트 디렉토리를 반환합니다."""
    return Path(__file__).resolve().parent.parent.parent

def safe_numeric_conversion(value: Any, context: str = "") -> float | None:
    """
    타입 안전성을 보장하는 숫자 변환 함수.
    변환 실패 시 None을 반환하고 오류를 로깅합니다.
    """
    try:
        numeric_value = pd.to_numeric(value, errors='raise')
        if pd.isna(numeric_value):
            logger.warning(f"NA value found in safe_numeric_conversion: {context}")
            return None
        return float(numeric_value)
    except (ValueError, TypeError) as e:
        logger.error(f"Type conversion failed for {context}: value='{value}', error={e}")
        return None 