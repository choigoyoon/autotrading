import pandas as pd
from pandas import DataFrame, Series

def perform_temp_analysis() -> None:
    """
    가상의 데이터프레임을 생성하고, 날짜 계산 및 필터링을 수행합니다.
    """
    # 가상의 데이터프레임 생성
    data = {
        'start_date': ['2023-01-01', '2023-01-05', '2023-01-10', pd.NaT, 'not_a_date'],
        'end_date': ['2023-01-11', '2023-01-12', '2023-01-15', '2023-01-20', '2023-01-25'],
        'value': [10, 20, 15, 30, 25],
        'condition1': [True, False, True, True, False],
        'condition2': [True, True, False, True, False]
    }
    df: DataFrame = pd.DataFrame(data)

    # pd.to_datetime으로 타입을 명확히 하고 .dt 접근자 사용
    # NaT나 잘못된 문자열이 있어도 errors='coerce'로 처리하여 NaT로 변환
    start_dates: Series = pd.to_datetime(df['start_date'], errors='coerce')
    end_dates: Series = pd.to_datetime(df['end_date'], errors='coerce')
    df['duration_days'] = (end_dates - start_dates).dt.days

    print("계산된 기간 (일):")
    print(df['duration_days'])

    # 불리언 필터링
    # df['conditionX']는 Series[bool] 타입이므로 `&` 연산이 가능합니다.
    # pyright가 이를 올바르게 추론할 수 있도록 타입을 명시하여 `# type: ignore`를 제거합니다.
    condition1: Series = df['condition1'].astype(bool)
    condition2: Series = df['condition2'].astype(bool)
    
    filtered_df: DataFrame = df[condition1 & condition2]

    print("\n필터링된 데이터프레임:")
    print(filtered_df)

if __name__ == "__main__":
    perform_temp_analysis() 