import sys
from pathlib import Path

def add_project_root_to_path() -> None:
    """
    프로젝트 루트 디렉토리('trading')를 sys.path의 최상단에 추가합니다.
    이 함수는 모듈이 임포트될 때 자동으로 호출되어,
    프로젝트 내 어느 위치에서 스크립트를 실행하더라도
    'src', 'tools', 'configs' 등의 최상위 모듈을 일관되게 임포트할 수 있도록 보장합니다.
    """
    try:
        # 이 파일의 절대 경로를 Path 객체로 가져옵니다.
        this_file_path = Path(__file__).resolve()
        # /trading/src/utils/path_manager.py -> /trading
        # .parent를 두 번 사용하여 상위 디렉토리로 이동합니다.
        project_root = this_file_path.parent.parent.parent
        
        # str으로 변환하여 sys.path와 비교 및 추가
        project_root_str = str(project_root)
        
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
            
    except (NameError, IndexError) as e:
        # __file__이 정의되지 않았거나, parent 구조가 예상과 다를 경우
        print(f"Project path 설정 중 오류 발생: {e}")
        # 개발 환경이나 특정 실행 방식에서는 __file__이 없을 수 있으므로,
        # 에러를 다시 raise하는 대신 경고만 출력하고 넘어갈 수 있습니다.
        # 하지만 현재 구조에서는 경로 설정이 필수적이므로 raise를 유지합니다.
        raise

# 이 모듈이 임포트되는 시점에 경로 추가 함수를 바로 실행합니다.
add_project_root_to_path() 