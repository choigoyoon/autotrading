import sys
import os

def add_project_root_to_path():
    """
    프로젝트 루트 디렉토리('trading')를 sys.path의 최상단에 추가합니다.
    이 함수는 모듈이 임포트될 때 자동으로 호출되어,
    프로젝트 내 어느 위치에서 스크립트를 실행하더라도
    'src', 'tools', 'configs' 등의 최상위 모듈을 일관되게 임포트할 수 있도록 보장합니다.
    """
    try:
        this_file_path = os.path.abspath(__file__)
        # /trading/src/utils/path_manager.py -> /trading
        project_root = os.path.abspath(os.path.join(os.path.dirname(this_file_path), '..', '..'))
        
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
    except Exception as e:
        print(f"Project path 설정 중 오류 발생: {e}")
        raise

# 이 모듈이 임포트되는 시점에 경로 추가 함수를 바로 실행합니다.
add_project_root_to_path() 