import json
from collections import Counter

try:
    # PowerShell 리디렉션 출력을 처리하기 위해 'utf-16' 인코딩 사용
    with open('pyright_full_report.json', 'r', encoding='utf-16') as f:
        data = json.load(f)

    general_diagnostics = data.get('generalDiagnostics', [])
    
    errors = [d.get('rule', 'unknown') for d in general_diagnostics]
    files_with_errors = set(d.get('file', '') for d in general_diagnostics)

    print('=== pyright 오류 현황 ===')
    print(f"총 오류 개수: {len(general_diagnostics)}")
    print(f"오류 파일 수: {len(files_with_errors)}")
    
    error_summary = data.get('summary', {})
    error_count = error_summary.get('errorCount', 0)
    warning_count = error_summary.get('warningCount', 0)
    information_count = error_summary.get('informationCount', 0)
    
    print(f"Error: {error_count}, Warning: {warning_count}, Information: {information_count}")

    print('\n=== 오류 유형별 통계 (상위 10개) ===')
    for rule, count in Counter(errors).most_common(10):
        print(f'{rule}: {count}개')

except FileNotFoundError:
    print("오류: 'pyright_full_report.json' 파일을 찾을 수 없습니다.")
except json.JSONDecodeError:
    print("오류: 'pyright_full_report.json' 파일이 올바른 JSON 형식이 아닙니다.")
except Exception as e:
    print(f"알 수 없는 오류가 발생했습니다: {e}") 