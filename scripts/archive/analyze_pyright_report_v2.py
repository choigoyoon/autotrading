import json
from collections import defaultdict
from typing import Dict, List, Any

# 리포트 파일 읽기 (예외 처리 추가)
try:
    with open('pyright_report_latest.json', 'r', encoding='utf-16') as f:
        report_data = json.load(f)
        if not isinstance(report_data, dict):
             print("오류: JSON 최상위 타입이 딕셔너리가 아닙니다.")
             report_data = {} # 빈 딕셔너리로 초기화
except FileNotFoundError:
    print("오류: 'pyright_report_latest.json' 파일을 찾을 수 없습니다. pyright를 먼저 실행하세요.")
    report_data = {}
except json.JSONDecodeError:
    print("오류: 'pyright_report_latest.json' 파일 분석에 실패했습니다.")
    report_data = {}

# 오류 타입별 분석
error_types: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
diagnostics = report_data.get('generalDiagnostics', [])

if isinstance(diagnostics, list):
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue

        rule = diagnostic.get('rule', 'unknown')
        file_path = diagnostic.get('file', 'unknown_file')
        
        # range 및 start가 딕셔너리인지 확인
        range_info = diagnostic.get('range', {})
        start_info = range_info.get('start', {}) if isinstance(range_info, dict) else {}
        line_num = start_info.get('line', -1) if isinstance(start_info, dict) else -1

        error_types[rule].append({
            'file': file_path,
            'line': line_num,
            'message': diagnostic.get('message', 'No message')
        })

# 우선순위 룰
priority_rules = [
    'reportOperatorIssue', 'reportOptionalOperand', 'reportArgumentType',
    'reportGeneralTypeIssues', 'reportUnusedCallResult', 'reportUnusedImport'
]

print("=== 우선순위별 오류 분석 ===")
total_errors = len(diagnostics) if isinstance(diagnostics, list) else 0

for rule in priority_rules:
    count = len(error_types[rule])
    if count > 0:
        print(f"\n🔴 {rule}: {count}개")
        
        files = defaultdict(int)
        for error in error_types.get(rule, []):
             if isinstance(error, dict):
                files[error.get('file', 'unknown_file')] += 1
        
        print("   파일별 분포:")
        for file, cnt in sorted(files.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   - {file}: {cnt}개")
            
        print("   예시:")
        for error in error_types.get(rule, [])[:3]:
            if isinstance(error, dict):
                line = error.get('line', -1)
                display_line = line + 1 if isinstance(line, int) and line != -1 else 'N/A'
                message = error.get('message', 'No message')
                file = error.get('file', 'N/A')
                print(f"   - {file}:{display_line} - {str(message)[:60]}...")

print(f"\n총 오류 수: {total_errors}") 