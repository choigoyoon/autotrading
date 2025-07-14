import json
from collections import Counter, defaultdict

def analyze_report_by_file(file_path='pyright_report.json'):
    """
    pyright_report.json 파일을 읽어, 오류 유형별로 어떤 파일에서
    가장 많이 발생하는지 분석하여 출력합니다.
    """
    encodings_to_try = ['utf-8', 'utf-16', 'latin-1', 'cp949']
    data = None

    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            # print(f"Successfully loaded file with '{encoding}' encoding.")
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue

    if not data:
        print("Could not read or parse the report file.")
        return

    diagnostics = data.get('generalDiagnostics', [])
    if not diagnostics:
        print("No diagnostics found in the report.")
        return

    # 오류 규칙별 파일 빈도수 계산
    errors_by_rule = defaultdict(lambda: Counter())
    for d in diagnostics:
        rule = d.get('rule')
        file = d.get('file')
        if rule and file:
            errors_by_rule[rule].update([file])

    if not errors_by_rule:
        print("No errors with rules found.")
        return

    # 가장 흔한 오류 유형 2개에 대해 상위 3개 파일 출력
    top_rules = [rule for rule, _ in Counter({r: sum(c.values()) for r, c in errors_by_rule.items()}).most_common(2)]

    for rule in top_rules:
        print(f"\n--- Top 3 files for rule '{rule}' ---")
        for file, count in errors_by_rule[rule].most_common(3):
            print(f"  {file}: {count}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        analyze_report_by_file(sys.argv[1])
    else:
        analyze_report_by_file() 