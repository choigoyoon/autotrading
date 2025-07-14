import json
import sys
from collections import Counter, defaultdict
from typing import TypedDict, cast

if sys.version_info < (3, 11):
    from typing_extensions import NotRequired
else:
    from typing import NotRequired  # type: ignore[unreachable]


class Diagnostic(TypedDict):
    file: str
    message: str
    severity: str
    rule: NotRequired[str]


class Summary(TypedDict):
    errorCount: int
    warningCount: int
    informationCount: int
    hintCount: int


class PyrightReport(TypedDict):
    version: str
    time: str
    generalDiagnostics: list[Diagnostic]
    summary: Summary


def analyze_report_by_file(file_path: str) -> None:
    """
    Analyzes a Pyright report to find which files have the most errors for each rule.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data: PyrightReport = cast(PyrightReport, json.load(f))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing the report file: {e}")
        return

    diagnostics = data.get("generalDiagnostics", [])

    if not diagnostics:
        print("No diagnostics found in the report.")
        return

    errors_by_rule: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for d in diagnostics:
        rule = d.get("rule", "unknown")
        file = d.get("file", "unknown_file")
        errors_by_rule[rule].update([file])

    if not errors_by_rule:
        print("No errors with rules found.")
        return

    rule_counts = {rule: sum(counts.values()) for rule, counts in errors_by_rule.items()}
    top_rules = [
        rule for rule, _ in Counter(rule_counts).most_common(2)
    ]

    for rule in top_rules:
        print(f"\n--- Top 3 files for rule '{rule}' ---")
        for file, count in errors_by_rule[rule].most_common(3):
            print(f"  {file}: {count}")


if __name__ == "__main__":
    REPORT_PATH = "pyright_report.json"
    analyze_report_by_file(REPORT_PATH) 