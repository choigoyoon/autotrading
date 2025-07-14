import json
import sys
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


def analyze_report(report_path: str) -> None:
    """
    Analyzes a Pyright report JSON file and prints a summary of diagnostics.

    Args:
        report_path: The path to the Pyright report JSON file.
    """
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            # json.load()가 Any를 반환하므로, 명시적으로 캐스팅합니다.
            # 이중 캐스팅은 특정 linter와의 호환성을 위해 사용될 수 있습니다.
            _data: PyrightReport = cast(PyrightReport, json.load(f))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing the report file: {e}")
        return

    # diagnostics = data.get("generalDiagnostics", [])

    # ... existing code ...