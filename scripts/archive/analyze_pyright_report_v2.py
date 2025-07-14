import json
import sys
from collections import defaultdict
from typing import TypedDict, cast

if sys.version_info < (3, 11):
    from typing_extensions import NotRequired
else:
    from typing import NotRequired  # type: ignore[unreachable]


# --- TypedDict Definitions ---
class Start(TypedDict):
    line: int
    character: int


class Range(TypedDict):
    start: Start
    end: Start


class Diagnostic(TypedDict):
    file: str
    message: str
    severity: str
    range: NotRequired[Range]
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


class ErrorDetails(TypedDict):
    file: str
    line: int
    message: str


def read_report(file_path: str) -> PyrightReport | None:
    """Reads and parses the Pyright JSON report file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # The loaded data is cast to the specific TypedDict.
            return cast(PyrightReport, json.load(f))
    except FileNotFoundError:
        print(f"Error: Report file not found at '{file_path}'.")
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON from '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


def analyze_diagnostics(diagnostics: list[Diagnostic]) -> None:
    """Analyzes and prints diagnostics by priority."""
    error_types: defaultdict[str, list[ErrorDetails]] = defaultdict(list)

    for diagnostic in diagnostics:
        rule = diagnostic.get("rule", "unknown")
        line_num = diagnostic.get("range", {}).get("start", {}).get("line", -1)

        details = ErrorDetails(
            file=diagnostic.get("file", "unknown_file"),
            line=line_num,
            message=diagnostic.get("message", "No message"),
        )
        error_types[rule].append(details)

    priority_rules = [
        "reportOperatorIssue", "reportOptionalOperand", "reportArgumentType",
        "reportGeneralTypeIssues", "reportUnusedCallResult", "reportUnusedImport",
        "reportUnknownMemberType", "reportUnknownVariableType", "reportAny"
    ]

    print("=== Priority Error Analysis ===")
    total_errors = len(diagnostics)

    for rule in priority_rules:
        if rule_errors := error_types.get(rule):
            count = len(rule_errors)
            print(f"\n🔴 {rule}: {count} errors")

            files: defaultdict[str, int] = defaultdict(int)
            for error in rule_errors:
                files[error["file"]] += 1

            print("   File Distribution:")
            for file, cnt in sorted(files.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   - {file}: {cnt}")

            print("   Examples:")
            for error in rule_errors[:3]:
                line_display = "N/A" if error["line"] == -1 else error["line"] + 1
                message_preview = error['message'][:70]
                print(f"   - {error['file']}:{line_display} - {message_preview}...")

    print(f"\nTotal Errors Found: {total_errors}")


def main() -> None:
    """Main execution function."""
    if len(sys.argv) > 1:
        report_path = sys.argv[1]
    else:
        report_path = "pyright_report_latest.json"

    if report_data := read_report(report_path):
        if diagnostics := report_data.get("generalDiagnostics"):
            analyze_diagnostics(diagnostics)
        else:
            print("No diagnostics to analyze.")


if __name__ == "__main__":
    main() 