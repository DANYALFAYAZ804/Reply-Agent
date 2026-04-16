from src.agent import LevelReport
from src.data import LEVEL_PROFILES

DIVIDER = "=" * 70
SECTION = "-" * 70


def print_level_header(level: int) -> None:
    profile = LEVEL_PROFILES.get(level, {})
    desc = profile.get("description", f"Level {level}")
    additional = profile.get("additional_data", "")
    print(f"\n{DIVIDER}")
    print(f"  THE EYE  |  MIRRORTPAY FRAUD DETECTION  |  {desc.upper()}")
    print(f"  Additional citizen data: {additional}")
    print(DIVIDER)


def print_section(title: str, content: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)
    print(content.strip())


def print_level_report(report: LevelReport) -> None:
    print_section(f"[LEVEL {report.level}] TRANSACTION ANALYST REPORT", report.analyst_report)
    print_section(f"[LEVEL {report.level}] FRAUD DETECTOR PREDICTIONS", report.detector_report)
    print_section(f"[LEVEL {report.level}] ADAPTIVE STRATEGIST ASSESSMENT", report.strategist_report)
    print_section(f"[LEVEL {report.level}] COORDINATOR FINAL VERDICT", report.coordinator_report)
    print(f"\n  [PREDICTIONS] {len(report.predictions)} transactions classified.")
    fraud_count = sum(1 for v in report.predictions.values() if v == "fraudulent")
    legit_count = len(report.predictions) - fraud_count
    print(f"  Fraudulent: {fraud_count}  |  Legitimate: {legit_count}")


def print_session_summary(session_id: str, total_levels: int) -> None:
    print(f"\n{DIVIDER}")
    print("  THE EYE — MISSION COMPLETE")
    print(DIVIDER)
    print(f"  Levels completed  : {total_levels}")
    print(f"  Session ID        : {session_id}")
    print(f"  Langfuse traces   : {total_levels * 4} agent calls grouped under session")
    print(f"  Submissions       : saved in submissions/ (first submission per level is final)")
    print(DIVIDER)
