from src.agent import LevelReport

DIVIDER = "=" * 70
SECTION = "-" * 70


def print_level_header(level: int) -> None:
    print(f"\n{DIVIDER}")
    print(f"  THE EYE  |  MIRRORTPAY FRAUD DETECTION  |  LEVEL {level}")
    print(DIVIDER)


def print_section(title: str, content: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)
    print(content.strip())


def print_level_report(report: LevelReport) -> None:
    print_section(f"[LEVEL {report.level}] ANALYST — FRAUD PATTERN REPORT", report.analyst_report)
    print_section(f"[LEVEL {report.level}] DETECTOR — FLAGGED TRANSACTIONS", report.detector_report)
    print_section(f"[LEVEL {report.level}] STRATEGIST — ADAPTIVE INTELLIGENCE", report.strategist_report)
    print_section(f"[LEVEL {report.level}] COORDINATOR — FINAL VERDICT", report.coordinator_report)
    print(f"\n  [RESULT] {len(report.suspected_ids)} transactions flagged as suspected fraud.")
    if report.suspected_ids:
        preview = report.suspected_ids[:5]
        more = len(report.suspected_ids) - 5
        print(f"  Sample IDs: {', '.join(preview)}" + (f" ... +{more} more" if more > 0 else ""))


def print_session_summary(session_id: str, total_levels: int) -> None:
    print(f"\n{DIVIDER}")
    print("  THE EYE — MISSION COMPLETE")
    print(DIVIDER)
    print(f"  Levels processed  : {total_levels}")
    print(f"  Session ID        : {session_id}")
    print(f"  Langfuse traces   : {total_levels * 4} agent calls grouped under session")
    print(f"  Output files      : submissions/level_N_output.txt (ASCII, one TXN ID per line)")
    print(f"  Scoring note      : FP cost = 1x  |  FN cost = 3x  (asymmetric)")
    print(DIVIDER)
