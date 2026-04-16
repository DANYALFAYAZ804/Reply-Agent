from src.agent import WaveReport
from src.data import WAVE_PROFILES


DIVIDER = "=" * 70
SECTION = "-" * 70


def print_wave_header(wave: int) -> None:
    profile = WAVE_PROFILES.get(wave, {})
    desc = profile.get("description", f"Wave {wave}")
    print(f"\n{DIVIDER}")
    print(f"  THE EYE  |  MIRRORTPAY FRAUD DETECTION  |  {desc.upper()}")
    print(DIVIDER)


def print_section(title: str, content: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)
    print(content.strip())


def print_wave_report(report: WaveReport) -> None:
    print_section(f"[WAVE {report.wave}] TRANSACTION ANALYST REPORT", report.analyst_report)
    print_section(f"[WAVE {report.wave}] FRAUD DETECTOR SCORING", report.detector_report)
    print_section(f"[WAVE {report.wave}] ADAPTIVE STRATEGIST ASSESSMENT", report.strategist_report)
    print_section(f"[WAVE {report.wave}] COORDINATOR FINAL VERDICT", report.coordinator_report)


def print_session_summary(session_id: str, total_waves: int) -> None:
    print(f"\n{DIVIDER}")
    print("  THE EYE — MISSION COMPLETE")
    print(DIVIDER)
    print(f"  Waves analysed  : {total_waves}")
    print(f"  Session ID      : {session_id}")
    print(f"  Langfuse trace  : all {total_waves * 4} agent calls grouped under session")
    print(DIVIDER)
