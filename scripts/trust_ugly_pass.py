"""Run one reproducible ugly-success trust pass for the repository.

Flow:
1) Generate a synthetic source clip.
2) Run judgement pipeline in dry-run mode (includes pairing score path).
3) Validate at least one scored decision with non-negative pairing score.
4) Emit machine-readable and human-readable reports under trust/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import Logger, run


def _write_structured_wav(dst: Path, sr: int = 22050) -> float:
    # Alternating tones and short silences create internal boundaries.
    parts: list[np.ndarray] = []
    for freq in (220.0, 330.0, 440.0, 550.0):
        t = np.linspace(0.0, 1.0, int(sr * 1.0), endpoint=False)
        tone = 0.45 * np.sin(2 * np.pi * freq * t)
        parts.append(tone.astype(np.float32))
        parts.append(np.zeros(int(sr * 0.35), dtype=np.float32))

    # Trailing silence keeps final scored decisions away from stream end.
    parts.append(np.zeros(int(sr * 2.0), dtype=np.float32))

    y = np.concatenate(parts)
    pcm16 = np.clip(y, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)

    with wave.open(str(dst), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

    return len(y) / sr


def _generate_source_clip(dst: Path) -> float:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for trust pass generation")

    wav_path = dst.with_suffix(".wav")
    duration = _write_structured_wav(wav_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=640x360:rate=24",
        "-i",
        str(wav_path),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=90)
    return duration


def run_ugly_success_pass(report_dir: Path, min_confidence: float = 0.35) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="judgement_trust_pass_") as tmp:
        tmp_dir = Path(tmp)
        source_clip = tmp_dir / "source_clip.mp4"
        log_path = tmp_dir / "trust_decisions.jsonl"

        duration_seconds = _generate_source_clip(source_clip)

        decisions = run(
            str(source_clip),
            log_path=str(log_path),
            min_confidence=min_confidence,
        )

        records = Logger(str(log_path)).load()
        scored = [rec for rec in records if rec.get("score") and rec.get("decision")]
        if not scored:
            raise RuntimeError("trust pass failed: pipeline produced no scored decisions")

        pairing_values = [float(rec["score"]["pairing_score"]) for rec in scored]
        if any(v < 0.0 for v in pairing_values):
            raise RuntimeError("trust pass failed: pairing score path was not successfully evaluated")

        final_scores = [float(rec["score"]["final_score"]) for rec in scored]
        avg_final = sum(final_scores) / len(final_scores)

        report = {
            "ok": True,
            "generated_at_unix": int(time.time()),
            "source_clip": {
                "kind": "synthetic_ffmpeg_clip",
                "duration_seconds": round(duration_seconds, 3),
                "resolution": "640x360",
                "audio_hz": 22050,
            },
            "pipeline": {
                "dry_run": True,
                "min_confidence": min_confidence,
                "decisions_count": len(decisions),
                "scored_count": len(scored),
                "pairing_min": min(pairing_values),
                "pairing_max": max(pairing_values),
                "final_score_avg": round(avg_final, 4),
            },
            "artifacts": {
                "decision_log": "generated during run in a temp directory",
                "report_json": "trust/ugly_success_report.json",
                "report_md": "trust/ugly_success_report.md",
            },
        }

    json_path = report_dir / "ugly_success_report.json"
    md_path = report_dir / "ugly_success_report.md"

    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(
        "# Ugly Successful Pass\n\n"
        "This is a reproducible trust pass proving one real flow:\n"
        "source clip -> judged pairing -> output report.\n\n"
        "## Result\n\n"
        f"- status: {'PASS' if report['ok'] else 'FAIL'}\n"
        f"- scored decisions: {report['pipeline']['scored_count']}\n"
        f"- pairing min/max: {report['pipeline']['pairing_min']:.4f} / {report['pipeline']['pairing_max']:.4f}\n"
        f"- average final score: {report['pipeline']['final_score_avg']:.4f}\n\n"
        "## Re-run\n\n"
        "```bash\n"
        "python scripts/trust_ugly_pass.py\n"
        "```\n",
        encoding="utf-8",
    )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one ugly successful trust pass")
    parser.add_argument("--report-dir", default="trust", help="Directory where trust report files are written")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="Pipeline confidence threshold")
    args = parser.parse_args()

    report = run_ugly_success_pass(Path(args.report_dir), min_confidence=args.min_confidence)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
