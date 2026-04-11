"""
pipeline.py — judgement pipeline orchestrator
----------------------------------------------
Planner  → propose cuts with explicit reasoning + confidence
Critic   → score each cut on three axes
Logger   → persist every decision + score as JSONL
Feedback → aggregate scores by rule → tells planner what's working

This is the learning loop. Every run makes the next run smarter.
"""

import json
import time
import hashlib
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from audio import load as load_audio
from beat_tracker import track_beats
from aligner import build_chunks, Chunk
from embedder import audio_embedding, visual_embedding, pairing_score


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class CutDecision:
    cut_id: str
    timestamp: float          # seconds — where to make the cut
    reason: str               # human-readable rule that fired
    confidence: float         # planner's self-assessed confidence [0, 1]
    chunk_idx: int            # index into chunk list
    features: dict            # raw feature values that drove the decision


@dataclass
class CriticScore:
    cut_id: str
    rhythm_score: float       # how close is cut to nearest beat [0, 1]
    speech_score: float       # penalise mid-sentence cuts [0, 1]
    energy_score: float       # energy coherence across cut [0, 1]
    pairing_score: float      # audio-visual compatibility [0, 1]; -1 if no video
    final_score: float        # weighted composite


# ══════════════════════════════════════════════════════════════════
# PLANNER
# ══════════════════════════════════════════════════════════════════

class Planner:
    """
    Rule-based cut planner. Rules are explicit and independently scoreable
    so the feedback loop can identify which rules produce good cuts.

    Rules are applied in priority order. Each rule returns a confidence
    increment; they accumulate. Cut is proposed if total confidence > threshold.
    """

    def __init__(self, confidence_threshold: float = 0.35):
        self.threshold = confidence_threshold
        # Will be updated by feedback loop
        self.rule_weights: dict[str, float] = {}

    def plan(
        self,
        chunks: list[Chunk],
        beat_times: np.ndarray,
        bpm: float,
    ) -> list[CutDecision]:
        decisions = []

        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]

            reasons = []
            confidence = 0.0
            features = {
                "prev_energy": prev.mean_energy,
                "curr_energy": curr.mean_energy,
                "prev_boundary": prev.boundary_type,
                "curr_boundary": curr.boundary_type,
                "prev_is_speech": prev.is_speech,
                "curr_is_speech": curr.is_speech,
                "bpm": bpm,
            }

            # ── Rule 1: Sentence boundary ────────────────────────────────
            # Best possible cut point: after a sentence ends.
            if prev.boundary_type == "sentence_end":
                confidence += self._weighted("sentence_end", 0.6)
                reasons.append("sentence_end")

            # ── Rule 2: Beat-aligned boundary ───────────────────────────
            # Cut falls within 80ms of a beat.
            if len(beat_times):
                nearest_beat = float(np.min(np.abs(beat_times - curr.start)))
                features["nearest_beat_dist"] = nearest_beat
                beat_window = 0.08  # 80ms tolerance
                if nearest_beat <= beat_window:
                    confidence += self._weighted("beat_aligned", 0.5)
                    reasons.append(f"beat_aligned({nearest_beat*1000:.0f}ms)")

            # ── Rule 3: Energy spike ─────────────────────────────────────
            # Current chunk starts with significantly more energy.
            energy_delta = curr.mean_energy - prev.mean_energy
            features["energy_delta"] = energy_delta
            if energy_delta > 0.03:
                confidence += self._weighted("energy_spike", 0.3)
                reasons.append(f"energy_spike(+{energy_delta:.3f})")

            # ── Rule 4: Energy drop (scene change) ───────────────────────
            if energy_delta < -0.04:
                confidence += self._weighted("energy_drop", 0.25)
                reasons.append(f"energy_drop({energy_delta:.3f})")

            # ── Rule 5: Silence boundary ─────────────────────────────────
            # Gap of silence is a natural edit point.
            if not curr.is_speech and not prev.is_speech:
                if curr.duration > 0.15:
                    confidence += self._weighted("silence_gap", 0.2)
                    reasons.append(f"silence_gap({curr.duration:.2f}s)")

            # ── Rule 6: Penalise mid-word cuts ───────────────────────────
            if prev.is_speech and prev.boundary_type == "syllable":
                confidence -= 0.25
                reasons.append("mid_syllable_penalty")

            confidence = max(0.0, min(1.0, confidence))

            if confidence >= self.threshold:
                cut_id = hashlib.md5(
                    f"{curr.start:.4f}{curr.boundary_type}".encode()
                ).hexdigest()[:10]

                decisions.append(CutDecision(
                    cut_id=cut_id,
                    timestamp=curr.start,
                    reason="|".join(r for r in reasons if "penalty" not in r),
                    confidence=confidence,
                    chunk_idx=i,
                    features=features,
                ))

        return decisions

    def _weighted(self, rule: str, base: float) -> float:
        """Apply learned weight multiplier if available."""
        w = self.rule_weights.get(rule, 1.0)
        return base * w

    def update_weights(self, feedback: dict[str, float]):
        """
        Update rule weights from critic feedback.
        feedback: {rule_name: avg_score} from Logger.rule_performance()

        Rules scoring above 0.7 get a boost.
        Rules scoring below 0.4 get penalised.
        """
        for rule, avg_score in feedback.items():
            if avg_score > 0.7:
                self.rule_weights[rule] = min(1.5, self.rule_weights.get(rule, 1.0) * 1.1)
            elif avg_score < 0.4:
                self.rule_weights[rule] = max(0.3, self.rule_weights.get(rule, 1.0) * 0.9)


# ══════════════════════════════════════════════════════════════════
# CRITIC
# ══════════════════════════════════════════════════════════════════

class Critic:
    """
    Scores cut decisions on three independent axes.
    Each axis is designed to be improvable independently.
    """

    def score(
        self,
        decision: CutDecision,
        chunks: list[Chunk],
        beat_times: np.ndarray,
        video_path: Optional[str] = None,
    ) -> CriticScore:

        chunk = chunks[decision.chunk_idx]
        prev_chunk = chunks[decision.chunk_idx - 1]

        # ── Rhythm score ─────────────────────────────────────────────────
        if len(beat_times):
            dist = float(np.min(np.abs(beat_times - decision.timestamp)))
            # Linear falloff: 0ms → 1.0, 250ms → 0.0
            rhythm = max(0.0, 1.0 - dist / 0.25)
        else:
            rhythm = 0.5

        # ── Speech integrity score ───────────────────────────────────────
        if not prev_chunk.is_speech:
            speech = 1.0  # not cutting speech at all — fine
        elif prev_chunk.boundary_type == "sentence_end":
            speech = 1.0
        elif prev_chunk.boundary_type == "word_gap":
            speech = 0.65
        elif prev_chunk.boundary_type == "syllable":
            speech = 0.2  # mid-syllable cut — bad
        else:
            speech = 0.8

        # ── Energy coherence score ───────────────────────────────────────
        # Penalise jarring energy changes that aren't beat-aligned
        e_delta = abs(chunk.mean_energy - prev_chunk.mean_energy)
        if rhythm > 0.7:
            # Beat-aligned cuts can handle energy jumps
            energy = max(0.0, 1.0 - e_delta * 5.0)
        else:
            energy = max(0.0, 1.0 - e_delta * 12.0)

        # ── Pairing score (audio-visual) ─────────────────────────────────
        pair = -1.0
        if video_path:
            try:
                y_seg, sr = load_audio(video_path)
                s = int(decision.timestamp * sr)
                e = min(int((decision.timestamp + 2.0) * sr), len(y_seg))
                a_emb = audio_embedding(y_seg[s:e], sr)
                v_emb = visual_embedding(video_path, decision.timestamp, decision.timestamp + 2.0)
                if v_emb is not None:
                    pair = pairing_score(a_emb, v_emb)
            except Exception:
                pair = -1.0

        # ── Final composite ──────────────────────────────────────────────
        if pair >= 0:
            final = rhythm * 0.35 + speech * 0.35 + energy * 0.2 + pair * 0.1
        else:
            final = rhythm * 0.40 + speech * 0.40 + energy * 0.20

        return CriticScore(
            cut_id=decision.cut_id,
            rhythm_score=round(rhythm, 4),
            speech_score=round(speech, 4),
            energy_score=round(energy, 4),
            pairing_score=round(pair, 4),
            final_score=round(final, 4),
        )


# ══════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════

class Logger:
    """
    Persistent JSONL decision log.

    Every run appends records. Over time this builds a dataset
    the planner can learn from.
    """

    def __init__(self, path: str = "decisions.jsonl"):
        self.path = Path(path)

    def write(
        self,
        source: str,
        decision: CutDecision,
        score: Optional[CriticScore],
        accepted: Optional[bool] = None,  # human override if available
    ):
        record = {
            "ts": round(time.time(), 3),
            "source": source,
            "decision": asdict(decision),
            "score": asdict(score) if score else None,
            "accepted": accepted,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def rule_performance(self) -> dict[str, dict]:
        """
        Aggregate final scores by individual rule.

        Returns {rule: {"avg": float, "count": int, "min": float, "max": float}}
        """
        from collections import defaultdict
        buckets: dict[str, list[float]] = defaultdict(list)

        for rec in self.load():
            if not rec.get("score"):
                continue
            final = rec["score"]["final_score"]
            reason = rec["decision"].get("reason", "")
            for rule in reason.split("|"):
                rule = rule.strip()
                if rule:
                    buckets[rule].append(final)

        result = {}
        for rule, scores in buckets.items():
            result[rule] = {
                "avg": round(sum(scores) / len(scores), 4),
                "count": len(scores),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
            }
        return result

    def worst_cuts(self, n: int = 10) -> list[dict]:
        """Return n lowest-scoring cuts for review / retraining signal."""
        records = [r for r in self.load() if r.get("score")]
        records.sort(key=lambda r: r["score"]["final_score"])
        return records[:n]

    def best_cuts(self, n: int = 10) -> list[dict]:
        """Return n highest-scoring cuts as positive examples."""
        records = [r for r in self.load() if r.get("score")]
        records.sort(key=lambda r: r["score"]["final_score"], reverse=True)
        return records[:n]


# ══════════════════════════════════════════════════════════════════
# EXECUTOR (stub — replace with actual capcut-cli calls)
# ══════════════════════════════════════════════════════════════════

def execute_cut(video_path: str, decision: CutDecision, output_dir: str = ".") -> str:
    """
    Call capcut-cli to make the cut.
    Replace the print/stub with actual subprocess call.
    """
    out_path = f"{output_dir}/cut_{decision.cut_id}.mp4"
    cmd = [
        "capcut-cli", "cut",
        "--input", video_path,
        "--at", f"{decision.timestamp:.4f}",
        "--output", out_path,
    ]
    print(f"[executor] {' '.join(cmd)}")
    # subprocess.run(cmd, check=True)
    return out_path


# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def run(
    video_path: str,
    log_path: str = "decisions.jsonl",
    output_dir: str = ".",
    min_confidence: float = 0.35,
    dry_run: bool = True,
) -> list[CutDecision]:
    """
    Full pipeline: ingest → plan → execute → critique → log.

    Parameters
    ----------
    video_path     : path to input video (any ffmpeg-supported format)
    log_path       : where to persist decisions
    output_dir     : where to write cut clips
    min_confidence : planner threshold
    dry_run        : if True, skip actual capcut-cli calls

    Returns
    -------
    List of CutDecisions made this run.
    """
    logger = Logger(log_path)
    planner = Planner(confidence_threshold=min_confidence)
    critic = Critic()

    # ── Load feedback from previous runs and update planner weights ──────
    previous_perf = logger.rule_performance()
    if previous_perf:
        avg_scores = {rule: data["avg"] for rule, data in previous_perf.items()}
        planner.update_weights(avg_scores)
        print(f"[planner] loaded weights from {sum(d['count'] for d in previous_perf.values())} past decisions")

    # ── Stage 1: Load audio ──────────────────────────────────────────────
    print("[1/4] loading audio...")
    y, sr = load_audio(video_path)
    print(f"      {len(y)/sr:.1f}s @ {sr}Hz")

    # ── Stage 2: Build semantic index ────────────────────────────────────
    print("[2/4] building semantic index...")
    chunks = build_chunks(y, sr)
    beat_times, bpm = track_beats(y, sr)
    print(f"      {len(chunks)} chunks, {len(beat_times)} beats, {bpm:.1f} BPM")

    # ── Stage 3: Plan cuts ───────────────────────────────────────────────
    print("[3/4] planning cuts...")
    decisions = planner.plan(chunks, beat_times, bpm)
    print(f"      {len(decisions)} candidates (threshold={min_confidence})")

    # ── Stage 4: Execute + critique + log ────────────────────────────────
    print("[4/4] executing, critiquing, logging...")
    for d in decisions:
        if not dry_run:
            execute_cut(video_path, d, output_dir)

        score = critic.score(d, chunks, beat_times, video_path=video_path)
        logger.write(video_path, d, score)

        flag = "✓" if score.final_score >= 0.65 else "~" if score.final_score >= 0.45 else "✗"
        print(
            f"  {flag} {d.timestamp:7.3f}s  [{score.final_score:.2f}]  "
            f"r={score.rhythm_score:.2f} s={score.speech_score:.2f} "
            f"e={score.energy_score:.2f}  {d.reason}"
        )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n[rule performance across all runs]")
    for rule, stats in sorted(
        logger.rule_performance().items(),
        key=lambda x: x[1]["avg"],
        reverse=True,
    ):
        bar = "█" * int(stats["avg"] * 20)
        print(f"  {stats['avg']:.3f} {bar:<20} {rule} (n={stats['count']})")

    return decisions


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if not args:
        print("usage: python pipeline.py <video> [--log <path>] [--live]")
        sys.exit(1)

    video = args[0]
    log = "decisions.jsonl"
    dry = True

    i = 1
    while i < len(args):
        if args[i] == "--log" and i + 1 < len(args):
            log = args[i + 1]
            i += 2
        elif args[i] == "--live":
            dry = False
            i += 1
        else:
            i += 1

    run(video, log_path=log, dry_run=dry)
