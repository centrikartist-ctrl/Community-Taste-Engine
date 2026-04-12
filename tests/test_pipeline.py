"""
test_pipeline.py — Unit tests for the Judgement Pipeline modules.

Uses only pytest and numpy.
"""

import sys
import os
import json
import tempfile

import numpy as np
import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Helpers ─────────────────────────────────────────────────────────────────

SR = 22050  # sample rate used throughout tests


def _sine(freq: float = 440.0, duration: float = 1.0, sr: int = SR) -> np.ndarray:
    """Generate a float32 mono sine wave."""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _click_track(bpm: float = 120.0, duration: float = 4.0, sr: int = SR) -> np.ndarray:
    """Generate a synthetic click track at the given BPM."""
    y = np.zeros(int(duration * sr), dtype=np.float32)
    period = 60.0 / bpm
    t = 0.0
    click_len = int(0.01 * sr)  # 10 ms click
    while t < duration:
        start = int(t * sr)
        end = min(start + click_len, len(y))
        y[start:end] = 1.0
        t += period
    return y


def _loud_signal(duration: float = 1.0, sr: int = SR) -> np.ndarray:
    """Loud sine wave guaranteed to trigger VAD."""
    return (_sine(200.0, duration, sr) * 0.9).astype(np.float32)


def _silence(duration: float = 1.0, sr: int = SR) -> np.ndarray:
    """Near-silent signal."""
    return np.zeros(int(duration * sr), dtype=np.float32)


# ─── stft.py ─────────────────────────────────────────────────────────────────

class TestStft:
    def test_output_shape(self):
        from stft import stft
        y = _sine()
        n_fft = 2048
        hop = 512
        S = stft(y, n_fft=n_fft, hop_length=hop)
        expected_bins = n_fft // 2 + 1
        assert S.ndim == 2
        assert S.shape[1] == expected_bins

    def test_output_nonnegative(self):
        """Magnitude spectrogram must be non-negative."""
        from stft import stft
        y = _sine()
        S = stft(y)
        assert np.all(S >= 0)

    def test_dtype_float32(self):
        from stft import stft
        y = _sine()
        S = stft(y)
        assert S.dtype == np.float32

    def test_sine_energy_concentrated(self):
        """A 440 Hz sine should concentrate energy near the 440 Hz bin."""
        from stft import stft
        n_fft = 2048
        y = _sine(440.0, duration=2.0)
        S = stft(y, n_fft=n_fft)
        mean_spectrum = S.mean(axis=0)
        peak_bin = int(np.argmax(mean_spectrum))
        # 440 Hz bin ≈ round(440 * n_fft / SR)
        expected_bin = round(440 * n_fft / SR)
        assert abs(peak_bin - expected_bin) <= 2


# ─── onset.py ────────────────────────────────────────────────────────────────

class TestOnset:
    def test_onset_strength_shape(self):
        from onset import onset_strength
        from stft import stft
        y = _sine()
        n_fft = 2048
        hop = 512
        S = stft(y, n_fft=n_fft, hop_length=hop)
        ose = onset_strength(y, SR, n_fft=n_fft, hop_length=hop)
        # OSE has one fewer frame than STFT
        assert ose.shape == (S.shape[0] - 1,)

    def test_onset_strength_range(self):
        """Onset strength envelope should be in [0, 1] after normalisation."""
        from onset import onset_strength
        y = _sine(duration=2.0)
        ose = onset_strength(y, SR)
        assert ose.min() >= 0.0
        assert ose.max() <= 1.0 + 1e-6

    def test_onset_strength_dtype(self):
        from onset import onset_strength
        y = _sine()
        ose = onset_strength(y, SR)
        assert ose.dtype == np.float32


# ─── vad.py ──────────────────────────────────────────────────────────────────

class TestVad:
    def test_loud_signal_has_speech(self):
        """A loud signal should produce at least one speech region."""
        from vad import detect_speech
        y = _loud_signal(duration=1.0)
        regions = detect_speech(y, SR)
        assert len(regions) >= 1

    def test_silence_no_speech(self):
        """Pure silence should produce no speech regions."""
        from vad import detect_speech
        y = _silence(duration=1.0)
        regions = detect_speech(y, SR)
        assert len(regions) == 0

    def test_regions_are_ordered(self):
        """Speech regions must be (start, end) with start < end."""
        from vad import detect_speech
        y = _loud_signal(duration=2.0)
        regions = detect_speech(y, SR)
        for start, end in regions:
            assert start < end

    def test_regions_within_bounds(self):
        from vad import detect_speech
        duration = 2.0
        y = _loud_signal(duration=duration)
        regions = detect_speech(y, SR)
        for start, end in regions:
            assert start >= 0.0
            assert end <= duration + 0.1  # small tolerance for rounding


# ─── beat_tracker.py ─────────────────────────────────────────────────────────

class TestBeatTracker:
    def test_returns_two_values(self):
        from beat_tracker import track_beats
        y = _click_track(bpm=120.0, duration=4.0)
        result = track_beats(y, SR)
        assert len(result) == 2

    def test_bpm_approximately_correct(self):
        """BPM estimate for a 120 BPM click track should be within 20%."""
        from beat_tracker import track_beats
        y = _click_track(bpm=120.0, duration=8.0)
        beat_times, bpm = track_beats(y, SR)
        assert 96.0 <= bpm <= 144.0, f"BPM {bpm:.1f} out of expected range"

    def test_beat_times_nonempty(self):
        from beat_tracker import track_beats
        y = _click_track(bpm=120.0, duration=4.0)
        beat_times, _ = track_beats(y, SR)
        assert len(beat_times) > 0

    def test_beat_times_sorted(self):
        from beat_tracker import track_beats
        y = _click_track(bpm=120.0, duration=4.0)
        beat_times, _ = track_beats(y, SR)
        assert np.all(np.diff(beat_times) >= 0)


# ─── aligner.py ──────────────────────────────────────────────────────────────

class TestAligner:
    def test_returns_nonempty_list(self):
        from aligner import build_chunks
        y = _loud_signal(duration=2.0)
        chunks = build_chunks(y, SR)
        assert len(chunks) > 0

    def test_chunks_cover_audio(self):
        """First chunk starts at 0.0; last chunk ends near signal duration."""
        from aligner import build_chunks
        duration = 2.0
        y = _loud_signal(duration=duration)
        chunks = build_chunks(y, SR)
        assert chunks[0].start == pytest.approx(0.0, abs=0.01)
        assert chunks[-1].end == pytest.approx(duration, abs=0.1)

    def test_chunks_ordered_and_non_overlapping(self):
        from aligner import build_chunks
        y = _loud_signal(duration=2.0)
        chunks = build_chunks(y, SR)
        for i in range(1, len(chunks)):
            assert chunks[i].start >= chunks[i - 1].end - 1e-6

    def test_chunk_fields(self):
        from aligner import build_chunks, Chunk
        y = _loud_signal(duration=1.0)
        chunks = build_chunks(y, SR)
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.start < c.end
            assert c.duration > 0
            assert c.mean_energy >= 0
            assert c.boundary_type in ("syllable", "word_gap", "sentence_end", "silence")


# ─── pipeline.py ─────────────────────────────────────────────────────────────

class TestPlanner:
    def _make_chunks(self):
        """Return two minimal chunks usable by the planner."""
        from aligner import Chunk
        return [
            Chunk(start=0.0, end=0.5, duration=0.5, mean_energy=0.01,
                  peak_energy=0.02, is_speech=False, boundary_type="silence"),
            Chunk(start=0.5, end=1.0, duration=0.5, mean_energy=0.08,
                  peak_energy=0.15, is_speech=True, boundary_type="sentence_end"),
            Chunk(start=1.0, end=1.5, duration=0.5, mean_energy=0.02,
                  peak_energy=0.03, is_speech=False, boundary_type="silence"),
        ]

    def test_produces_decisions(self):
        from pipeline import Planner
        planner = Planner(confidence_threshold=0.1)
        chunks = self._make_chunks()
        beat_times = np.array([0.5, 1.0, 1.5])
        decisions = planner.plan(chunks, beat_times, 120.0, source="test.mp4")
        assert isinstance(decisions, list)

    def test_decision_fields(self):
        from pipeline import Planner, CutDecision
        planner = Planner(confidence_threshold=0.1)
        chunks = self._make_chunks()
        beat_times = np.array([0.5, 1.0])
        decisions = planner.plan(chunks, beat_times, 120.0, source="test.mp4")
        for d in decisions:
            assert isinstance(d, CutDecision)
            assert 0.0 <= d.confidence <= 1.0
            assert len(d.cut_id) == 10

    def test_cut_id_includes_source(self):
        """Same timestamp but different source should give different cut_id."""
        from pipeline import Planner
        planner = Planner(confidence_threshold=0.1)
        chunks = self._make_chunks()
        beat_times = np.array([0.5])
        d1 = planner.plan(chunks, beat_times, 120.0, source="file_a.mp4")
        d2 = planner.plan(chunks, beat_times, 120.0, source="file_b.mp4")
        ids1 = {d.cut_id for d in d1}
        ids2 = {d.cut_id for d in d2}
        # At least one ID should differ between the two sources
        assert ids1 != ids2 or (len(ids1) == 0 and len(ids2) == 0)


class TestCritic:
    def _make_chunks(self):
        from aligner import Chunk
        return [
            Chunk(start=0.0, end=0.5, duration=0.5, mean_energy=0.01,
                  peak_energy=0.02, is_speech=False, boundary_type="silence"),
            Chunk(start=0.5, end=1.0, duration=0.5, mean_energy=0.05,
                  peak_energy=0.08, is_speech=True, boundary_type="sentence_end"),
        ]

    def test_scores_decision(self):
        from pipeline import Critic, CutDecision, CriticScore
        critic = Critic()
        chunks = self._make_chunks()
        decision = CutDecision(
            cut_id="abc1234567",
            timestamp=0.5,
            reason="sentence_end",
            confidence=0.8,
            chunk_idx=1,
            features={},
        )
        beat_times = np.array([0.5, 1.0])
        score = critic.score(decision, chunks, beat_times, video_path=None)
        assert isinstance(score, CriticScore)
        assert 0.0 <= score.final_score <= 1.0
        assert 0.0 <= score.rhythm_score <= 1.0
        assert 0.0 <= score.speech_score <= 1.0
        assert 0.0 <= score.energy_score <= 1.0
        assert score.pairing_score == -1.0  # no video path


class TestLogger:
    def test_write_and_load_roundtrip(self):
        from pipeline import Logger, CutDecision, CriticScore
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            logger = Logger(path)
            decision = CutDecision(
                cut_id="test000001",
                timestamp=1.23,
                reason="sentence_end",
                confidence=0.75,
                chunk_idx=1,
                features={"bpm": 120.0},
            )
            score = CriticScore(
                cut_id="test000001",
                rhythm_score=0.9,
                speech_score=1.0,
                energy_score=0.8,
                pairing_score=-1.0,
                final_score=0.92,
            )
            logger.write("video.mp4", decision, score)
            records = logger.load()
            assert len(records) == 1
            rec = records[0]
            assert rec["source"] == "video.mp4"
            assert rec["decision"]["cut_id"] == "test000001"
            assert rec["score"]["final_score"] == 0.92
        finally:
            os.unlink(path)

    def test_rule_performance_strips_parens(self):
        """rule_performance() should strip parametric suffixes before bucketing."""
        from pipeline import Logger, CutDecision, CriticScore
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            logger = Logger(path)
            for ms in ["45ms", "80ms"]:
                decision = CutDecision(
                    cut_id=f"id_{ms}",
                    timestamp=1.0,
                    reason=f"beat_aligned({ms})",
                    confidence=0.6,
                    chunk_idx=1,
                    features={},
                )
                score = CriticScore(
                    cut_id=f"id_{ms}",
                    rhythm_score=0.8,
                    speech_score=0.9,
                    energy_score=0.7,
                    pairing_score=-1.0,
                    final_score=0.82,
                )
                logger.write("video.mp4", decision, score)

            perf = logger.rule_performance()
            # Both entries should be bucketed under "beat_aligned" (not "beat_aligned(45ms)")
            assert "beat_aligned" in perf
            assert perf["beat_aligned"]["count"] == 2
            assert "beat_aligned(45ms)" not in perf
            assert "beat_aligned(80ms)" not in perf
        finally:
            os.unlink(path)

    def test_empty_log_returns_empty(self):
        from pipeline import Logger
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            logger = Logger(path)
            assert logger.load() == []
            assert logger.rule_performance() == {}
        finally:
            os.unlink(path)


class TestWeightPersistence:
    def test_save_and_load_weights(self):
        from pipeline import Planner
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            planner = Planner()
            planner.rule_weights = {"sentence_end": 1.3, "beat_aligned": 0.8}
            planner.save_weights(path)

            planner2 = Planner()
            planner2.load_weights(path)
            assert planner2.rule_weights["sentence_end"] == pytest.approx(1.3)
            assert planner2.rule_weights["beat_aligned"] == pytest.approx(0.8)
        finally:
            os.unlink(path)

    def test_load_nonexistent_weights_is_noop(self):
        """Loading from a missing file should not raise and leave weights unchanged."""
        from pipeline import Planner
        planner = Planner()
        planner.load_weights("/tmp/nonexistent_weights_xyz.json")
        assert planner.rule_weights == {}


class TestEdgeCases:
    def test_short_audio_build_chunks_graceful(self):
        from aligner import build_chunks
        y = np.zeros(128, dtype=np.float32)
        chunks = build_chunks(y, SR)
        assert isinstance(chunks, list)

    def test_short_audio_vad_returns_empty(self):
        from vad import detect_speech
        y = np.zeros(128, dtype=np.float32)
        regions = detect_speech(y, SR)
        assert regions == []

    def test_short_audio_beat_tracker_graceful(self):
        from beat_tracker import track_beats
        y = np.zeros(128, dtype=np.float32)
        beat_times, bpm = track_beats(y, SR)
        assert isinstance(beat_times, np.ndarray)
        assert bpm >= 0.0

    def test_run_rejects_invalid_confidence(self):
        from pipeline import run
        with pytest.raises(ValueError):
            run("whatever.wav", min_confidence=2.0)


class TestCriticCaching:
    def test_score_uses_audio_cache_without_reload(self, monkeypatch):
        from pipeline import Critic, CutDecision
        from aligner import Chunk

        def _boom(_):
            raise AssertionError("load_audio should not be called when audio_cache is provided")

        monkeypatch.setattr("pipeline.load_audio", _boom)

        monkeypatch.setattr("pipeline.audio_embedding", lambda *_args, **_kwargs: np.ones(32, dtype=np.float32))
        monkeypatch.setattr("pipeline.visual_embedding", lambda *_args, **_kwargs: np.ones(32, dtype=np.float32))

        critic = Critic()
        chunks = [
            Chunk(start=0.0, end=0.5, duration=0.5, mean_energy=0.01,
                  peak_energy=0.02, is_speech=False, boundary_type="silence"),
            Chunk(start=0.5, end=1.0, duration=0.5, mean_energy=0.05,
                  peak_energy=0.08, is_speech=True, boundary_type="sentence_end"),
        ]
        decision = CutDecision(
            cut_id="cache00001",
            timestamp=0.5,
            reason="sentence_end",
            confidence=0.8,
            chunk_idx=1,
            features={},
        )

        y = np.zeros(SR * 2, dtype=np.float32)
        score = critic.score(
            decision,
            chunks,
            beat_times=np.array([0.5]),
            video_path="fake.mp4",
            audio_cache=(y, SR),
        )
        assert 0.0 <= score.final_score <= 1.0

    def test_score_raises_when_visual_embedding_fails(self, monkeypatch):
        from pipeline import Critic, CutDecision
        from aligner import Chunk

        monkeypatch.setattr("pipeline.audio_embedding", lambda *_args, **_kwargs: np.ones(32, dtype=np.float32))
        monkeypatch.setattr("pipeline.visual_embedding", lambda *_args, **_kwargs: None)

        critic = Critic()
        chunks = [
            Chunk(start=0.0, end=0.5, duration=0.5, mean_energy=0.01,
                  peak_energy=0.02, is_speech=False, boundary_type="silence"),
            Chunk(start=0.5, end=1.0, duration=0.5, mean_energy=0.05,
                  peak_energy=0.08, is_speech=True, boundary_type="sentence_end"),
        ]
        decision = CutDecision(
            cut_id="cache00002",
            timestamp=0.5,
            reason="sentence_end",
            confidence=0.8,
            chunk_idx=1,
            features={},
        )

        with pytest.raises(RuntimeError, match="visual embedding failed"):
            critic.score(
                decision,
                chunks,
                beat_times=np.array([0.5]),
                video_path="fake.mp4",
                audio_cache=(np.zeros(SR * 2, dtype=np.float32), SR),
            )


class TestExecutor:
    def test_execute_cut_uses_capcut_compose(self, monkeypatch):
        from pipeline import execute_cut, CutDecision
        from capcut_automation import ComposeResult

        calls = []

        def _fake_compose(self, request, timeout=120):
            calls.append((request, timeout))
            os.makedirs(request.output_dir, exist_ok=True)
            with open(os.path.join(request.output_dir, "render.mp4"), "wb") as f:
                f.write(b"00")
            return ComposeResult(
                command=["capcut-cli", "compose"],
                cwd=request.output_dir,
                duration_seconds=request.duration_seconds,
                sound_id=request.sound_id,
                clip_ids=request.clip_ids,
                started_at=0.0,
                finished_at=0.1,
                elapsed_seconds=0.1,
            )

        monkeypatch.setattr("capcut_automation.CapCutAutomation.compose", _fake_compose)

        decision = CutDecision(
            cut_id="exec000001",
            timestamp=1.25,
            reason="sentence_end",
            confidence=0.9,
            chunk_idx=1,
            features={},
        )

        with tempfile.TemporaryDirectory() as out_dir:
            out = execute_cut(
                "input.mp4",
                decision,
                output_dir=out_dir,
                sound_id="sound_123",
                clip_ids=["clip_a", "clip_b"],
                duration_seconds=24,
            )
            assert out.endswith("render.mp4")
            assert calls, "expected compose call"
            req = calls[0][0]
            assert req.sound_id == "sound_123"
            assert req.clip_ids == ["clip_a", "clip_b"]
            assert req.duration_seconds == 24

    def test_execute_cut_requires_capcut_cli(self, monkeypatch):
        from pipeline import execute_cut, CutDecision
        from capcut_automation import CapCutDependencyError

        def _raise_dep(*_args, **_kwargs):
            raise CapCutDependencyError("capcut-cli not found in PATH")

        monkeypatch.setattr("capcut_automation.CapCutAutomation.compose", _raise_dep)

        decision = CutDecision(
            cut_id="exec000002",
            timestamp=1.25,
            reason="sentence_end",
            confidence=0.9,
            chunk_idx=1,
            features={},
        )

        with pytest.raises(CapCutDependencyError):
            execute_cut("input.mp4", decision, output_dir="out", sound_id="sound_1", clip_ids=["clip_1"])

    def test_execute_cut_requires_ids(self, monkeypatch):
        from pipeline import execute_cut, CutDecision
        from capcut_automation import CapCutInputError

        decision = CutDecision(
            cut_id="exec000003",
            timestamp=1.25,
            reason="sentence_end",
            confidence=0.9,
            chunk_idx=1,
            features={},
        )

        with pytest.raises(CapCutInputError):
            execute_cut("input.mp4", decision, output_dir="out")

    def test_execute_cut_raises_when_no_output_created(self, monkeypatch):
        from pipeline import execute_cut, CutDecision
        from capcut_automation import ComposeResult

        def _fake_compose(self, request, timeout=120):
            os.makedirs(request.output_dir, exist_ok=True)
            return ComposeResult(
                command=["capcut-cli", "compose"],
                cwd=request.output_dir,
                duration_seconds=request.duration_seconds,
                sound_id=request.sound_id,
                clip_ids=request.clip_ids,
                started_at=0.0,
                finished_at=0.1,
                elapsed_seconds=0.1,
            )

        monkeypatch.setattr("capcut_automation.CapCutAutomation.compose", _fake_compose)

        decision = CutDecision(
            cut_id="exec000004",
            timestamp=1.25,
            reason="sentence_end",
            confidence=0.9,
            chunk_idx=1,
            features={},
        )

        with tempfile.TemporaryDirectory() as out_dir:
            with pytest.raises(RuntimeError, match="no media artifacts"):
                execute_cut(
                    "input.mp4",
                    decision,
                    output_dir=out_dir,
                    sound_id="sound_123",
                    clip_ids=["clip_a"],
                    duration_seconds=24,
                )
