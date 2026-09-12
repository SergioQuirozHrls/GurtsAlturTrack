"""Behavioral timing features derived from ch0 (caller) vs ch1 (agent) turns.

Primary signal (CLAUDE.md section 2): humans recover from interruptions,
silence, and overlap messily and inconsistently; scripted TTS+LLM pipelines
recover on a tighter, more consistent schedule. That gap is what these
features are meant to capture.
"""

from __future__ import annotations

import numpy as np

Turn = tuple[float, float]

FEATURE_NAMES = [
    "n_caller_turns",
    "n_agent_turns",
    "caller_turn_dur_mean",
    "caller_turn_dur_cv",
    "response_latency_mean",
    "response_latency_cv",
    "barge_in_rate",
    "overlap_rate",
    "post_silence_reentry_mean",
    "post_silence_reentry_std",
    "barge_in_overlap_depth_mean",
    "barge_in_overlap_depth_std",
    "latency_agent_dur_ratio_mean",
    "latency_agent_dur_ratio_std",
]

LONG_SILENCE_S = 1.0


def _default_features() -> dict[str, float]:
    return dict.fromkeys(FEATURE_NAMES, 0.0)


def _cv(arr: np.ndarray) -> float:
    """Coefficient of variation (std/mean), scale-free unlike raw std -- a
    call with a slower absolute pace shouldn't register as "more variable"
    just because its mean is larger. That scale-freeness is what should let
    this generalize across accents, devices and call lengths (CLAUDE.md
    section 2 Robustness), where raw std would pick up pace differences that
    have nothing to do with human vs. synthetic.
    """
    mean = float(arr.mean())
    if mean == 0:
        return 0.0
    return float(arr.std() / mean)


def turn_features(
    caller_turns: list[Turn], agent_turns: list[Turn]
) -> dict[str, float]:
    """Flat dict, fixed key set (FEATURE_NAMES), scalar values -- independent
    of call length or turn count, so it drops directly into a feature matrix
    row.
    """
    feats = _default_features()
    if not caller_turns:
        return feats

    caller_turns = sorted(caller_turns)
    agent_turns = sorted(agent_turns)

    feats["n_caller_turns"] = float(len(caller_turns))
    feats["n_agent_turns"] = float(len(agent_turns))

    durations = np.array([end - start for start, end in caller_turns])
    feats["caller_turn_dur_mean"] = float(durations.mean())
    feats["caller_turn_dur_cv"] = _cv(durations)

    response_latencies = []
    barge_ins = 0
    reentries = []
    # Barge-in overlap depth: how far into the agent's turn the caller talks
    # before the agent's turn ends, i.e. how deep each cut-in goes. (An
    # earlier version measured the gap to the caller's *next* turn instead --
    # that didn't survive speaker-disjoint val, see PR discussion.) Depth is
    # read straight off the two channels' overlap, independent of wording or
    # voice, so a consistently shallow/deep cut-in pattern should transfer to
    # callers and TTS engines never seen in training.
    barge_in_depths = []
    # Response-latency-to-agent-turn-length ratio: humans tend to need more
    # thinking time after a longer, more information-dense agent turn: an
    # ASR->LLM->TTS pipeline's latency is dominated by fixed per-turn
    # pipeline overhead and should stay roughly flat regardless of how long
    # the agent just spoke. This is a relationship between two timing
    # signals rather than an absolute latency value, so it doesn't depend on
    # who is speaking or what was said -- harder for a synthetic pipeline to
    # coincidentally match, and it should hold across unseen speakers/engines.
    latency_agent_dur_ratios = []
    for c_start, c_end in caller_turns:
        prior_agent = [
            (a_start, a_end) for a_start, a_end in agent_turns if a_end <= c_start
        ]
        overlapping_agent = [
            (a_start, a_end)
            for a_start, a_end in agent_turns
            if a_start < c_start < a_end
        ]
        if overlapping_agent:
            barge_ins += 1
            _, a_end = max(overlapping_agent, key=lambda t: t[1])
            barge_in_depths.append(min(c_end, a_end) - c_start)
            continue
        if not prior_agent:
            continue
        a_start, a_end = max(prior_agent, key=lambda t: t[1])
        gap = c_start - a_end
        response_latencies.append(gap)
        agent_dur = a_end - a_start
        if agent_dur > 0:
            latency_agent_dur_ratios.append(gap / agent_dur)
        if gap >= LONG_SILENCE_S:
            reentries.append(gap)

    if response_latencies:
        arr = np.array(response_latencies)
        feats["response_latency_mean"] = float(arr.mean())
        feats["response_latency_cv"] = _cv(arr)

    feats["barge_in_rate"] = barge_ins / len(caller_turns)

    if barge_in_depths:
        arr = np.array(barge_in_depths)
        feats["barge_in_overlap_depth_mean"] = float(arr.mean())
        feats["barge_in_overlap_depth_std"] = float(arr.std())

    if latency_agent_dur_ratios:
        arr = np.array(latency_agent_dur_ratios)
        feats["latency_agent_dur_ratio_mean"] = float(arr.mean())
        feats["latency_agent_dur_ratio_std"] = float(arr.std())

    if reentries:
        arr = np.array(reentries)
        feats["post_silence_reentry_mean"] = float(arr.mean())
        feats["post_silence_reentry_std"] = float(arr.std())

    all_ends = [e for _, e in caller_turns] + [e for _, e in agent_turns]
    all_starts = [s for s, _ in caller_turns] + [s for s, _ in agent_turns]
    call_duration = max(all_ends) - min(all_starts)
    if call_duration > 0:
        overlap_total = 0.0
        for c_start, c_end in caller_turns:
            for a_start, a_end in agent_turns:
                overlap_total += max(0.0, min(c_end, a_end) - max(c_start, a_start))
        feats["overlap_rate"] = overlap_total / call_duration

    return feats
