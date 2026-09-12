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
    "caller_turn_dur_std",
    "response_latency_mean",
    "response_latency_std",
    "barge_in_rate",
    "overlap_rate",
    "post_silence_reentry_mean",
    "post_silence_reentry_std",
    "barge_in_recovery_mean",
    "barge_in_recovery_std",
]

LONG_SILENCE_S = 1.0


def _default_features() -> dict[str, float]:
    return dict.fromkeys(FEATURE_NAMES, 0.0)


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
    feats["caller_turn_dur_std"] = float(durations.std())

    response_latencies = []
    barge_ins = 0
    reentries = []
    # Barge-in recovery: gap from the end of an interrupting caller turn to
    # the start of their *next* turn. This is the closest we can get, from
    # turn boundaries alone, to measuring how a caller settles back into the
    # conversation after cutting the agent off. Humans barge in and then
    # stumble -- pause to collect a thought, or immediately blurt more --
    # producing an inconsistent gap here; a TTS+LLM pipeline that barges in
    # tends to resume on a tighter, more uniform schedule. Unlike timbre,
    # this is a rhythm signal that should hold up across unseen voices/engines.
    barge_in_recoveries = []
    for i, (c_start, c_end) in enumerate(caller_turns):
        prior_agent_ends = [a_end for a_start, a_end in agent_turns if a_end <= c_start]
        overlapping_agent = [
            (a_start, a_end)
            for a_start, a_end in agent_turns
            if a_start < c_start < a_end
        ]
        if overlapping_agent:
            barge_ins += 1
            if i + 1 < len(caller_turns):
                barge_in_recoveries.append(caller_turns[i + 1][0] - c_end)
            continue
        if not prior_agent_ends:
            continue
        gap = c_start - max(prior_agent_ends)
        response_latencies.append(gap)
        if gap >= LONG_SILENCE_S:
            reentries.append(gap)

    if response_latencies:
        arr = np.array(response_latencies)
        feats["response_latency_mean"] = float(arr.mean())
        feats["response_latency_std"] = float(arr.std())

    feats["barge_in_rate"] = barge_ins / len(caller_turns)

    if barge_in_recoveries:
        arr = np.array(barge_in_recoveries)
        feats["barge_in_recovery_mean"] = float(arr.mean())
        feats["barge_in_recovery_std"] = float(arr.std())

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
