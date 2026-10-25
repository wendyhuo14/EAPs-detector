"""Peak-first action potential detection and Detected EAP rules.

This module is the next-generation pipeline for this project.

Key idea:
1. Detect APs by their peaks, not by upward voltage-threshold crossings.
2. Measure transparent voltage and normalized rising phase-space features.
3. Classify Detected EAPs without using human labels as classifier inputs.

The code is intentionally beginner-friendly and uses plain functions instead
of a complicated class structure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


# A common voltage grid makes every normalized rising phase directly
# comparable. These defaults can be overridden by the pipeline functions.
DEFAULT_PHASE_VOLTAGE_GRID = np.arange(0.05, 1.00, 0.05)

# Production EAP evidence thresholds.
#
# Keep these values together so the classifier, saved decision details, and
# diagnostic plots cannot silently use different cutoffs. The phase cutoff is
# event-specific because every row has its own reference-derived threshold.
DEFAULT_EAP_BASELINE_THRESHOLD_MV = -55.0
DEFAULT_EAP_BASELINE_DIFFERENCE_THRESHOLD_MV = 3.0
DEFAULT_EAP_PHASE_THRESHOLD_MULTIPLIER = 0.9
DEFAULT_EAP_MINIMUM_EVIDENCE_COUNT = 2
DEFAULT_EAP_MINIMUM_AMPLITUDE_MV = 60.0


def _phase_grid_column_name(normalized_voltage: float) -> str:
    """Return a readable column name such as phase_dvdt_norm_at_v05."""

    percent = int(round(float(normalized_voltage) * 100))
    return f"phase_dvdt_norm_at_v{percent:02d}"


def infer_current_step_times(
    abf,
    command_channel: int = 1,
    reference_sweep: int | None = None,
    baseline_window_sec: float = 0.05,
    command_change_threshold_pA: float = 1.0,
) -> tuple[float, float]:
    """Estimate step times using a zero-based internal command-channel index."""

    if reference_sweep is None:
        # The last sweep usually has a large command step, making the
        # transition easy to identify.
        reference_sweep = abf.sweepCount - 1

    abf.setSweep(sweepNumber=int(reference_sweep), channel=int(command_channel))

    time_sec = abf.sweepX.copy()
    command_pA = abf.sweepC.copy()
    sampling_rate_hz = abf.dataRate

    baseline_sample_count = max(
        1,
        int(round(baseline_window_sec * sampling_rate_hz)),
    )
    baseline_pA = np.median(command_pA[:baseline_sample_count])
    command_change_pA = np.abs(command_pA - baseline_pA)
    step_indices = np.where(command_change_pA > command_change_threshold_pA)[0]

    if len(step_indices) == 0:
        raise ValueError(
            "Could not detect a current step from the command waveform. "
            "Try a different reference sweep or command threshold."
        )

    step_start_sec = float(time_sec[step_indices[0]])

    # The first sample after the command step marks the boundary between
    # during-step and after-step spikes.
    step_stop_index = step_indices[-1] + 1
    if step_stop_index < len(time_sec):
        step_stop_sec = float(time_sec[step_stop_index])
    else:
        step_stop_sec = float(time_sec[step_indices[-1]])

    return step_start_sec, step_stop_sec


def current_step_amplitude_from_sweep(
    sweep_number: int,
    first_sweep_current_pA: float = -200,
    current_increment_pA: float = 5,
) -> float:
    """Return the fallback current-step amplitude for the FineSteps protocol."""

    return first_sweep_current_pA + current_increment_pA * int(sweep_number)


def classify_spike_timing(
    peak_time_sec: float,
    step_start_sec: float,
    step_stop_sec: float,
) -> str:
    """Label a spike as before, during, or after the current step."""

    if peak_time_sec < step_start_sec:
        return "before_step"
    if peak_time_sec <= step_stop_sec:
        return "during_step"
    return "after_step"


def _ms_to_samples(milliseconds: float, sampling_rate_hz: float) -> int:
    """Convert milliseconds into a sample count."""

    return int(round((milliseconds / 1000) * sampling_rate_hz))


def _odd_savgol_window_samples(
    window_ms: float,
    sampling_rate_hz: float,
    polyorder: int = 2,
    maximum_samples: int | None = None,
) -> int:
    """Return a valid odd sample count for Savitzky-Golay smoothing."""

    window_samples = _ms_to_samples(window_ms, sampling_rate_hz)
    window_samples = max(window_samples, polyorder + 1, 3)

    if window_samples % 2 == 0:
        window_samples += 1

    if maximum_samples is not None and window_samples > maximum_samples:
        window_samples = maximum_samples
        if window_samples % 2 == 0:
            window_samples -= 1

    if window_samples <= polyorder:
        raise ValueError(
            "The voltage trace is too short for the selected Savitzky-Golay "
            "polyorder."
        )

    return window_samples


def _median_voltage_between_times(
    time_sec: np.ndarray,
    voltage_mV: np.ndarray,
    start_sec: float,
    stop_sec: float,
) -> float:
    """Return median voltage between two times."""

    mask = (time_sec >= start_sec) & (time_sec <= stop_sec)

    if not np.any(mask):
        return np.nan

    return float(np.median(voltage_mV[mask]))


def _first_dvdt_crossing_index(
    candidate_indices: np.ndarray,
    dvdt_mV_per_ms: np.ndarray,
    threshold_mV_per_ms: float,
) -> int | None:
    """Return the first index where dV/dt reaches a threshold."""

    passing_indices = candidate_indices[dvdt_mV_per_ms[candidate_indices] >= threshold_mV_per_ms]

    if len(passing_indices) == 0:
        return None

    return int(passing_indices[0])


def _safe_value(values: np.ndarray, index: int | None) -> float:
    """Return values[index], or NaN when index is missing."""

    if index is None:
        return np.nan

    return float(values[index])


def _interpolate_crossing_time_sec(
    time_sec: np.ndarray,
    voltage_mV: np.ndarray,
    first_index: int,
    target_voltage_mV: float,
) -> float:
    """Estimate when voltage crosses a target between two samples."""

    second_index = first_index + 1
    first_voltage = float(voltage_mV[first_index])
    second_voltage = float(voltage_mV[second_index])

    if second_voltage == first_voltage:
        return float(time_sec[first_index])

    fraction = (target_voltage_mV - first_voltage) / (
        second_voltage - first_voltage
    )
    return float(
        time_sec[first_index]
        + fraction * (time_sec[second_index] - time_sec[first_index])
    )


def _measure_duration_at_half_height_ms(
    time_sec: np.ndarray,
    smoothed_voltage_mV: np.ndarray,
    peak_index: int,
    baseline_mV: float,
    peak_voltage_mV: float,
    search_window_ms: float = 3.0,
) -> float:
    """Measure spike width at half of its baseline-to-peak amplitude."""

    if np.isnan(baseline_mV) or peak_voltage_mV <= baseline_mV:
        return np.nan

    sampling_rate_hz = 1 / float(time_sec[1] - time_sec[0])
    search_samples = _ms_to_samples(search_window_ms, sampling_rate_hz)
    half_height_mV = baseline_mV + 0.5 * (peak_voltage_mV - baseline_mV)

    start_index = max(0, peak_index - search_samples)
    stop_index = min(len(smoothed_voltage_mV) - 1, peak_index + search_samples)

    rising_crossings = np.where(
        (smoothed_voltage_mV[start_index:peak_index] < half_height_mV)
        & (smoothed_voltage_mV[start_index + 1 : peak_index + 1] >= half_height_mV)
    )[0]
    falling_crossings = np.where(
        (smoothed_voltage_mV[peak_index:stop_index] >= half_height_mV)
        & (smoothed_voltage_mV[peak_index + 1 : stop_index + 1] < half_height_mV)
    )[0]

    if len(rising_crossings) == 0 or len(falling_crossings) == 0:
        return np.nan

    rising_index = start_index + int(rising_crossings[-1])
    falling_index = peak_index + int(falling_crossings[0])

    rising_time_sec = _interpolate_crossing_time_sec(
        time_sec,
        smoothed_voltage_mV,
        rising_index,
        half_height_mV,
    )
    falling_time_sec = _interpolate_crossing_time_sec(
        time_sec,
        smoothed_voltage_mV,
        falling_index,
        half_height_mV,
    )

    return float((falling_time_sec - rising_time_sec) * 1000)


def _longest_true_run(values: np.ndarray) -> int:
    """Return the longest consecutive run of True values."""

    longest_run = 0
    current_run = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    return int(longest_run)


def _interpolate_with_repeated_x_values(
    x_values: np.ndarray,
    y_values: np.ndarray,
    target_x_values: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate after safely combining repeated x values.

    ``numpy.interp`` requires an increasing x-axis. Rising AP traces can have
    tiny reversals or repeated voltage samples, so x is first converted to a
    non-decreasing envelope. The median y value is used where envelope values
    repeat. This preserves interpolation between samples without selecting a
    nearest sample.
    """

    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    target_x_values = np.asarray(target_x_values, dtype=float)
    output = np.full(target_x_values.shape, np.nan, dtype=float)

    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    if finite_mask.sum() < 2:
        return output

    monotonic_x = np.maximum.accumulate(x_values[finite_mask])
    finite_y = y_values[finite_mask]

    unique_x, inverse = np.unique(monotonic_x, return_inverse=True)
    if len(unique_x) < 2:
        return output

    combined_y = np.array(
        [np.median(finite_y[inverse == group]) for group in range(len(unique_x))],
        dtype=float,
    )
    inside_range = (
        (target_x_values >= unique_x[0])
        & (target_x_values <= unique_x[-1])
    )
    output[inside_range] = np.interp(
        target_x_values[inside_range],
        unique_x,
        combined_y,
    )
    return output


def _normalized_rising_phase_features(
    smoothed_voltage_mV: np.ndarray,
    dvdt_mV_per_ms: np.ndarray,
    peak_index: int,
    baseline_mV: float,
    peak_voltage_mV: float,
    sampling_rate_hz: float,
    phase_pre_peak_window_ms: float,
    phase_voltage_grid: np.ndarray,
    onset_normalized_voltage: float = 0.05,
    stable_crossing_samples: int = 3,
    minimum_upward_step_fraction: float = 0.55,
) -> dict:
    """Measure one normalized onset-to-peak rising phase-space trajectory.

    The onset is the last stable upward crossing of normalized voltage 0.05
    found while searching backward from the peak. A small amount of voltage
    reversal is allowed, but the trajectory must progress generally upward.
    """

    grid = np.asarray(phase_voltage_grid, dtype=float)
    grid_columns = [_phase_grid_column_name(value) for value in grid]
    unavailable = {
        "phase_features_available": False,
        "phase_feature_qc": "invalid_rising_trajectory",
        "normalized_rising_onset_index": np.nan,
        "max_rising_dvdt_mV_per_ms": np.nan,
        "voltage_at_10pct_max_dvdt_mV": np.nan,
        "voltage_at_25pct_max_dvdt_mV": np.nan,
        "voltage_at_50pct_max_dvdt_mV": np.nan,
        "normalized_voltage_at_10pct_max_dvdt": np.nan,
        "normalized_voltage_at_25pct_max_dvdt": np.nan,
        "normalized_voltage_at_50pct_max_dvdt": np.nan,
    }
    unavailable.update({column: np.nan for column in grid_columns})

    amplitude_mV = peak_voltage_mV - baseline_mV
    if not np.isfinite(amplitude_mV) or amplitude_mV <= 0:
        unavailable["phase_feature_qc"] = "invalid_amplitude"
        return unavailable

    pre_peak_samples = max(
        1,
        _ms_to_samples(phase_pre_peak_window_ms, sampling_rate_hz),
    )
    search_start_index = max(0, peak_index - pre_peak_samples)
    search_indices = np.arange(search_start_index, peak_index + 1)
    normalized_voltage = (
        smoothed_voltage_mV[search_indices] - baseline_mV
    ) / amplitude_mV

    crossing_positions = np.where(
        (normalized_voltage[:-1] < onset_normalized_voltage)
        & (normalized_voltage[1:] >= onset_normalized_voltage)
    )[0] + 1

    onset_position = None
    for crossing_position in crossing_positions[::-1]:
        stable_stop = min(
            len(normalized_voltage),
            crossing_position + stable_crossing_samples,
        )
        stable_values = normalized_voltage[crossing_position:stable_stop]
        candidate_values = normalized_voltage[crossing_position:]
        voltage_steps = np.diff(candidate_values)

        stable_crossing = (
            len(stable_values) == stable_crossing_samples
            and np.all(stable_values >= onset_normalized_voltage - 0.01)
        )
        enough_progress = (
            len(candidate_values) >= 3
            and candidate_values[-1] >= 0.80
            and candidate_values[-1] - candidate_values[0] >= 0.60
        )
        upward_fraction = (
            np.mean(voltage_steps >= -0.01) if len(voltage_steps) else 0.0
        )
        generally_upward = upward_fraction >= minimum_upward_step_fraction

        if stable_crossing and enough_progress and generally_upward:
            onset_position = int(crossing_position)
            break

    if onset_position is None:
        unavailable["phase_feature_qc"] = "no_stable_v05_onset"
        return unavailable

    rising_indices = search_indices[onset_position:]
    rising_normalized_voltage = (
        smoothed_voltage_mV[rising_indices] - baseline_mV
    ) / amplitude_mV
    rising_dvdt = dvdt_mV_per_ms[rising_indices]

    finite_dvdt = rising_dvdt[np.isfinite(rising_dvdt)]
    if len(finite_dvdt) == 0:
        unavailable["phase_feature_qc"] = "invalid_rising_dvdt"
        return unavailable

    max_rising_dvdt = float(np.max(finite_dvdt))
    if not np.isfinite(max_rising_dvdt) or max_rising_dvdt <= 0:
        unavailable["phase_feature_qc"] = "invalid_max_rising_dvdt"
        return unavailable

    normalized_dvdt = rising_dvdt / max_rising_dvdt
    interpolated_dvdt = _interpolate_with_repeated_x_values(
        x_values=rising_normalized_voltage,
        y_values=normalized_dvdt,
        target_x_values=grid,
    )

    # Estimate voltage at 10%, 25%, and 50% of maximum rising dV/dt on the
    # ascending dV/dt limb. Interpolation is used in both raw and normalized
    # voltage coordinates.
    max_dvdt_position = int(np.nanargmax(rising_dvdt))
    ascending_norm_dvdt = normalized_dvdt[: max_dvdt_position + 1]
    ascending_voltage_mV = smoothed_voltage_mV[
        rising_indices[: max_dvdt_position + 1]
    ]
    ascending_normalized_voltage = rising_normalized_voltage[
        : max_dvdt_position + 1
    ]
    percent_targets = np.array([0.10, 0.25, 0.50])
    voltage_at_percent = _interpolate_with_repeated_x_values(
        ascending_norm_dvdt,
        ascending_voltage_mV,
        percent_targets,
    )
    normalized_voltage_at_percent = _interpolate_with_repeated_x_values(
        ascending_norm_dvdt,
        ascending_normalized_voltage,
        percent_targets,
    )

    phase_features_available = bool(np.isfinite(interpolated_dvdt).sum() >= 2)
    result = {
        "phase_features_available": phase_features_available,
        "phase_feature_qc": (
            "valid" if phase_features_available else "insufficient_grid_coverage"
        ),
        "normalized_rising_onset_index": int(rising_indices[0]),
        "max_rising_dvdt_mV_per_ms": max_rising_dvdt,
        "voltage_at_10pct_max_dvdt_mV": float(voltage_at_percent[0]),
        "voltage_at_25pct_max_dvdt_mV": float(voltage_at_percent[1]),
        "voltage_at_50pct_max_dvdt_mV": float(voltage_at_percent[2]),
        "normalized_voltage_at_10pct_max_dvdt": float(normalized_voltage_at_percent[0]),
        "normalized_voltage_at_25pct_max_dvdt": float(normalized_voltage_at_percent[1]),
        "normalized_voltage_at_50pct_max_dvdt": float(normalized_voltage_at_percent[2]),
    }
    if not result["phase_features_available"]:
        result["phase_feature_qc"] = "insufficient_grid_coverage"
    result.update(
        {
            column: float(value)
            for column, value in zip(grid_columns, interpolated_dvdt)
        }
    )
    return result


def detect_spikes_in_sweep_peak_based(
    abf,
    sweep_number: int,
    channel: int = 2,
    min_peak_voltage_mV: float = -5,
    min_peak_prominence_mV: float = 20,
    min_peak_distance_ms: float = 1.2,
) -> pd.DataFrame:
    """Detect APs in one sweep using peak finding.

    ``channel`` is a zero-based internal pyABF index. User-facing code converts
    one-based channel numbers before calling this function.

    Unlike voltage threshold crossing, this does not require the voltage to
    fall below a specific threshold before the next spike can be detected.
    """

    abf.setSweep(sweepNumber=int(sweep_number), channel=int(channel))
    time_sec = abf.sweepX.copy()
    voltage_mV = abf.sweepY.copy()
    sampling_rate_hz = abf.dataRate

    min_peak_distance_samples = max(
        _ms_to_samples(min_peak_distance_ms, sampling_rate_hz),
        1,
    )

    peaks, properties = find_peaks(
        voltage_mV,
        height=min_peak_voltage_mV,
        prominence=min_peak_prominence_mV,
        distance=min_peak_distance_samples,
    )

    detected_spikes = []

    for peak_number, peak_index in enumerate(peaks):
        detected_spikes.append(
            {
                "sweep": int(sweep_number),
                "channel_index": int(channel),
                "peak_index": int(peak_index),
                "peak_time_sec": float(time_sec[peak_index]),
                "peak_voltage_mV": float(voltage_mV[peak_index]),
                "peak_prominence_mV": float(properties["prominences"][peak_number]),
            }
        )

    return pd.DataFrame(
        detected_spikes,
        columns=[
            "sweep",
            "channel_index",
            "peak_index",
            "peak_time_sec",
            "peak_voltage_mV",
            "peak_prominence_mV",
        ],
    )


def detect_spikes_all_sweeps_peak_based(
    abf,
    channel: int = 2,
    min_peak_voltage_mV: float = -5,
    min_peak_prominence_mV: float = 20,
    min_peak_distance_ms: float = 1.2,
) -> pd.DataFrame:
    """Run detection using a zero-based internal pyABF channel index."""

    spike_tables = []

    for sweep_number in range(abf.sweepCount):
        sweep_spikes = detect_spikes_in_sweep_peak_based(
            abf=abf,
            sweep_number=sweep_number,
            channel=channel,
            min_peak_voltage_mV=min_peak_voltage_mV,
            min_peak_prominence_mV=min_peak_prominence_mV,
            min_peak_distance_ms=min_peak_distance_ms,
        )

        if len(sweep_spikes) > 0:
            spike_tables.append(sweep_spikes)

    if len(spike_tables) == 0:
        return pd.DataFrame(
            columns=[
                "sweep",
                "channel_index",
                "peak_index",
                "peak_time_sec",
                "peak_voltage_mV",
                "peak_prominence_mV",
            ]
        )

    return pd.concat(spike_tables, ignore_index=True)


def measure_peak_based_spike_features(
    abf,
    spike_row,
    step_start_sec: float,
    step_stop_sec: float,
    baseline_start_ms_before_peak: float = 5.0,
    baseline_stop_ms_before_peak: float = 3.0,
    phase_pre_peak_window_ms: float = 5.0,
    phase_post_peak_window_ms: float = 6.0,
    savgol_window_ms: float = 0.25,
    savgol_polyorder: int = 2,
    phase_voltage_grid: np.ndarray | None = None,
    current_step_pA: float | None = None,
) -> dict:
    """Measure baseline, legacy, and normalized phase-space AP features.

    The primary phase-space comparison uses only the rising phase. The longer
    post-peak window is retained for complete waveform visualization and
    clipping checks. Savitzky-Golay smoothing and dV/dt calculation are shared
    by candidates and normal-reference APs.
    """

    sweep_number = int(spike_row["sweep"])
    channel = int(spike_row["channel_index"])
    peak_index = int(spike_row["peak_index"])

    abf.setSweep(sweepNumber=sweep_number, channel=channel)
    time_sec = abf.sweepX.copy()
    voltage_mV = abf.sweepY.copy()
    sampling_rate_hz = abf.dataRate
    dt_ms = 1000 / sampling_rate_hz

    if phase_voltage_grid is None:
        phase_voltage_grid = DEFAULT_PHASE_VOLTAGE_GRID.copy()

    peak_time_sec = float(time_sec[peak_index])
    peak_voltage_mV = float(voltage_mV[peak_index])

    baseline_start_sec = peak_time_sec - baseline_start_ms_before_peak / 1000
    baseline_stop_sec = peak_time_sec - baseline_stop_ms_before_peak / 1000
    baseline_mV = _median_voltage_between_times(
        time_sec=time_sec,
        voltage_mV=voltage_mV,
        start_sec=baseline_start_sec,
        stop_sec=baseline_stop_sec,
    )

    savgol_window_samples = _odd_savgol_window_samples(
        window_ms=savgol_window_ms,
        sampling_rate_hz=sampling_rate_hz,
        polyorder=savgol_polyorder,
        maximum_samples=len(voltage_mV),
    )
    smoothed_voltage_mV = savgol_filter(
        voltage_mV,
        window_length=savgol_window_samples,
        polyorder=savgol_polyorder,
        mode="interp",
    )
    dvdt_mV_per_ms = np.gradient(smoothed_voltage_mV, dt_ms)

    duration_at_half_height_ms = _measure_duration_at_half_height_ms(
        time_sec=time_sec,
        smoothed_voltage_mV=smoothed_voltage_mV,
        peak_index=peak_index,
        baseline_mV=baseline_mV,
        peak_voltage_mV=peak_voltage_mV,
    )

    shape_start_sec = peak_time_sec - phase_pre_peak_window_ms / 1000
    shape_stop_sec = peak_time_sec + phase_post_peak_window_ms / 1000
    shape_mask = (time_sec >= shape_start_sec) & (time_sec <= shape_stop_sec)
    shape_indices = np.where(shape_mask)[0]

    rising_mask = (time_sec >= shape_start_sec) & (time_sec <= peak_time_sec)
    rising_indices = np.where(rising_mask)[0]

    dvdt20_index = _first_dvdt_crossing_index(rising_indices, dvdt_mV_per_ms, 20)
    dvdt50_index = _first_dvdt_crossing_index(rising_indices, dvdt_mV_per_ms, 50)
    dvdt100_index = _first_dvdt_crossing_index(rising_indices, dvdt_mV_per_ms, 100)
    dvdt300_index = _first_dvdt_crossing_index(rising_indices, dvdt_mV_per_ms, 300)

    voltage_at_dvdt20_mV = _safe_value(smoothed_voltage_mV, dvdt20_index)
    voltage_at_dvdt50_mV = _safe_value(smoothed_voltage_mV, dvdt50_index)
    voltage_at_dvdt100_mV = _safe_value(smoothed_voltage_mV, dvdt100_index)
    voltage_at_dvdt300_mV = _safe_value(smoothed_voltage_mV, dvdt300_index)

    if dvdt20_index is None or dvdt100_index is None:
        foot_duration_20_to_100_ms = np.nan
        foot_voltage_span_20_to_100_mV = np.nan
    else:
        foot_duration_20_to_100_ms = float((time_sec[dvdt100_index] - time_sec[dvdt20_index]) * 1000)
        foot_voltage_span_20_to_100_mV = float(
            smoothed_voltage_mV[dvdt100_index] - smoothed_voltage_mV[dvdt20_index]
        )

    if len(rising_indices) == 0:
        max_dvdt_mV_per_ms = np.nan
        voltage_at_max_dvdt_mV = np.nan
    else:
        max_dvdt_index = int(rising_indices[np.argmax(dvdt_mV_per_ms[rising_indices])])
        max_dvdt_mV_per_ms = float(dvdt_mV_per_ms[max_dvdt_index])
        voltage_at_max_dvdt_mV = float(smoothed_voltage_mV[max_dvdt_index])

    phase_features = _normalized_rising_phase_features(
        smoothed_voltage_mV=smoothed_voltage_mV,
        dvdt_mV_per_ms=dvdt_mV_per_ms,
        peak_index=peak_index,
        baseline_mV=baseline_mV,
        peak_voltage_mV=peak_voltage_mV,
        sampling_rate_hz=sampling_rate_hz,
        phase_pre_peak_window_ms=phase_pre_peak_window_ms,
        phase_voltage_grid=phase_voltage_grid,
    )

    # A perfectly flat peak for several consecutive samples can indicate
    # clipping. This is a configurable-reference exclusion feature, not an
    # EAP rule by itself.
    plateau_tolerance_mV = 0.01
    peak_neighborhood = voltage_mV[shape_indices]
    peak_plateau_sample_count = _longest_true_run(
        np.isclose(
            peak_neighborhood,
            peak_voltage_mV,
            atol=plateau_tolerance_mV,
            rtol=0,
        )
    )
    possible_saturation_or_clipping = peak_plateau_sample_count >= 3

    timing_region = classify_spike_timing(
        peak_time_sec=peak_time_sec,
        step_start_sec=step_start_sec,
        step_stop_sec=step_stop_sec,
    )

    if current_step_pA is None:
        current_step_pA = current_step_amplitude_from_sweep(sweep_number)

    measured_features = {
        "sweep": sweep_number,
        "channel_index": channel,
        "current_step_pA": float(current_step_pA),
        "timing_region": timing_region,
        "time_after_step_end_ms": float((peak_time_sec - step_stop_sec) * 1000),
        "peak_index": peak_index,
        "peak_time_sec": peak_time_sec,
        "peak_voltage_mV": peak_voltage_mV,
        "peak_prominence_mV": float(spike_row.get("peak_prominence_mV", np.nan)),
        "baseline_mV": baseline_mV,
        "amp_from_baseline_mV": float(peak_voltage_mV - baseline_mV),
        "duration_at_half_height_ms": duration_at_half_height_ms,
        "max_dvdt_mV_per_ms": max_dvdt_mV_per_ms,
        "voltage_at_max_dvdt_mV": voltage_at_max_dvdt_mV,
        "voltage_at_dvdt20_mV": voltage_at_dvdt20_mV,
        "voltage_at_dvdt50_mV": voltage_at_dvdt50_mV,
        "voltage_at_dvdt100_mV": voltage_at_dvdt100_mV,
        "voltage_at_dvdt300_mV": voltage_at_dvdt300_mV,
        "foot_duration_20_to_100_ms": foot_duration_20_to_100_ms,
        "foot_voltage_span_20_to_100_mV": foot_voltage_span_20_to_100_mV,
        "peak_plateau_sample_count": peak_plateau_sample_count,
        "possible_saturation_or_clipping": possible_saturation_or_clipping,
    }
    measured_features.update(phase_features)
    return measured_features


def build_peak_based_feature_table(
    abf,
    peak_spikes_df: pd.DataFrame,
    step_start_sec: float,
    step_stop_sec: float,
    confirmed_eap_sweeps: list[int] | None = None,
    current_step_pA_by_sweep: dict[int, float] | None = None,
    phase_pre_peak_window_ms: float = 5.0,
    phase_post_peak_window_ms: float = 6.0,
    savgol_window_ms: float = 0.25,
    savgol_polyorder: int = 2,
    phase_voltage_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build one feature row for every peak-based spike."""

    if confirmed_eap_sweeps is None:
        confirmed_eap_sweeps = []

    feature_rows = []

    for _, spike_row in peak_spikes_df.iterrows():
        sweep_number = int(spike_row["sweep"])
        current_step_pA = None
        if current_step_pA_by_sweep is not None:
            current_step_pA = current_step_pA_by_sweep.get(sweep_number)

        feature_rows.append(
            measure_peak_based_spike_features(
                abf=abf,
                spike_row=spike_row,
                step_start_sec=step_start_sec,
                step_stop_sec=step_stop_sec,
                phase_pre_peak_window_ms=phase_pre_peak_window_ms,
                phase_post_peak_window_ms=phase_post_peak_window_ms,
                savgol_window_ms=savgol_window_ms,
                savgol_polyorder=savgol_polyorder,
                phase_voltage_grid=phase_voltage_grid,
                current_step_pA=current_step_pA,
            )
        )

    feature_df = pd.DataFrame(feature_rows)

    confirmed_eap_sweeps = set(int(sweep) for sweep in confirmed_eap_sweeps)
    confirmed_mask = (
        feature_df["sweep"].isin(confirmed_eap_sweeps)
        & (feature_df["timing_region"] == "after_step")
    )

    feature_df["manual_label"] = "unlabeled"
    feature_df.loc[confirmed_mask, "manual_label"] = "EAP"

    return feature_df


def _phase_reference_quality_mask(
    feature_df: pd.DataFrame,
    minimum_amplitude_mV: float,
    maximum_amplitude_mV: float,
    minimum_peak_voltage_mV: float,
    maximum_peak_voltage_mV: float,
    minimum_half_height_ms: float,
    maximum_half_height_ms: float,
    minimum_max_dvdt_mV_per_ms: float,
    maximum_max_dvdt_mV_per_ms: float,
    minimum_valid_grid_points: int,
) -> pd.Series:
    """Identify clean normal AP shapes using configurable physiological QC."""

    phase_columns = [
        column
        for column in feature_df.columns
        if column.startswith("phase_dvdt_norm_at_v")
    ]
    valid_grid_point_count = feature_df[phase_columns].notna().sum(axis=1)
    return (
        (feature_df["timing_region"] == "during_step")
        & feature_df["phase_features_available"].fillna(False)
        & feature_df["baseline_mV"].between(-110, -30, inclusive="both")
        & feature_df["amp_from_baseline_mV"].between(
            minimum_amplitude_mV, maximum_amplitude_mV, inclusive="both"
        )
        & feature_df["peak_voltage_mV"].between(
            minimum_peak_voltage_mV, maximum_peak_voltage_mV, inclusive="both"
        )
        & feature_df["duration_at_half_height_ms"].between(
            minimum_half_height_ms, maximum_half_height_ms, inclusive="both"
        )
        & feature_df["max_dvdt_mV_per_ms"].between(
            minimum_max_dvdt_mV_per_ms,
            maximum_max_dvdt_mV_per_ms,
            inclusive="both",
        )
        & (~feature_df["possible_saturation_or_clipping"].fillna(True))
        & (valid_grid_point_count >= minimum_valid_grid_points)
    )


def _phase_curve_distances(
    candidate_values: np.ndarray,
    template_values: np.ndarray,
    phase_voltage_grid: np.ndarray,
) -> tuple[float, float, float, int]:
    """Return median, maximum, area, and point count for two phase curves."""

    candidate_values = np.asarray(candidate_values, dtype=float)
    template_values = np.asarray(template_values, dtype=float)
    grid = np.asarray(phase_voltage_grid, dtype=float)
    valid_mask = (
        np.isfinite(candidate_values)
        & np.isfinite(template_values)
        & np.isfinite(grid)
    )
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        return np.nan, np.nan, np.nan, 0

    differences = np.abs(candidate_values[valid_mask] - template_values[valid_mask])
    median_distance = float(np.median(differences))
    max_distance = float(np.max(differences))
    area_distance = (
        float(np.trapezoid(differences, grid[valid_mask]))
        if valid_count >= 2
        else np.nan
    )
    return median_distance, max_distance, area_distance, valid_count


def _leave_one_out_median_templates(reference_matrix: np.ndarray) -> np.ndarray:
    """Build one median template per excluded row without an O(n-squared) loop.

    For each normalized-voltage column, sorted ranks reveal the median after
    removing any one value. This is exactly equivalent to repeatedly calling
    nanmedian, but remains practical with tens of thousands of normal APs.
    """

    matrix = np.asarray(reference_matrix, dtype=float)
    template_matrix = np.full(matrix.shape, np.nan, dtype=float)

    for column_number in range(matrix.shape[1]):
        column_values = matrix[:, column_number]
        finite_row_indices = np.where(np.isfinite(column_values))[0]
        finite_count = len(finite_row_indices)
        if finite_count == 0:
            continue
        if finite_count == 1:
            template_matrix[
                ~np.isfinite(column_values), column_number
            ] = column_values[finite_row_indices[0]]
            continue

        sorted_order = np.argsort(column_values[finite_row_indices], kind="stable")
        sorted_rows = finite_row_indices[sorted_order]
        sorted_values = column_values[sorted_rows]
        rank_by_row = np.full(len(column_values), -1, dtype=int)
        rank_by_row[sorted_rows] = np.arange(finite_count)

        # A missing value was not part of the median, so excluding that row
        # leaves the ordinary all-finite median unchanged.
        template_matrix[~np.isfinite(column_values), column_number] = np.median(
            sorted_values
        )

        remaining_count = finite_count - 1
        if remaining_count % 2 == 1:
            remaining_positions = [(remaining_count - 1) // 2]
        else:
            remaining_positions = [
                remaining_count // 2 - 1,
                remaining_count // 2,
            ]

        for row_index in finite_row_indices:
            excluded_rank = rank_by_row[row_index]
            source_positions = [
                position + 1 if excluded_rank <= position else position
                for position in remaining_positions
            ]
            template_matrix[row_index, column_number] = float(
                np.mean(sorted_values[source_positions])
            )

    return template_matrix


def add_normal_references(
    spike_feature_df: pd.DataFrame,
    step_start_sec: float,
    step_stop_sec: float,
    phase_voltage_grid: np.ndarray | None = None,
    matched_protocol_reference_df: pd.DataFrame | None = None,
    minimum_baseline_reference_spikes: int = 5,
    minimum_phase_reference_spikes: int = 5,
    preferred_phase_reference_spikes: int = 10,
    reference_edge_buffer_ms: float = 20.0,
    reference_minimum_amplitude_mV: float = 40.0,
    reference_maximum_amplitude_mV: float = 140.0,
    reference_minimum_peak_voltage_mV: float = -20.0,
    reference_maximum_peak_voltage_mV: float = 80.0,
    reference_minimum_half_height_ms: float = 0.10,
    reference_maximum_half_height_ms: float = 3.0,
    reference_minimum_max_dvdt_mV_per_ms: float = 20.0,
    reference_maximum_max_dvdt_mV_per_ms: float = 2000.0,
    minimum_valid_phase_grid_points: int = 15,
    phase_distance_percentile: float = 90.0,
    phase_mad_epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Add independent baseline and normalized phase-space references.

    Baselines prefer same-sweep normal APs and fall back to the same file.
    Phase-space uses only clean, physiologically plausible, non-clipped APs.
    A matched-protocol feature table may be supplied if this file has fewer
    than five clean shapes. The default 90th percentile prioritizes initial
    sensitivity; it is not scientifically final and must be validated on
    held-out ABF files. Human labels are deliberately not read here.
    """

    result_df = spike_feature_df.copy()
    if phase_voltage_grid is None:
        phase_voltage_grid = DEFAULT_PHASE_VOLTAGE_GRID.copy()
    phase_voltage_grid = np.asarray(phase_voltage_grid, dtype=float)
    phase_columns = [_phase_grid_column_name(value) for value in phase_voltage_grid]
    missing_columns = [column for column in phase_columns if column not in result_df]
    if missing_columns:
        raise ValueError(
            "Missing normalized phase-space columns: " + ", ".join(missing_columns)
        )

    # Baseline-reference QC is intentionally less restrictive than AP-shape
    # QC. A file can have usable baselines but stimulus-contaminated shapes.
    edge_buffer_sec = reference_edge_buffer_ms / 1000
    baseline_mask = (
        (result_df["timing_region"] == "during_step")
        & (result_df["peak_time_sec"] >= step_start_sec + edge_buffer_sec)
        & (result_df["peak_time_sec"] <= step_stop_sec - edge_buffer_sec)
        & result_df["baseline_mV"].between(-110, -30, inclusive="both")
        & (result_df["amp_from_baseline_mV"] >= 20)
    )
    baseline_df = result_df[baseline_mask].copy()
    legacy_columns = [
        "baseline_mV",
        "voltage_at_dvdt20_mV",
        "voltage_at_dvdt50_mV",
        "voltage_at_dvdt100_mV",
        "foot_duration_20_to_100_ms",
        "max_dvdt_mV_per_ms",
    ]
    normal_names = {
        "baseline_mV": "normal_median_baseline_mV",
        "voltage_at_dvdt20_mV": "normal_median_voltage_at_dvdt20_mV",
        "voltage_at_dvdt50_mV": "normal_median_voltage_at_dvdt50_mV",
        "voltage_at_dvdt100_mV": "normal_median_voltage_at_dvdt100_mV",
        "foot_duration_20_to_100_ms": "normal_median_foot_duration_20_to_100_ms",
        "max_dvdt_mV_per_ms": "normal_median_max_dvdt_mV_per_ms",
    }
    global_baseline_count = len(baseline_df)
    global_baseline = baseline_df[legacy_columns].median(numeric_only=True)
    per_sweep = (
        baseline_df.groupby("sweep")[legacy_columns]
        .median(numeric_only=True)
        .rename(columns=normal_names)
        .reset_index()
    )
    per_sweep_counts = (
        baseline_df.groupby("sweep")
        .size()
        .reset_index(name="same_sweep_baseline_reference_spike_count")
    )
    per_sweep = per_sweep.merge(per_sweep_counts, on="sweep", how="left")
    result_df = result_df.merge(per_sweep, on="sweep", how="left")

    use_global = (
        result_df["same_sweep_baseline_reference_spike_count"].fillna(0)
        < minimum_baseline_reference_spikes
    )
    result_df["baseline_reference_source"] = np.where(
        use_global, "same_file", "same_sweep"
    )
    for raw_column, normal_column in normal_names.items():
        global_value = (
            float(global_baseline[raw_column])
            if raw_column in global_baseline.index
            else np.nan
        )
        replace_mask = use_global | result_df[normal_column].isna()
        result_df.loc[replace_mask, normal_column] = global_value

    result_df["baseline_reference_spike_count"] = np.where(
        use_global,
        global_baseline_count,
        result_df["same_sweep_baseline_reference_spike_count"].fillna(0),
    ).astype(int)
    result_df["baseline_reference_available"] = (
        (result_df["baseline_reference_spike_count"] >= minimum_baseline_reference_spikes)
        & result_df["normal_median_baseline_mV"].notna()
    )
    result_df["baseline_reference_qc"] = np.where(
        result_df["baseline_reference_available"],
        "valid",
        "insufficient_reference",
    )
    result_df.loc[
        ~result_df["baseline_reference_available"], "baseline_reference_source"
    ] = "none"

    quality_arguments = dict(
        minimum_amplitude_mV=reference_minimum_amplitude_mV,
        maximum_amplitude_mV=reference_maximum_amplitude_mV,
        minimum_peak_voltage_mV=reference_minimum_peak_voltage_mV,
        maximum_peak_voltage_mV=reference_maximum_peak_voltage_mV,
        minimum_half_height_ms=reference_minimum_half_height_ms,
        maximum_half_height_ms=reference_maximum_half_height_ms,
        minimum_max_dvdt_mV_per_ms=reference_minimum_max_dvdt_mV_per_ms,
        maximum_max_dvdt_mV_per_ms=reference_maximum_max_dvdt_mV_per_ms,
        minimum_valid_grid_points=minimum_valid_phase_grid_points,
    )
    phase_mask = _phase_reference_quality_mask(result_df, **quality_arguments)
    result_df["phase_reference_event_qc_pass"] = phase_mask
    selected_phase_df = result_df[phase_mask].copy()
    phase_source = "same_file"

    if (
        len(selected_phase_df) < minimum_phase_reference_spikes
        and matched_protocol_reference_df is not None
    ):
        matched_df = matched_protocol_reference_df.copy()
        matched_mask = _phase_reference_quality_mask(matched_df, **quality_arguments)
        matched_clean_df = matched_df[matched_mask].copy()
        if len(matched_clean_df) >= minimum_phase_reference_spikes:
            selected_phase_df = matched_clean_df
            phase_source = "matched_protocol"

    reference_count = len(selected_phase_df)
    reference_available = reference_count >= minimum_phase_reference_spikes
    if reference_available:
        reference_qc = (
            "low_reference_count"
            if reference_count < preferred_phase_reference_spikes
            else (
                "valid_same_file"
                if phase_source == "same_file"
                else "valid_matched_protocol"
            )
        )
    else:
        during_df = result_df[result_df["timing_region"] == "during_step"]
        contaminated = (
            (during_df["peak_voltage_mV"] > reference_maximum_peak_voltage_mV)
            | (during_df["amp_from_baseline_mV"] > reference_maximum_amplitude_mV)
            | (during_df["duration_at_half_height_ms"] > reference_maximum_half_height_ms)
        )
        if len(during_df) and contaminated.mean() >= 0.5:
            reference_qc = "stimulus_contaminated"
        elif reference_count:
            reference_qc = "insufficient_reference"
        else:
            reference_qc = "invalid_reference"
        phase_source = "none"

    template_values = (
        selected_phase_df[phase_columns].median().to_numpy(float)
        if reference_available
        else np.full(len(phase_columns), np.nan)
    )
    for column, value in zip(phase_columns, template_values):
        template_column = column.replace(
            "phase_dvdt_norm_at_", "normal_phase_template_dvdt_norm_at_"
        )
        result_df[template_column] = value

    # Leave-one-out normal distances prevent self-comparison bias.
    normal_distances = np.array([], dtype=float)
    if reference_available:
        matrix = selected_phase_df[phase_columns].to_numpy(float)
        leave_one_out_templates = _leave_one_out_median_templates(matrix)
        absolute_differences = np.abs(matrix - leave_one_out_templates)
        valid_point_counts = np.isfinite(absolute_differences).sum(axis=1)
        with np.errstate(all="ignore"):
            all_normal_distances = np.nanmedian(absolute_differences, axis=1)
        valid_rows = (
            (valid_point_counts >= minimum_valid_phase_grid_points)
            & np.isfinite(all_normal_distances)
        )
        normal_distances = all_normal_distances[valid_rows]

    if len(normal_distances) >= minimum_phase_reference_spikes:
        normal_distance_median = float(np.median(normal_distances))
        normal_distance_mad = float(
            np.median(np.abs(normal_distances - normal_distance_median))
        )
        normal_distance_threshold = float(
            np.percentile(normal_distances, phase_distance_percentile)
        )
    else:
        normal_distance_median = np.nan
        normal_distance_mad = np.nan
        normal_distance_threshold = np.nan
        if reference_available:
            # Clean APs existed, but too few valid leave-one-out distances
            # remained to characterize normal phase-space variability.
            reference_qc = "invalid_reference"
        reference_available = False

    result_df["phase_reference_available"] = reference_available
    result_df["phase_reference_source"] = phase_source if reference_available else "none"
    result_df["phase_reference_spike_count"] = reference_count
    result_df["phase_reference_qc"] = reference_qc
    result_df["normal_phase_distance_median"] = normal_distance_median
    result_df["normal_phase_distance_mad"] = normal_distance_mad
    # The column name is kept explicit for the requested initial 90th-percentile
    # setting. ``phase_distance_percentile`` remains configurable.
    result_df["normal_phase_distance_percentile_90"] = normal_distance_threshold

    curve_rows = [
        _phase_curve_distances(
            row[phase_columns].to_numpy(dtype=float),
            template_values,
            phase_voltage_grid,
        )
        for _, row in result_df.iterrows()
    ]
    curve_df = pd.DataFrame(
        curve_rows,
        columns=[
            "phase_curve_median_distance",
            "phase_curve_max_distance",
            "phase_curve_area_distance",
            "phase_curve_valid_point_count",
        ],
        index=result_df.index,
    )
    result_df[curve_df.columns] = curve_df
    # When all normal curves are identical, MAD is zero. The small documented
    # epsilon prevents division by zero while preserving a very sensitive z.
    safe_mad = (
        max(normal_distance_mad, phase_mad_epsilon)
        if np.isfinite(normal_distance_mad)
        else np.nan
    )
    result_df["phase_curve_robust_z"] = (
        result_df["phase_curve_median_distance"] - normal_distance_median
    ) / safe_mad
    return result_df


def classify_detected_eaps(
    spike_feature_df: pd.DataFrame,
    baseline_threshold_mV: float = DEFAULT_EAP_BASELINE_THRESHOLD_MV,
    baseline_difference_threshold_mV: float = (
        DEFAULT_EAP_BASELINE_DIFFERENCE_THRESHOLD_MV
    ),
    phase_threshold_multiplier: float = DEFAULT_EAP_PHASE_THRESHOLD_MULTIPLIER,
    minimum_amplitude_mV: float = DEFAULT_EAP_MINIMUM_AMPLITUDE_MV,
    minimum_evidence_count: int = DEFAULT_EAP_MINIMUM_EVIDENCE_COUNT,
) -> pd.DataFrame:
    """Create the one algorithmic ``detected_eap`` classification.

    A Detected EAP must occur before or after the injection, have amplitude at
    least 60 mV, and pass at least two available evidence rules. Manual labels
    are not read by this function.
    """

    result_df = spike_feature_df.copy()
    result_df["baseline_shift_from_normal_mV"] = (
        result_df["normal_median_baseline_mV"] - result_df["baseline_mV"]
    )
    for threshold in (20, 50, 100):
        result_df[f"phase_shift_dvdt{threshold}_mV"] = (
            result_df[f"normal_median_voltage_at_dvdt{threshold}_mV"]
            - result_df[f"voltage_at_dvdt{threshold}_mV"]
        )
    result_df["foot_duration_ratio_vs_normal"] = (
        result_df["foot_duration_20_to_100_ms"]
        / result_df["normal_median_foot_duration_20_to_100_ms"]
    )
    result_df.loc[
        result_df["normal_median_foot_duration_20_to_100_ms"] <= 0,
        "foot_duration_ratio_vs_normal",
    ] = np.nan

    result_df["baseline_feature_available"] = (
        result_df["baseline_mV"].notna()
        & result_df["amp_from_baseline_mV"].notna()
    )
    result_df["baseline_reference_available"] = (
        result_df["baseline_reference_available"].fillna(False)
        & result_df["normal_median_baseline_mV"].notna()
    )
    result_df["phase_features_available"] = (
        result_df["phase_features_available"].fillna(False)
        & result_df["phase_curve_median_distance"].notna()
    )
    result_df["phase_reference_available"] = (
        result_df["phase_reference_available"].fillna(False)
        & result_df["normal_phase_distance_percentile_90"].notna()
    )

    result_df["rule_eap_valid_timing"] = result_df["timing_region"].isin(
        ["before_step", "after_step"]
    )
    result_df["rule_eap_amplitude"] = (
        result_df["amp_from_baseline_mV"] >= minimum_amplitude_mV
    )
    # Save every applied threshold beside the decision. Diagnostic plots and
    # exports should read these columns instead of reimplementing the rule.
    result_df["eap_minimum_amplitude_mV"] = minimum_amplitude_mV
    result_df["eap_baseline_threshold_mV"] = baseline_threshold_mV
    result_df["eap_baseline_difference_threshold_mV"] = (
        baseline_difference_threshold_mV
    )
    result_df["eap_phase_threshold_multiplier"] = (
        phase_threshold_multiplier
    )
    result_df["eap_original_phase_threshold"] = result_df[
        "normal_phase_distance_percentile_90"
    ]
    result_df["eap_applied_phase_threshold"] = (
        result_df["eap_original_phase_threshold"]
        * result_df["eap_phase_threshold_multiplier"]
    )

    result_df["rule_eap_negative_baseline"] = (
        result_df["baseline_mV"] <= result_df["eap_baseline_threshold_mV"]
    )
    result_df["rule_eap_baseline_shift"] = (
        result_df["baseline_shift_from_normal_mV"]
        >= result_df["eap_baseline_difference_threshold_mV"]
    )
    result_df["rule_eap_phase_space_abnormal"] = (
        result_df["phase_reference_available"]
        & result_df["phase_features_available"]
        & (
            result_df["phase_curve_median_distance"]
            > result_df["eap_applied_phase_threshold"]
        )
    )
    evidence_rule_columns = [
        "rule_eap_negative_baseline",
        "rule_eap_baseline_shift",
        "rule_eap_phase_space_abnormal",
    ]
    result_df["eap_evidence_score"] = (
        result_df[evidence_rule_columns].fillna(False).astype(int).sum(axis=1)
    )
    result_df["rule_eap_sufficient_evidence"] = (
        result_df["eap_evidence_score"] >= minimum_evidence_count
    )

    result_df["eap_available_evidence_count"] = (
        result_df["baseline_feature_available"].astype(int)
        + (
            result_df["baseline_feature_available"]
            & result_df["baseline_reference_available"]
        ).astype(int)
        + (
            result_df["phase_features_available"]
            & result_df["phase_reference_available"]
        ).astype(int)
    )
    result_df["detected_eap"] = (
        result_df["rule_eap_valid_timing"]
        & result_df["rule_eap_amplitude"]
        & result_df["rule_eap_sufficient_evidence"]
    )
    insufficient = result_df["eap_available_evidence_count"] < 2
    result_df.loc[insufficient, "detected_eap"] = False
    result_df.loc[result_df["timing_region"] == "during_step", "detected_eap"] = False
    result_df["needs_manual_review"] = insufficient
    result_df["classification_qc"] = "classified"
    result_df.loc[insufficient, "classification_qc"] = (
        "insufficient_evidence_features"
    )
    return result_df


def infer_current_step_amplitudes_by_sweep(
    abf,
    command_channel: int = 1,
    step_start_sec: float | None = None,
    step_stop_sec: float | None = None,
) -> dict[int, float]:
    """Infer command current amplitude for every sweep from the command waveform.

    ``command_channel`` is a zero-based internal pyABF index.

    This is better than assuming a fixed formula because different ABF files
    can use different starting currents or step durations.
    """

    current_by_sweep = {}

    for sweep_number in range(abf.sweepCount):
        abf.setSweep(sweepNumber=sweep_number, channel=command_channel)
        time_sec = abf.sweepX.copy()
        command_pA = abf.sweepC.copy()

        if step_start_sec is None or step_stop_sec is None:
            baseline_pA = float(np.median(command_pA[: max(1, int(0.05 * abf.dataRate))]))
            step_pA = float(np.max(command_pA) if np.max(command_pA) > abs(np.min(command_pA)) else np.min(command_pA))
            current_by_sweep[int(sweep_number)] = step_pA - baseline_pA
            continue

        baseline_mask = time_sec < step_start_sec
        step_mask = (time_sec >= step_start_sec) & (time_sec <= step_stop_sec)

        baseline_pA = float(np.median(command_pA[baseline_mask])) if np.any(baseline_mask) else 0.0
        step_pA = float(np.median(command_pA[step_mask])) if np.any(step_mask) else baseline_pA
        current_by_sweep[int(sweep_number)] = step_pA - baseline_pA

    return current_by_sweep


def plot_full_sweep_eap_trace(
    abf,
    spike_feature_df: pd.DataFrame,
    sweep_number: int,
    channel: int = 2,
    eap_column: str = "detected_eap",
    step_start_sec: float | None = None,
    step_stop_sec: float | None = None,
    save_path: str | Path | None = None,
):
    """Plot one complete sweep and mark normal APs and Detected EAPs.

    ``channel`` is a zero-based internal index; the title converts it to the
    one-based number shown to users.

    The complete sweep remains visible from start to finish. Normal APs are
    blue markers, while rows passing the selected EAP rule are red markers.
    """

    if eap_column not in spike_feature_df.columns:
        raise ValueError(f"Column '{eap_column}' is not in the feature table.")

    abf.setSweep(sweepNumber=int(sweep_number), channel=int(channel))
    time_sec = abf.sweepX.copy()
    voltage_mV = abf.sweepY.copy()

    sweep_df = spike_feature_df[
        spike_feature_df["sweep"] == int(sweep_number)
    ].copy()
    eap_df = sweep_df[sweep_df[eap_column]].copy()
    normal_df = sweep_df[~sweep_df[eap_column]].copy()

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(
        time_sec,
        voltage_mV,
        color="black",
        linewidth=0.55,
        label="Voltage trace",
        zorder=1,
    )

    if len(normal_df) > 0:
        ax.scatter(
            normal_df["peak_time_sec"],
            normal_df["peak_voltage_mV"],
            color="tab:blue",
            s=14,
            alpha=0.65,
            label="Normal AP",
            zorder=3,
        )

    if len(eap_df) > 0:
        ax.scatter(
            eap_df["peak_time_sec"],
            eap_df["peak_voltage_mV"],
            color="red",
            edgecolors="black",
            linewidths=0.5,
            s=60,
            label="Detected EAP",
            zorder=4,
        )

    if step_start_sec is not None:
        ax.axvline(
            step_start_sec,
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Current step start",
        )

    if step_stop_sec is not None:
        ax.axvline(
            step_stop_sec,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label="Current step end",
        )

    # Explicitly keep the full recording visible rather than zooming around
    # the detected spikes.
    ax.set_xlim(float(time_sec[0]), float(time_sec[-1]))
    ax.set_title(
        f"Sweep {sweep_number}, Channel {channel + 1}: full trace with "
        f"{len(eap_df)} Detected EAP(s)"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Membrane potential (mV)")
    ax.grid(True, alpha=0.25)
    # Keep the legend outside the data so it cannot hide an EAP marker near
    # the end of the sweep.
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax


def run_peak_based_pipeline(
    abf,
    channel: int = 2,
    step_start_sec: float = 0.11565,
    step_stop_sec: float = 1.11565,
    confirmed_eap_sweeps: list[int] | None = None,
    current_step_pA_by_sweep: dict[int, float] | None = None,
    matched_protocol_reference_df: pd.DataFrame | None = None,
    phase_pre_peak_window_ms: float = 5.0,
    phase_post_peak_window_ms: float = 6.0,
    savgol_window_ms: float = 0.25,
    savgol_polyorder: int = 2,
    phase_voltage_grid: np.ndarray | None = None,
    phase_distance_percentile: float = 90.0,
    min_peak_voltage_mV: float = -5.0,
    min_peak_prominence_mV: float = 20.0,
    min_peak_distance_ms: float = 1.2,
    eap_minimum_amplitude_mV: float = DEFAULT_EAP_MINIMUM_AMPLITUDE_MV,
    eap_baseline_threshold_mV: float = DEFAULT_EAP_BASELINE_THRESHOLD_MV,
    eap_baseline_difference_threshold_mV: float = (
        DEFAULT_EAP_BASELINE_DIFFERENCE_THRESHOLD_MV
    ),
    eap_phase_threshold_multiplier: float = DEFAULT_EAP_PHASE_THRESHOLD_MULTIPLIER,
    eap_minimum_evidence_count: int = DEFAULT_EAP_MINIMUM_EVIDENCE_COUNT,
) -> dict:
    """Run the pipeline using a zero-based internal pyABF channel index."""

    if confirmed_eap_sweeps is None:
        confirmed_eap_sweeps = []

    peak_spikes_df = detect_spikes_all_sweeps_peak_based(
        abf=abf,
        channel=channel,
        min_peak_voltage_mV=min_peak_voltage_mV,
        min_peak_prominence_mV=min_peak_prominence_mV,
        min_peak_distance_ms=min_peak_distance_ms,
    )

    feature_df = build_peak_based_feature_table(
        abf=abf,
        peak_spikes_df=peak_spikes_df,
        step_start_sec=step_start_sec,
        step_stop_sec=step_stop_sec,
        confirmed_eap_sweeps=confirmed_eap_sweeps,
        current_step_pA_by_sweep=current_step_pA_by_sweep,
        phase_pre_peak_window_ms=phase_pre_peak_window_ms,
        phase_post_peak_window_ms=phase_post_peak_window_ms,
        savgol_window_ms=savgol_window_ms,
        savgol_polyorder=savgol_polyorder,
        phase_voltage_grid=phase_voltage_grid,
    )
    feature_df = add_normal_references(
        spike_feature_df=feature_df,
        step_start_sec=step_start_sec,
        step_stop_sec=step_stop_sec,
        phase_voltage_grid=phase_voltage_grid,
        matched_protocol_reference_df=matched_protocol_reference_df,
        phase_distance_percentile=phase_distance_percentile,
    )
    feature_df = classify_detected_eaps(
        feature_df,
        baseline_threshold_mV=eap_baseline_threshold_mV,
        baseline_difference_threshold_mV=(
            eap_baseline_difference_threshold_mV
        ),
        phase_threshold_multiplier=eap_phase_threshold_multiplier,
        minimum_amplitude_mV=eap_minimum_amplitude_mV,
        minimum_evidence_count=eap_minimum_evidence_count,
    )

    detected_eap_df = feature_df[feature_df["detected_eap"]].copy()

    return {
        "peak_spikes_df": peak_spikes_df,
        "feature_df": feature_df,
        "detected_eap_df": detected_eap_df,
    }
