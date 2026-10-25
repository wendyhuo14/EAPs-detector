"""Plot normal APs and Detected EAPs from the same sweep.

These plots are for visual validation of the rule-based EAP detector.

For each selected sweep, the code compares:
- normal APs during the current step
- algorithm-detected EAPs outside the current step

The plots show voltage vs time and normalized rising phase-space trajectories.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _channel_index_from_spike_row(spike_row: pd.Series) -> int:
    """Return a zero-based index from an internal row or exported CSV row.

    Internal feature tables use ``channel_index``. Public CSV files use the
    one-based ``channel`` field. Supporting both keeps these plotting helpers
    consistent without ever passing a one-based value directly to pyABF.
    """

    has_internal = (
        "channel_index" in spike_row.index
        and pd.notna(spike_row["channel_index"])
    )
    has_exported = "channel" in spike_row.index and pd.notna(spike_row["channel"])

    if not has_internal and not has_exported:
        raise ValueError("Spike row is missing channel_index or channel.")

    if has_exported:
        user_channel = float(spike_row["channel"])
        if user_channel < 1 or not user_channel.is_integer():
            raise ValueError("Exported channel must be a positive one-based integer.")
        exported_index = int(user_channel) - 1
    else:
        exported_index = None

    if has_internal:
        internal_index = int(spike_row["channel_index"])
        if internal_index < 0:
            raise ValueError("Internal channel_index cannot be negative.")
        if exported_index is not None and exported_index != internal_index:
            raise ValueError("channel and channel_index describe different channels.")
        return internal_index

    return int(exported_index)


def format_eap_decision_details(spike_row: pd.Series) -> str:
    """Format the exact classifier values used for one EAP decision.

    Threshold columns are written by ``classify_detected_eaps()``. Reading
    them here avoids applying the phase multiplier again in a plot.
    """

    return (
        f"t={float(spike_row['peak_time_sec']):.6f} s | "
        f"B: {float(spike_row['baseline_mV']):.3f} <= "
        f"{float(spike_row['eap_baseline_threshold_mV']):g} "
        f"({bool(spike_row['rule_eap_negative_baseline'])}) | "
        f"D: {float(spike_row['baseline_shift_from_normal_mV']):.3f} >= "
        f"{float(spike_row['eap_baseline_difference_threshold_mV']):g} "
        f"({bool(spike_row['rule_eap_baseline_shift'])}) | "
        f"phase: {float(spike_row['phase_curve_median_distance']):.6f} > "
        f"{float(spike_row['eap_applied_phase_threshold']):.6f} "
        f"= {float(spike_row['eap_original_phase_threshold']):.6f} x "
        f"{float(spike_row['eap_phase_threshold_multiplier']):g} "
        f"({bool(spike_row['rule_eap_phase_space_abnormal'])}) | "
        f"score {int(spike_row['eap_evidence_score'])}/3"
    )


def plot_eap_event_diagnostic(
    abf,
    spike_row: pd.Series,
    save_path: str | Path | None = None,
    status_text: str = "classifier decision",
):
    """Plot one event using the exact decision values saved by the classifier."""

    channel_index = _channel_index_from_spike_row(spike_row)
    abf.setSweep(
        sweepNumber=int(spike_row["sweep"]),
        channel=channel_index,
    )
    time_sec = abf.sweepX.copy()
    voltage_mV = abf.sweepY.copy()
    peak_time_sec = float(spike_row["peak_time_sec"])
    peak_index = int(spike_row["peak_index"])
    waveform_mask = (
        (time_sec >= peak_time_sec - 0.008)
        & (time_sec <= peak_time_sec + 0.005)
    )
    baseline_mask = (
        (time_sec >= peak_time_sec - 0.005)
        & (time_sec <= peak_time_sec - 0.003)
    )

    phase_columns = sorted(
        [column for column in spike_row.index if column.startswith("phase_dvdt_norm_at_v")],
        key=lambda column: int(column.rsplit("v", 1)[1]),
    )
    template_columns = [
        column.replace(
            "phase_dvdt_norm_at_", "normal_phase_template_dvdt_norm_at_"
        )
        for column in phase_columns
    ]
    normalized_voltage = np.array(
        [int(column.rsplit("v", 1)[1]) / 100 for column in phase_columns]
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5.4),
        gridspec_kw={"width_ratios": [1.4, 1.15, 1]},
    )
    axes[0].plot(
        (time_sec[waveform_mask] - peak_time_sec) * 1000,
        voltage_mV[waveform_mask],
        color="black",
        linewidth=1,
    )
    axes[0].scatter(
        (time_sec[baseline_mask] - peak_time_sec) * 1000,
        voltage_mV[baseline_mask],
        color="tab:orange",
        s=14,
        label="raw -5 to -3 ms samples",
        zorder=3,
    )
    axes[0].scatter([0], [voltage_mV[peak_index]], color="red", s=35, zorder=4)
    axes[0].axhline(
        float(spike_row["baseline_mV"]), color="tab:orange", linestyle="--"
    )
    axes[0].set_xlabel("Time from peak (ms)")
    axes[0].set_ylabel("Membrane potential (mV)")
    axes[0].set_title("Waveform and baseline window")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    if phase_columns and all(column in spike_row.index for column in template_columns):
        axes[1].plot(
            normalized_voltage,
            spike_row[template_columns].to_numpy(dtype=float),
            color="tab:blue",
            linewidth=2,
            label="normal template",
        )
        axes[1].plot(
            normalized_voltage,
            spike_row[phase_columns].to_numpy(dtype=float),
            color="red",
            linewidth=2,
            label="event",
        )
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Normalized voltage")
    axes[1].set_ylabel("Normalized dV/dt")
    axes[1].set_title("Normalized rising phase-space")
    axes[1].grid(alpha=0.2)

    axes[2].axis("off")
    decision_text = (
        f"File: {spike_row.get('file_id', 'unknown')}\n"
        f"Sweep: {int(spike_row['sweep'])} (zero-based)\n"
        f"Channel: {channel_index + 1} (one-based)\n"
        f"Peak: {peak_time_sec:.6f} s\n"
        f"Status: {status_text}\n\n"
        f"Parameters\n"
        f"B = {float(spike_row['eap_baseline_threshold_mV']):g} mV\n"
        f"D = {float(spike_row['eap_baseline_difference_threshold_mV']):g} mV\n"
        f"M = {float(spike_row['eap_phase_threshold_multiplier']):g}\n\n"
        f"Evidence ({int(spike_row['eap_evidence_score'])}/3)\n"
        f"1 baseline: {float(spike_row['baseline_mV']):.4f} <= "
        f"{float(spike_row['eap_baseline_threshold_mV']):g}: "
        f"{bool(spike_row['rule_eap_negative_baseline'])}\n"
        f"2 difference: {float(spike_row['baseline_shift_from_normal_mV']):.4f} >= "
        f"{float(spike_row['eap_baseline_difference_threshold_mV']):g}: "
        f"{bool(spike_row['rule_eap_baseline_shift'])}\n"
        f"3 phase: {float(spike_row['phase_curve_median_distance']):.6f} > "
        f"{float(spike_row['eap_applied_phase_threshold']):.6f}: "
        f"{bool(spike_row['rule_eap_phase_space_abnormal'])}\n"
        f"   original {float(spike_row['eap_original_phase_threshold']):.6f} "
        f"x {float(spike_row['eap_phase_threshold_multiplier']):g}\n\n"
        f"Detected EAP: {bool(spike_row['detected_eap'])}\n"
        f"Timing: {spike_row['timing_region']}\n"
        f"Amplitude: {float(spike_row['amp_from_baseline_mV']):.3f} mV"
    )
    axes[2].text(0, 1, decision_text, va="top", family="monospace", fontsize=9.5)
    fig.suptitle(
        f"{spike_row.get('file_id', 'unknown')} | sweep {int(spike_row['sweep'])} | "
        f"channel {channel_index + 1} | "
        f"{peak_time_sec:.6f} s | production defaults",
        fontsize=12,
    )
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, axes


def _extract_spike_window(
    abf,
    spike_row,
    pre_peak_window_ms: float,
    post_peak_window_ms: float,
) -> dict:
    """Extract a short voltage window around one spike peak."""

    sweep_number = int(spike_row["sweep"])
    channel = _channel_index_from_spike_row(spike_row)
    peak_time_sec = float(spike_row["peak_time_sec"])

    abf.setSweep(sweepNumber=sweep_number, channel=channel)
    time_sec = abf.sweepX.copy()
    voltage_mV = abf.sweepY.copy()
    sampling_rate_hz = abf.dataRate

    peak_index = int(np.argmin(np.abs(time_sec - peak_time_sec)))

    pre_samples = int(round(pre_peak_window_ms / 1000 * sampling_rate_hz))
    post_samples = int(round(post_peak_window_ms / 1000 * sampling_rate_hz))

    start_index = max(0, peak_index - pre_samples)
    stop_index = min(len(voltage_mV), peak_index + post_samples + 1)

    window_time_sec = time_sec[start_index:stop_index]
    window_voltage_mV = voltage_mV[start_index:stop_index]

    relative_time_ms = (window_time_sec - time_sec[peak_index]) * 1000

    return {
        "relative_time_ms": relative_time_ms,
        "voltage_mV": window_voltage_mV,
        "sampling_rate_hz": sampling_rate_hz,
        "peak_index_inside_window": peak_index - start_index,
    }


def _normalized_phase_curve_from_row(spike_row) -> tuple[np.ndarray, np.ndarray]:
    """Read the transparent normalized rising-curve columns from one row."""

    phase_columns = sorted(
        [
            column
            for column in spike_row.index
            if column.startswith("phase_dvdt_norm_at_v")
        ],
        key=lambda column: int(column.rsplit("v", 1)[1]),
    )
    normalized_voltage = np.array(
        [int(column.rsplit("v", 1)[1]) / 100 for column in phase_columns],
        dtype=float,
    )
    normalized_dvdt = spike_row[phase_columns].to_numpy(dtype=float)
    valid_mask = np.isfinite(normalized_voltage) & np.isfinite(normalized_dvdt)
    return normalized_voltage[valid_mask], normalized_dvdt[valid_mask]


def select_same_sweep_comparison_spikes(
    spike_feature_df: pd.DataFrame,
    sweep_number: int,
    max_normal_spikes: int = 5,
    eap_column: str = "detected_eap",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select clean normal APs and Detected EAPs from the same sweep.

    Normal APs are selected from during-step spikes in the same sweep.
    If there are many normal APs, the code picks evenly spaced examples so
    the plot stays readable.
    """

    sweep_df = spike_feature_df[spike_feature_df["sweep"] == int(sweep_number)].copy()

    detected_eap_rows = sweep_df[sweep_df[eap_column]].copy()
    normal_df = sweep_df[
        (sweep_df["timing_region"] == "during_step")
        & (~sweep_df[eap_column])
    ].copy()
    if "phase_reference_event_qc_pass" in normal_df.columns:
        normal_df = normal_df[normal_df["phase_reference_event_qc_pass"]].copy()

    detected_eap_rows = detected_eap_rows.sort_values("peak_time_sec")
    normal_df = normal_df.sort_values("peak_time_sec")

    if len(normal_df) > max_normal_spikes:
        # Pick evenly spaced normal spikes across the sweep.
        selected_positions = np.linspace(0, len(normal_df) - 1, max_normal_spikes)
        selected_positions = np.round(selected_positions).astype(int)
        normal_df = normal_df.iloc[selected_positions].copy()

    return normal_df, detected_eap_rows


def plot_same_sweep_normal_vs_eap(
    abf,
    spike_feature_df: pd.DataFrame,
    sweep_number: int,
    max_normal_spikes: int = 5,
    pre_peak_window_ms: float = 5.0,
    post_peak_window_ms: float = 6.0,
    save_path: str | None = None,
    eap_column: str = "detected_eap",
    selected_eap_row: pd.Series | None = None,
):
    """Plot normal APs and Detected EAPs from the same sweep.

    The left plot shows voltage vs time aligned to each spike peak.
    The right plot shows rising-phase phase-space trajectories.
    """

    if selected_eap_row is None:
        normal_df, detected_eap_rows = select_same_sweep_comparison_spikes(
            spike_feature_df=spike_feature_df,
            sweep_number=sweep_number,
            max_normal_spikes=max_normal_spikes,
            eap_column=eap_column,
        )
    else:
        # For a focused comparison, show one selected EAP while keeping
        # several normal APs from the same sweep as references.
        same_sweep_df = spike_feature_df[
            spike_feature_df["sweep"] == int(sweep_number)
        ].copy()
        normal_df = same_sweep_df[
            (same_sweep_df["timing_region"] == "during_step")
            & (~same_sweep_df[eap_column])
        ].sort_values("peak_time_sec")
        if "phase_reference_event_qc_pass" in normal_df.columns:
            normal_df = normal_df[normal_df["phase_reference_event_qc_pass"]].copy()

        if len(normal_df) > max_normal_spikes:
            selected_positions = np.linspace(
                0,
                len(normal_df) - 1,
                max_normal_spikes,
            )
            selected_positions = np.round(selected_positions).astype(int)
            normal_df = normal_df.iloc[selected_positions].copy()

        detected_eap_rows = pd.DataFrame([selected_eap_row])

    if len(detected_eap_rows) == 0:
        raise ValueError(f"Sweep {sweep_number} has no Detected EAPs to compare.")

    if len(normal_df) == 0:
        raise ValueError(f"Sweep {sweep_number} has no during-step normal APs to compare.")

    channel_number = _channel_index_from_spike_row(detected_eap_rows.iloc[0]) + 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for normal_number, (_, normal_row) in enumerate(normal_df.iterrows(), start=1):
        spike_window = _extract_spike_window(
            abf=abf,
            spike_row=normal_row,
            pre_peak_window_ms=pre_peak_window_ms,
            post_peak_window_ms=post_peak_window_ms,
        )
        normalized_voltage, normalized_dvdt = _normalized_phase_curve_from_row(
            normal_row
        )

        label = "normal during-step AP" if normal_number == 1 else None
        axes[0].plot(
            spike_window["relative_time_ms"],
            spike_window["voltage_mV"],
            color="tab:blue",
            alpha=0.35,
            linewidth=1.2,
            label=label,
        )
        axes[1].plot(
            normalized_voltage,
            normalized_dvdt,
            color="tab:blue",
            alpha=0.35,
            linewidth=1.2,
            label=label,
        )

    for eap_number, (_, eap_row) in enumerate(detected_eap_rows.iterrows(), start=1):
        spike_window = _extract_spike_window(
            abf=abf,
            spike_row=eap_row,
            pre_peak_window_ms=pre_peak_window_ms,
            post_peak_window_ms=post_peak_window_ms,
        )
        normalized_voltage, normalized_dvdt = _normalized_phase_curve_from_row(
            eap_row
        )

        label = "Detected EAP" if eap_number == 1 else None
        axes[0].plot(
            spike_window["relative_time_ms"],
            spike_window["voltage_mV"],
            color="red",
            alpha=0.9,
            linewidth=2.5,
            label=label,
        )
        axes[1].plot(
            normalized_voltage,
            normalized_dvdt,
            color="red",
            alpha=0.9,
            linewidth=2.5,
            label=label,
        )

    axes[0].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title(
        f"Sweep {sweep_number}, Channel {channel_number}: voltage vs time"
    )
    axes[0].set_xlabel("Time relative to peak (ms)")
    axes[0].set_ylabel("Membrane potential (mV)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title(
        f"Sweep {sweep_number}, Channel {channel_number}: normalized rising phase-space"
    )
    axes[1].set_xlabel("Normalized membrane potential")
    axes[1].set_ylabel("Normalized rising dV/dt")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        f"Normal APs vs Detected EAPs in sweep {sweep_number}, "
        f"Channel {channel_number}",
        fontsize=14,
        y=1.03,
    )
    decision_lines = [
        format_eap_decision_details(eap_row)
        for _, eap_row in detected_eap_rows.iterrows()
    ]
    fig.text(
        0.01,
        -0.02,
        "Classifier decision details (row-specific phase threshold; multiplier applied once):\n"
        + "\n".join(decision_lines),
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
    )
    plt.tight_layout(rect=(0, min(0.22, 0.08 + 0.035 * len(decision_lines)), 1, 1))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, axes, normal_df, detected_eap_rows


def choose_random_detected_eap_for_comparison(
    spike_feature_df: pd.DataFrame,
    eap_column: str = "detected_eap",
    random_seed: int = 42,
) -> pd.Series | None:
    """Choose one reproducible random Detected EAP with same-sweep normals.

    A fixed seed makes the selected comparison reproducible. Change the
    seed in the runner when a different random example is desired.
    """

    if eap_column not in spike_feature_df.columns:
        raise ValueError(f"Column '{eap_column}' is not in the feature table.")

    candidate_df = spike_feature_df[
        spike_feature_df[eap_column]
    ].reset_index(drop=True)

    eligible_rows = []
    for _, eap_row in candidate_df.iterrows():
        same_sweep_df = spike_feature_df[
            spike_feature_df["sweep"] == int(eap_row["sweep"])
        ]
        normal_df = same_sweep_df[
            (same_sweep_df["timing_region"] == "during_step")
            & (~same_sweep_df[eap_column])
        ]
        if "phase_reference_event_qc_pass" in normal_df.columns:
            normal_df = normal_df[normal_df["phase_reference_event_qc_pass"]]
        if len(normal_df) > 0:
            eligible_rows.append(eap_row)

    if len(eligible_rows) == 0:
        return None

    eligible_df = pd.DataFrame(eligible_rows).reset_index(drop=True)
    random_generator = np.random.default_rng(random_seed)
    selected_position = int(random_generator.integers(len(eligible_df)))
    return eligible_df.iloc[selected_position].copy()
