"""Run the rule-based EAP screening workflow on one user-supplied ABF file.

This command-line runner contains no recording paths or laboratory data. Run
``python run_eap_analysis.py --help`` to see every configurable option.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pyabf

from eap_comparison_plots import (
    choose_random_detected_eap_for_comparison,
    plot_same_sweep_normal_vs_eap,
)
from eap_peak_detection import (
    DEFAULT_EAP_BASELINE_DIFFERENCE_THRESHOLD_MV,
    DEFAULT_EAP_BASELINE_THRESHOLD_MV,
    DEFAULT_EAP_MINIMUM_AMPLITUDE_MV,
    DEFAULT_EAP_MINIMUM_EVIDENCE_COUNT,
    DEFAULT_EAP_PHASE_THRESHOLD_MULTIPLIER,
    infer_current_step_amplitudes_by_sweep,
    infer_current_step_times,
    plot_full_sweep_eap_trace,
    run_peak_based_pipeline,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface in one easy-to-find place."""

    parser = argparse.ArgumentParser(
        description=(
            "Detect action potentials in an ABF recording, measure transparent "
            "waveform features, and screen for possible EAP candidates."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .abf file.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New or empty directory for generated CSV and PNG files.",
    )
    parser.add_argument(
        "--voltage-channel",
        type=int,
        default=3,
        help=(
            "One-based membrane-voltage channel number, matching the ABF "
            "viewer (default: 3)."
        ),
    )
    parser.add_argument(
        "--command-channel",
        type=int,
        default=2,
        help=(
            "One-based channel whose command waveform is available through "
            "pyABF sweepC (default: 2)."
        ),
    )
    parser.add_argument(
        "--step-start-sec",
        type=float,
        help="Optional known current-step start. Supply together with --step-stop-sec.",
    )
    parser.add_argument(
        "--step-stop-sec",
        type=float,
        help="Optional known current-step stop. Supply together with --step-start-sec.",
    )
    parser.add_argument(
        "--reference-sweep",
        type=int,
        help="Sweep used to infer step boundaries. Defaults to the last sweep.",
    )
    parser.add_argument(
        "--command-change-threshold-pa",
        type=float,
        default=1.0,
        help="Minimum command change used when inferring the current step (default: 1).",
    )

    # Action-potential peak detection. These defaults are the preserved v0.1.0
    # settings; command-line options make the runner portable across protocols.
    parser.add_argument("--min-peak-voltage-mv", type=float, default=-5.0)
    parser.add_argument("--min-peak-prominence-mv", type=float, default=20.0)
    parser.add_argument("--min-peak-distance-ms", type=float, default=1.2)

    # EAP screening defaults. Changing them changes classification behavior and
    # therefore requires validation on the user's own reviewed recordings.
    parser.add_argument(
        "--eap-minimum-amplitude-mv",
        type=float,
        default=DEFAULT_EAP_MINIMUM_AMPLITUDE_MV,
    )
    parser.add_argument(
        "--eap-baseline-threshold-mv",
        type=float,
        default=DEFAULT_EAP_BASELINE_THRESHOLD_MV,
    )
    parser.add_argument(
        "--eap-baseline-difference-threshold-mv",
        type=float,
        default=DEFAULT_EAP_BASELINE_DIFFERENCE_THRESHOLD_MV,
    )
    parser.add_argument(
        "--eap-phase-threshold-multiplier",
        type=float,
        default=DEFAULT_EAP_PHASE_THRESHOLD_MULTIPLIER,
    )
    parser.add_argument(
        "--eap-minimum-evidence-count",
        type=int,
        default=DEFAULT_EAP_MINIMUM_EVIDENCE_COUNT,
    )
    parser.add_argument("--phase-distance-percentile", type=float, default=90.0)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def validate_arguments(args: argparse.Namespace) -> tuple[Path, Path]:
    """Validate paths and paired settings before reading a recording."""

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if input_path.suffix.lower() != ".abf":
        raise ValueError("The input file must have the .abf extension.")
    if (args.step_start_sec is None) != (args.step_stop_sec is None):
        raise ValueError(
            "Supply both --step-start-sec and --step-stop-sec, or supply neither."
        )
    if args.step_start_sec is not None and args.step_start_sec >= args.step_stop_sec:
        raise ValueError("The step start must be earlier than the step stop.")
    if not 0 <= args.phase_distance_percentile <= 100:
        raise ValueError("--phase-distance-percentile must be between 0 and 100.")
    if not 1 <= args.eap_minimum_evidence_count <= 3:
        raise ValueError("--eap-minimum-evidence-count must be between 1 and 3.")

    output_path.mkdir(parents=True, exist_ok=True)
    if any(output_path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_path}\n"
            "Choose a new or empty directory so existing results are not overwritten."
        )
    return input_path, output_path


def create_detected_eap_table(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Return an export table with one-based user-facing channel numbers."""

    detected_eap_df = feature_df[feature_df["detected_eap"]].sort_values(
        ["sweep", "peak_time_sec"]
    ).copy()
    channel_column_position = detected_eap_df.columns.get_loc("channel_index")
    user_channel_numbers = detected_eap_df.pop("channel_index") + 1
    detected_eap_df.insert(
        channel_column_position,
        "channel",
        user_channel_numbers,
    )
    return detected_eap_df


def user_channel_to_index(
    user_channel: int,
    channel_count: int,
    option_name: str,
) -> int:
    """Validate a one-based CLI channel and return its zero-based index."""

    if not 1 <= user_channel <= channel_count:
        raise ValueError(
            f"{option_name} must be between 1 and {channel_count} (one-based); "
            f"received {user_channel}."
        )
    return user_channel - 1


def save_full_trace_plots(
    abf,
    feature_df: pd.DataFrame,
    output_path: Path,
    voltage_channel: int,
    step_start_sec: float,
    step_stop_sec: float,
) -> None:
    """Save one complete trace for every sweep containing a Detected EAP."""

    eap_sweeps = sorted(
        int(value)
        for value in feature_df.loc[feature_df["detected_eap"], "sweep"].unique()
    )
    if not eap_sweeps:
        return

    trace_path = output_path / "traces"
    trace_path.mkdir()
    for sweep_number in eap_sweeps:
        fig, _ = plot_full_sweep_eap_trace(
            abf=abf,
            spike_feature_df=feature_df,
            sweep_number=sweep_number,
            channel=voltage_channel,
            step_start_sec=step_start_sec,
            step_stop_sec=step_stop_sec,
            save_path=trace_path / f"sweep_{sweep_number}_Detected_EAP_trace.png",
        )
        plt.close(fig)


def save_phase_space_comparison(
    abf,
    feature_df: pd.DataFrame,
    output_path: Path,
    random_seed: int,
) -> None:
    """Save one reproducible same-sweep waveform and phase-space comparison."""

    selected_eap = choose_random_detected_eap_for_comparison(
        feature_df,
        random_seed=random_seed,
    )
    if selected_eap is None:
        return

    fig, _, _, _ = plot_same_sweep_normal_vs_eap(
        abf=abf,
        spike_feature_df=feature_df,
        sweep_number=int(selected_eap["sweep"]),
        max_normal_spikes=5,
        pre_peak_window_ms=5.0,
        post_peak_window_ms=6.0,
        save_path=output_path / "Detected_EAP_vs_normal_phase_space.png",
        selected_eap_row=selected_eap,
    )
    plt.close(fig)


def main() -> None:
    """Run the complete public workflow without accessing data on import."""

    args = build_argument_parser().parse_args()
    input_path, output_path = validate_arguments(args)
    abf = pyabf.ABF(str(input_path))

    # CLI and exported channel numbers are one-based. Convert once here, at
    # the input boundary, because pyABF and NumPy require zero-based indices.
    voltage_channel_index = user_channel_to_index(
        args.voltage_channel,
        abf.channelCount,
        "--voltage-channel",
    )
    command_channel_index = user_channel_to_index(
        args.command_channel,
        abf.channelCount,
        "--command-channel",
    )

    if args.step_start_sec is None:
        step_start_sec, step_stop_sec = infer_current_step_times(
            abf,
            command_channel=command_channel_index,
            reference_sweep=args.reference_sweep,
            command_change_threshold_pA=args.command_change_threshold_pa,
        )
    else:
        step_start_sec = float(args.step_start_sec)
        step_stop_sec = float(args.step_stop_sec)

    current_by_sweep = infer_current_step_amplitudes_by_sweep(
        abf,
        command_channel=command_channel_index,
        step_start_sec=step_start_sec,
        step_stop_sec=step_stop_sec,
    )
    analysis = run_peak_based_pipeline(
        abf=abf,
        channel=voltage_channel_index,
        step_start_sec=step_start_sec,
        step_stop_sec=step_stop_sec,
        current_step_pA_by_sweep=current_by_sweep,
        phase_distance_percentile=args.phase_distance_percentile,
        min_peak_voltage_mV=args.min_peak_voltage_mv,
        min_peak_prominence_mV=args.min_peak_prominence_mv,
        min_peak_distance_ms=args.min_peak_distance_ms,
        eap_minimum_amplitude_mV=args.eap_minimum_amplitude_mv,
        eap_baseline_threshold_mV=args.eap_baseline_threshold_mv,
        eap_baseline_difference_threshold_mV=(
            args.eap_baseline_difference_threshold_mv
        ),
        eap_phase_threshold_multiplier=args.eap_phase_threshold_multiplier,
        eap_minimum_evidence_count=args.eap_minimum_evidence_count,
    )
    feature_df = analysis["feature_df"]
    detected_eap_df = create_detected_eap_table(feature_df)
    detected_eap_df.to_csv(output_path / "Detected_EAPs.csv", index=False)

    save_full_trace_plots(
        abf,
        feature_df,
        output_path,
        voltage_channel_index,
        step_start_sec,
        step_stop_sec,
    )
    save_phase_space_comparison(
        abf,
        feature_df,
        output_path,
        args.random_seed,
    )

    print(f"Detected EAP candidates: {len(detected_eap_df)}")
    print(f"Results directory: {output_path}")


if __name__ == "__main__":
    main()
