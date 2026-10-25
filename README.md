# Rule-based ectopic action potential screening

## Introduction and purpose

This Python tool screens for potential ectopic action potentials (EAPs) in
current-clamp electrophysiology recordings stored in Axon Binary Format
(`.abf`). Unlike typical action potentials, which begin at the axon initial
segment, EAPs can originate at more distal sites along the axon. Persistent
ectopic firing has been observed in parvalbumin-expressing inhibitory
interneurons in the mouse neocortex [1]. Investigating this firing may help
clarify how these neurons regulate excessive activity associated with epilepsy
[2].

This tool was developed as part of research in the Theyel Lab at Brown
University, under the supervision of Dr. Brian Theyel.

The program detects action-potential peaks, measures waveform features, and
flags events that meet an explicit rule-based definition of a **Detected EAP
candidate**. It does not use machine learning or artificial intelligence. A
Detected EAP candidate is a screening result, not definitive proof that an event is
biologically ectopic. 

## Detection criteria

By default, the program identifies action-potential peaks with a voltage of at
least `-5 mV`, prominence of at least `20 mV`, and separation of at least
`1.2 ms`.

Events are categorized as `before_step`, `during_step`, or `after_step`
according to the configured current-injection period. Only before-step and
after-step events are eligible for EAP classification.

Suitable during-step action potentials provide normal references. Baseline
comparisons prefer normal events from the same sweep and can fall back to
suitable events from the same file. Phase-space references are built from
suitable during-step events in the same file. The pre-spike baseline is the
median membrane voltage measured 5 to 3 ms before each peak.

An eligible event is classified as a Detected EAP candidate if its
baseline-to-peak amplitude is at least `60 mV` and it meets at least two of
three evidence checks:

1. **Negative baseline:** baseline voltage `<= -55 mV`.
2. **Baseline difference:** baseline at least `3 mV` more negative than the
   selected normal baseline reference.
3. **Phase-space difference:** normalized rising phase-space distance strictly
   greater than the event's normal-reference 90th-percentile distance
   multiplied by `0.9`.

At least five suitable events are required for a reference set. Reference sets
with five through nine events are marked as having a low reference count.

## Performance evaluation

The detector was evaluated on 22 manually inspected recordings from regular-spiking 
neurons in the mouse orbitofrontal cortex and somatosensory cortex, each containing 
an average of approximately 30 EAPs. It achieved approximately 95% precision, 
meaning that 95% of detected events matched manual annotations. These results may 
not generalize to other recording conditions. Users should verify detection results 
after automated analysis.

The validation recordings are not included in this release. Therefore, the actual 
performance of the detector may vary under different experimental conditions.

## Current limitations

### Timing and stimulation protocols

Only events classified as `before_step` or `after_step` are eligible for EAP
detection. Events during stimulation are excluded even if they meet all
waveform criteria. The program supports one continuous current step, with
automatically inferred or user-specified boundaries. It does not model multiple
stimulation epochs. Incorrect boundaries can affect both classification and
reference selection.

### Detection thresholds

By default, action-potential detection requires a peak voltage of at least
`-5 mV`, prominence of at least `20 mV`, and peak separation of at least
`1.2 ms`. These thresholds can exclude smaller or low-prominence events and can
merge closely spaced spikes. The program is not designed to detect EAPs with
peak voltages below `-5 mV` under its default settings. Events with
baseline-to-peak amplitudes below `60 mV` are also excluded from EAP
classification.

### Baseline and reference dependence

The pre-spike baseline can be distorted by nearby spikes, afterpotentials,
drift, or artifacts. Normal references use during-step action potentials,
prioritizing same-sweep baseline comparisons with a same-file fallback.

At least five suitable reference events are required; five through nine are
flagged as a low reference count. Differences in firing rate or membrane state
can make reference events unrepresentative.

Events with fewer than two available evidence sources are excluded from EAP
classification and marked `needs_manual_review=True` in the in-memory feature
table. The current command-line runner does **not** export that full table or
save these manual-review events separately. The only CSV output contains
Detected EAP candidates, so users should not interpret the absence of a saved
event as evidence that it was fully classified.

### Generalizability and interpretation

The program was developed using recordings from regular-spiking neurons in the
mouse orbitofrontal cortex and somatosensory cortex. Its performance has not
been established across other cell types, species, recording preparations,
acquisition systems, or stimulation protocols. Users should validate the
default settings for their own experimental conditions.

## Installation

Download and extract the release candidate, then open a terminal in that
folder. 

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies and check the command-line interface:

```bash
python -m pip install -r requirements.txt
python run_eap_analysis.py --help
```

## Usage

Users must provide their own recordings. No laboratory recordings or example
data are distributed with this software.

Run the detector with paths to an ABF file and a new or empty output directory:

```bash
python run_eap_analysis.py \
  --input /path/to/recording.abf \
  --output /path/to/new_results_directory \
  --voltage-channel 3 \
  --command-channel 2
```

- `--input` is the path to the ABF recording.
- `--output` must be a new or empty directory. The program stops rather than
  mixing results with existing files.
- `--voltage-channel` selects the membrane-voltage channel in millivolts.
- `--command-channel` selects the channel whose current-command waveform is
  available through `pyABF.ABF.sweepC`.

The program channel numbers start at **1**. Channel names and the signals recorded on 
each channel may differ between ABF files, so it is important to verify the selected 
voltage and command traces in an ABF viewer.

The program infers current-step boundaries from the selected command waveform
by default. To provide known boundaries instead:

```bash
python run_eap_analysis.py \
  --input /path/to/recording.abf \
  --output /path/to/new_results_directory \
  --voltage-channel 3 \
  --command-channel 2 \
  --step-start-sec 0.100 \
  --step-stop-sec 1.100
```

Both timing options must be supplied together. When boundaries are inferred,
`--reference-sweep` can select the sweep used for inference; it is a
**zero-based** sweep index and defaults to the last sweep. Run the following
command for all supported options and their exact names:

```bash
python run_eap_analysis.py --help
```

These multiline commands use macOS/Linux shell syntax. In Windows Command
Prompt, enter the complete command on one line without the backslashes.

## Outputs

Results are written to the directory supplied with `--output`.

| Output | Contents and creation condition |
|---|---|
| `Detected_EAPs.csv` | Always created. Contains one row per Detected EAP candidate and the event measurements, selected reference values, applied thresholds, evidence checks, and QC fields used for its decision. It can contain zero rows. It does not contain all detected APs or events flagged only for manual review. |
| `traces/sweep_<number>_Detected_EAP_trace.png` | One full voltage-trace plot for each zero-based sweep containing at least one Detected EAP candidate. The `traces/` directory is omitted when no candidates are detected. |
| `Detected_EAP_vs_normal_phase_space.png` | One reproducibly selected Detected EAP candidate compared with suitable normal during-step APs from the same zero-based sweep. Omitted when there is no detected candidate with at least one suitable same-sweep normal AP. |

User-facing channel numbers in the CLI, plot labels, and exported `channel`
column start at **1**. The exported `sweep` and `peak_index` columns, sweep
numbers in plot filenames, and `--reference-sweep` argument remain
**zero-based**. Units are included in column names where applicable, such as
`_mV`, `_ms`, and `_sec`.

No MATLAB export or separate manual-review-event export is implemented in this
release.

## Project files

- `eap_peak_detection.py`: AP detection, feature measurement, reference
  construction, EAP rules, and full-trace plotting.
- `eap_comparison_plots.py`: same-sweep waveform and phase-space visual checks.
- `run_eap_analysis.py`: portable ABF command-line workflow.
- `requirements.txt`: Python package dependencies and compatible versions.
- `LICENSE`: MIT License terms and copyright notice.

## References

1. Theyel, B. B., Stevenson, R. J., & Connors, B. W. (2024).
   Activity-dependent ectopic spiking in parvalbumin-expressing interneurons of
   the neocortex. *eNeuro, 11*(5), ENEURO.0314-23.2024.
   https://doi.org/10.1523/ENEURO.0314-23.2024

2. Hill, S. F., Liebergall, S. R., Wengert, E. R., Goldberg, E. M., & Theyel,
   B. B. (2025). Interneurons exhibit attenuated ectopic action potential
   firing in a severe neurodevelopmental disorder. *Journal of Neurophysiology,
   133*(6), 1692-1698. https://doi.org/10.1152/jn.00133.2025

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the
full license text.
