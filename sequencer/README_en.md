# sequencer

A real-time visualization tool for exhibitions (codename: **Molecule Caller**)
that simulates the process of nanopore-sequencer signal reading → base
calling → assembly.

While streaming synthesized waveform data, it visualizes the following in
real time in a single-window dashboard.

- **Real-time signal**: the waveform (Raw / Assigned Signal), an Events track
  (color-coded blocks), pills showing the most recently decoded sequence, and
  a Window slider to change how much time is shown
- **Current call**: the code currently being read with a confidence bar, plus
  a confidence ranking (shown with a warning color when reading a
  hard-to-distinguish pair)
- **Sequence assembly**: mean consensus accuracy (with a trend arrow), Phred Q
  score, consensus trace, depth chart, and yield graph
- **KPI tiles**: Reads / Yield / Mean Q / Pass rate
- **Top bar**: run selector, LIVE indicator, elapsed time, Pause/Stop (Stop
  shows a confirmation dialog and then quits the app)
- **Export**: saves a screenshot of the current screen, plus a text file with
  the timestamp, sequence info, and the detection time of each code, into the
  `copy/` folder
- **Settings**: lets you pick a different batch (a full set of generated data)
  from `seq_data/` and restarts the app with it
- **Docs**: view this README in Japanese or English from the sidebar

## Directory structure

```
sequencer/
├── batch_generate.py              # Script that generates synthetic waveform data
├── signal_formation.py            # Core functions for waveform generation / noise
├── sequence_stream_pyqtgraph.py   # Dashboard app (PyQtGraph-based)
├── history/                       # Archive of past versions
├── assets/
│   └── jin_mark_white.png         # Logo mark shown in the sidebar (app still works without it)
├── README.md / README_en.md       # This file (Japanese / English)
├── copy/                          # Output destination for the Export feature (not tracked in Git)
└── seq_data/                      # Output destination for batch_generate.py (not tracked in Git)
    ├── _latest_batch.txt          # Name of the auto-selected "latest batch" folder
    ├── _selected_batch.txt        # Batch explicitly picked via Settings (takes priority if present)
    └── {experiment}_{sample}_{sequence_name}_{timestamp}/
        ├── manifest.csv
        └── *.csv
```

## Usage

### 1. Generate data

```bash
uv run batch_generate.py
```

The sequence and its identifying labels can be set via the parameters at the
top of the script.

```python
sequence = 'GADGVGKSAL'          # The actual sequence (string)
sequence_name = 'sample_seqA'    # A human-readable name for the sequence (blank = unset)
sample_name = ''                 # Sample name (blank = unset)
experiment_name = 'exp01'        # Experiment name (blank = unset)
```

Running it produces a waveform CSV for each run (including a `Code` column)
and a `manifest.csv` inside
`seq_data/{experiment}_{sample}_{sequence_name}_{timestamp}/`, and updates
`seq_data/_latest_batch.txt` (so the visualizer automatically loads this
batch the next time it starts).

Sampling: 1 data point = 0.1 ms (10 kHz).

### 2. Run the dashboard

```bash
uv run sequence_stream_pyqtgraph.py
```

With no configuration, it automatically loads the most recently generated
batch. To view an older batch, either pick it from **Settings** inside the
app, or set `BATCH_DIR_OVERRIDE` to the folder name near the top of the
script.

## Requirements

```
pyqtgraph
PyQt5  (or PyQt6)
numpy
pandas
```

## Notes

- The generated/output data under `seq_data/` and `copy/` constitutes
  experimental data, so per the repository's development policy it is not
  tracked in Git (adding it to `.gitignore` is recommended).
- A small compatibility layer is included around Qt's enum types so the app
  runs under both PyQt5 and PyQt6 (which one is actually used depends on the
  environment).
- The layout assumes a dark theme and a color scheme suited to exhibition
  use.
- If `assets/jin_mark_white.png` is not found, the app still starts normally
  without the logo (it prints whether the file was found or not at startup).
