# sequencer

A real-time visualization tool for exhibitions (codename: **Molecule Caller**)
that simulates the process of nanopore-sequencer signal reading → base
calling → assembly.

While streaming synthesized waveform data, it visualizes the following in
real time:

- The waveform (Raw / Assigned Signal) together with a HUD display of the
  code (amino acid code) currently being read
- A sequence track (genome-browser-style color-coded blocks + per-base QV
  numbers)
- A confidence ranking for the calls (CONFIDENCE panel)
- Assembly results (alignment to the reference sequence, depth, consensus
  accuracy)
- QC metrics such as Phred Q score, N50, depth CV, and pass rate
- Dashboard flourishes such as an accuracy gauge, trend arrows, and warning
  badges

## Directory structure

```
sequencer/
├── batch_generate.py              # Script that generates synthetic waveform data
├── visualize_signals_stream_v4.py # Current version (for exhibitions / production)
├── visualize_signals_stream_v5.py # Experimental / derivative version (needs cleanup)
├── ...
├── visualize_signals_stream_v10.py
└── seq_data/                      # Output destination for batch_generate.py (not tracked in Git)
    ├── manifest.csv
    └── *.csv
```

> **About v5–v10**: Several experimental/derivative versions currently remain
> unorganized. The current finished version is **v4**. Once the role of each
> of the other versions has been sorted out, the unneeded ones will either be
> deleted or moved under `archive/`.

## Usage

### 1. Generate data

```bash
python batch_generate.py
```

This produces `seq_data/manifest.csv` along with a waveform CSV for each run
(each including a `Code` column).

### 2. Run the streaming visualization

```bash
python visualize_signals_stream_v4.py
```

Two windows will open.

- **Figure 1 (MOLECULE CALLER)**: Waveform + sequence track + CONFIDENCE panel
- **Figure 2 (SEQUENCE ASSEMBLY)**: Accuracy gauge + KPI cards + reference
  sequence + consensus trace + depth chart + yield graph

To save the output as a GIF, change `save_as_gif = True` near the top of the
script.

## Requirements

```
numpy
pandas
matplotlib
```

## Notes

- The data generated under `seq_data/` is experimental data, so per the
  repository's development policy it is not tracked in Git (adding it to
  `.gitignore` is recommended).
- The layout assumes a dark theme, neon color scheme, and the larger fonts
  suited to exhibition use. For regular analysis use, consider adjusting the
  font-size constants (`FS_*`, `LW_*`).
