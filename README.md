# TSE-RL

**Time Window Optimization for Motor Imagery Classification via Temporal Structure Embedded Reinforcement Learning**

Huan Luo, Na Lu, Xiaopeng Wang, and Ziyang Shang

**Accepted for publication in** *IEEE Journal of Biomedical and Health Informatics* (2026).

**Preprint:** [arXiv:2610.16921](https://arxiv.org/abs/2610.16921) · [Paper PDF](https://arxiv.org/pdf/2610.16921)

TSE-RL optimizes time windows for motor imagery EEG classification. An actor-critic policy trained with PPO assigns Discard, Onset, Sustain, and Offset actions to successive EEG windows. A finite-state machine constrains the available actions, a Transformer encodes window history, and action-dependent weights aggregate the predictions of retained windows. The implementation supports eight backbones with separate feature extractors (F) and discriminators (D).

## Supplementary material

The paper's [supplementary material (PDF)](Supplementary%20Material.pdf) is included in this repository.

## Installation

Use Python 3.10.18. The main dependencies, including PyTorch 2.3.1 and MOABB 1.2.0, are pinned in `requirements.txt`.

Run these commands from the project directory.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CPU execution is supported. To use an available CUDA device, add `--device cuda:0` to the training and evaluation commands.

## Data acquisition

Data are downloaded automatically through MOABB when `prepare` is run. Raw recordings are stored in `data/raw` by default. Use `--data-root` and `--output` to change the download and prepared-data locations.

| Dataset | MOABB class | Sampling rate used |
| --- | --- | --- |
| BNCI2014001 | `moabb.datasets.BNCI2014_001` | 250 Hz |
| BNCI2014002 | `moabb.datasets.BNCI2014_002` | 512 Hz |
| BNCI2015001 | `moabb.datasets.BNCI2015_001` | 512 Hz |
| OpenBMI / Lee2019 | `moabb.datasets.Lee2019_MI` | 1000 Hz |
| Liu2024 | `moabb.datasets.Liu2024` | 250 Hz |

MOABB uses one-based subject IDs. `OpenBMI`, `Lee2019`, and `Lee2019_MI` are accepted as dataset names. The three BNCI classes use the current names corresponding to the dataset names above. See the [MOABB dataset documentation](https://moabb.neurotechx.com/docs/api.html#datasets) for the original dataset references.

Preprocessing uses an 8–32 Hz band for the BNCI datasets and OpenBMI, and an 8–30 Hz band for Liu2024. The Liu2024 loader also downloads the original trial metadata through the MOABB downloader. Prepared EEG uses microvolt units. Continuous replay and sliding-window construction are implemented in `tse_rl/windows.py`.

Prepare one BNCI2014001 subject.

```bash
python -m tse_rl prepare --dataset BNCI2014001 --subject 1
```

Examples for the other datasets follow.

```bash
python -m tse_rl prepare --dataset BNCI2014002 --subject 1 --output data/prepared/bnci002_s1.npz
python -m tse_rl prepare --dataset BNCI2015001 --subject 1 --output data/prepared/bnci501_s1.npz
python -m tse_rl prepare --dataset OpenBMI --subject 1 --output data/prepared/openbmi_s1.npz
python -m tse_rl prepare --dataset Liu2024 --protocol patient-fold --fold 0 --output data/prepared/liu2024_fold0.npz
```

For cross-subject training, use `--protocol loso --subject 1`, where `--subject` identifies the held-out participant. Liu2024 supports five patient folds numbered 0–4.

## Training and evaluation

Run the three stages in order. These commands use the BNCI2014001 data prepared above and save checkpoints in `runs/bnci001_s1_eegnet`.

Stage 1 pretrains the feature extractor and discriminator using sliding windows.

```bash
python -m tse_rl stage1 --backbone EEGNet --epochs 300
```

Stage 2 trains the masked actor-critic policy and history encoder while freezing the current-window feature parameters.

```bash
python -m tse_rl stage2 --steps 40000
```

Stage 3 fine-tunes the policy and the selected feature-extractor layers.

```bash
python -m tse_rl stage3 --steps 20000
```

Evaluate the Stage 3 checkpoint.

```bash
python -m tse_rl evaluate --stage 3
```

Evaluation writes accuracy, retained-window counts, decision times, and per-trial predictions to `stage3_evaluation.json` in the run directory.

To change the dataset or output directory, pass the same `--data` and `--run-dir` arguments to all four commands. Select a backbone in Stage 1 using one of the following names.

```text
EEGNet
TSLANet
EEGConformer
FBCNet
ATCNet
M-ShallowConvNet
mAtt
CSPNet
```

The window length, step, and history length default to 1.0 s, 0.1 s, and five windows. They can be configured with `--window-sec`, `--step-sec`, and `--history-len` in Stage 1. Run `python -m tse_rl COMMAND --help` to see the arguments for each command.

## Directory structure

```text
TSE-RL/
├── README.md
├── CITATION.cff
├── Supplementary Material.pdf
├── LICENSE
├── requirements.txt
├── .python-version
├── .gitignore
├── third_party/
│   ├── FBCNet-LICENSE
│   ├── MAtt-LICENSE
│   ├── TSLANet-LICENSE
│   └── TorchEEG-LICENSE
├── tse_rl/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── data.py
│   ├── liu2024.py
│   ├── windows.py
│   ├── environment.py
│   ├── training.py
│   ├── evaluation.py
│   └── backbones/
│       ├── __init__.py
│       ├── eegnet.py
│       ├── tslanet.py
│       ├── eegconformer.py
│       ├── fbcnet.py
│       ├── atcnet.py
│       ├── mshallowconvnet.py
│       ├── matt.py
│       ├── cspnet.py
│       └── logits.py
└── tests/
    └── test_core.py
```

`data.py` and `liu2024.py` handle acquisition and preprocessing. `windows.py` provides stream concatenation and sliding windows. `environment.py` contains FSM masks, rewards, and weighted aggregation. Each backbone module contains its F/D decomposition and Transformer history encoder. `training.py` runs the supervised and PPO stages, and `evaluation.py` evaluates the trained policy. `third_party/` retains upstream license notices.

## Citation

If you use TSE-RL in your research, please cite our paper. It has been accepted for publication in *IEEE Journal of Biomedical and Health Informatics*. Until the journal version is available, please cite the arXiv preprint below.

```bibtex
@misc{luo2026tserl,
  title         = {Time Window Optimization for Motor Imagery Classification via Temporal Structure Embedded Reinforcement Learning},
  author        = {Luo, Huan and Lu, Na and Wang, Xiaopeng and Shang, Ziyang},
  year          = {2026},
  eprint        = {2610.16921},
  archivePrefix = {arXiv},
  primaryClass  = {cs.NE},
  doi           = {10.48550/arXiv.2610.16921},
  url           = {https://arxiv.org/abs/2610.16921}
}
```
