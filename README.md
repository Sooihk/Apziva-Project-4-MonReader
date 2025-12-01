# monreader

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   └── raw            <- The original, immutable data dump.
|       ├── testing/
|               ├── flip   
│               └── notflip    
│       └── training/
|               ├── flip   
│               └── notflip    
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         MonReader and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── MonReader/
    ├── __init__.py          # Makes MonReader importable as a package
    ├── config.py            # Global configuration: paths, hyperparameters, experiment tags
    ├── utils.py             # Utility helpers: seeding, device selection, logging niceties
    │
    ├── data/
    │   ├── __init__.py
    │   ├── datamodules.py   # DataModule-style wrappers that build train/val/test DataLoaders
    │   └── transforms.py    # Torchvision transform pipelines for training/validation/testing
    │
    ├── evaluation/
    │   ├── __init__.py
    │   ├── metrics.py       # Metric helpers, e.g. classification report -> dict
    │   └── plots.py         # Plotting utilities: metric bars, radar plots, Pareto, KD summary, etc.
    │
    ├── experiments/
    │   ├── run_all_models.py      # Trains all backbones on train/val, collects metrics & F1 histories,
    │   │                          # and generates validation plots (bars, radar, Pareto, F1 curves).
    │   └── run_kd_vs_baselines.py # Trains the MobileNetV2 KD student, loads the best checkpoints for
    │                              # VGG16/ResNet18/EfficientNetB0/base MobileNetV2, evaluates all models
    │                              # on the test set, and produces a comparison table + KD summary figure.
    │
    ├── models/
    │   ├── __init__.py
    │   ├── efficientnet_b0.py # EfficientNet-B0 binary classifier wrapper
    │   ├── mobilenet_v2.py    # MobileNetV2 binary classifier + student factory for KD
    │   ├── resnet18.py        # ResNet18 binary classifier wrapper
    │   ├── teachers.py        # Teacher loading utilities (e.g. ResNet18 best checkpoint for KD)
    │   ├── utils.py           # Model utilities: e.g. BatchNorm freezing helpers
    │   └── vgg16.py           # VGG16 binary classifier wrapper
    │
    └── training/
        ├── __init__.py
        ├── distillation.py        # DistillationLoss and KD training loop for the student
        ├── engine.py              # Core training/eval loop + CheckpointManager abstraction
        ├── train_efficientnet_b0.py  # Two-phase EfficientNet-B0 training entrypoint
        ├── train_mobilenet_v2.py     # Two-phase MobileNetV2 training entrypoint
        ├── train_mobilenet_v2_kd.py  # Single-phase full-network KD training for MobileNetV2 student
        ├── train_resnet18.py         # Two-phase ResNet18 training entrypoint
        └── train_vgg16.py            # Two-phase VGG16 training entrypoint
```

--------
## Getting Started:  
Working with Python 3.12.2 for this project. 
Clone the repository and install the dependencies:

`pip install -r requirements.txt`

## How to run experiments
### 1. Train all models and generate validation plots
From the project root:

`python -m MonReader.experiments.run_all_models`

This script:
* Trains ResNet18, VGG16, EfficientNet-B0, MobileNetV2, and MobileNetV2-KD
on the training set with validation monitoring.
* Collects:
  - the best validation metrics (accuracy/precision/recall/F1, model size),
  - and per-epoch validation F1 histories for each model.

* Produces comparison plots:
  - grouped metric bar charts,
  - radar (spider) plots,
  - F1-vs-model-size Pareto plot,
  - F1-vs-epoch curves.

Use this when you want to see how all architectures behave on train/validation.

### 2. Compare KD vs baselines on the test set

From the project root:

`python -m MonReader.experiments.run_kd_vs_baselines`

This script:
1. Trains the MobileNetV2 KD student.

2. Rebuilds VGG16, ResNet18, EfficientNet-B0, and base MobileNetV2 and loads their best-validation checkpoints.

3. Evaluates all five models on the test dataset using a shared `evaluate_on_test()` helper.

4. Aggregates test metrics into a `model_results` dictionary and calls:
  * `display_academic_results_table(model_results)` to show a publication-style comparison table.
  * `plot_kd_summary_figure(model_results)` to visualize how KD compares tothe baseline models.
