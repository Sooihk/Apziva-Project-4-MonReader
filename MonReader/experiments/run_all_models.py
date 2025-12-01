# monreader/experiments/run_all_models.py
import pickle
from pathlib import Path

from MonReader.config import Config
from MonReader.training.train_resnet18 import train_resnet18
from MonReader.training.train_vgg16 import train_vgg16
from MonReader.training.train_efficientnet_b0 import train_efficientnet_b0
from MonReader.training.train_mobilenet_v2 import train_mobilenet_v2
from MonReader.evaluation.plots import plot_f1_curves, plot_metric_bars, plot_pareto_frontier, plot_radar_metrics

def main():
    # These dicts are what your plotting functions will consume
    model_metrics = {}   # model_name -> metrics dict
    f1_histories = {}    # model_name -> list[float] (val F1 per epoch)

    # --------------------------------------------------------------
    # 1) ResNet-18
    cfg_resnet = Config(tag="resnet18_pageflip")
    resnet_model, resnet_metrics, resnet_f1_history = train_resnet18(cfg_resnet)

    model_metrics["ResNet-18"] = resnet_metrics
    f1_histories["ResNet-18"] = resnet_f1_history

    # --------------------------------------------------------------
    # 2) VGG16
    cfg_vgg = Config(tag="vgg16_pageflip")
    vgg_model, vgg_metrics, vgg_f1_history = train_vgg16(cfg_vgg)

    model_metrics["VGG16"] = vgg_metrics
    f1_histories["VGG16"] = vgg_f1_history

    # --------------------------------------------------------------
    # 3) EfficientNet-B0
    cfg_eff = Config(tag="efficientnet_b0_pageflip")
    eff_model, eff_metrics, eff_f1_history = train_efficientnet_b0(cfg_eff)

    model_metrics["EfficientNet-B0"] = eff_metrics
    f1_histories["EfficientNet-B0"] = eff_f1_history

    # --------------------------------------------------------------
    # 4) MobileNetV2 (vanilla)
    cfg_mbv2 = Config(tag="mobilenet_v2_pageflip")
    mbv2_model, mbv2_metrics, mbv2_f1_history = train_mobilenet_v2(cfg_mbv2)

    model_metrics["MobileNetV2"] = mbv2_metrics
    f1_histories["MobileNetV2"] = mbv2_f1_history

    # --------------------------------------------------------------
    # Call plotting utilities
    # Pareto frontier: e.g. F1 vs size (MB)
    plot_pareto_frontier(model_metrics)

    # Bar chart of e.g. F1 / Precision / Recall per model
    plot_metric_bars(model_metrics)

    # F1 vs epoch curves for each model
    plot_f1_curves(f1_histories)

    # Radar plot of normalized metrics per model
    plot_radar_metrics(model_metrics)

    # --------------------------------------------------------------
    # Save dictionaries for later use
    cfg = Config()
    results_dir = (cfg.root / "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "model_results.pkl"

    with open(results_path, "wb") as f:
        pickle.dump(model_metrics, f)

    print(f"Saved metrics and histories to: {results_path}")

if __name__ == "__main__":
    main()
