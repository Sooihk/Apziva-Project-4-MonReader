# libraries to import
from __future__ import annotations
import os, random, math, time, copy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# For a clean, stratified split
from sklearn.model_selection import StratifiedShuffleSplit

# visualizations 
import matplotlib.pyplot as plt
import json

# Reproducible function
def set_seed(seed: int = 42) -> None:
    """Set seeds for python, numpy (if used), and torch to have reproducible splits/training."""
    random.seed(seed)
    try:
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Config:
    # Task
    num_classes: int = 2
    img_size: int = 224

    # DataLoader
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # Seed
    seed: int = 42

    # Paths
    root: Path = field(default_factory=lambda: Path.cwd().parent)
    dataset_dir: Path = field(init=False)
    TRAIN_DIR: Path = field(init=False)
    TEST_DIR: Path = field(init=False)

    # I/O for experiements
    output_dir: Path = field(init=False)
    tag: str = 'resnet18_pageflip'

    # Training hyperparameters
    epochs_head_classifier: int = 5
    epochs_fine_tuning: int = 10
    lr_head: float = 1e-3
    lr_backbone: float = 2e-5
    weight_decay: float = 1e-4
    dropout: float = 0.0
    use_amp: bool = True

    # file system pointers to the training and testing images, fed into torchvision.datatsets.ImageFolder
    # training root containing one subfolder per class, held out test root, not touched during training/hyperparameter tuning
    def __post_init__(self):
        self.dataset_dir = self.root / 'data'
        self.TRAIN_DIR = self.dataset_dir / 'raw' / 'training'
        self.TEST_DIR = self.dataset_dir / 'raw' / 'testing'
        self.output_dir = self.root / 'models' / 'resnet18' / 'checkpoints'
        # Print what each directory resolves to
        print(f"Root directory:       {self.root}")
        print(f"Dataset directory:    {self.dataset_dir}")
        print(f"Training directory:   {self.TRAIN_DIR}")
        print(f"Testing directory:    {self.TEST_DIR}")
        print(f"Ouput ResNet directory:    {self.output_dir}")

cfg = Config()
set_seed(cfg.seed)

def get_model_size(model: nn.Module) -> float:
    """ 
    Return model parameter size (in MB) when stored
    """
    total_bytes = 0
    for p in model.state_dict().values():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 * 1024)

# Data Loaders (ImageFolder)
# channel wise mean and std to normalize images
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
        # A light set of augmentations to improve generalization while preserving label semantics.
        # 1. scale short side from 224 to 256
        # 2. Randomly choooses a crop then resizes that crop to 224x224, in order to train model to be robust to framing/zoom changes
        # 3. rotates image up to 8 degrees, small label preserving perturbation
        # 4 ColorJitter for mild brightness/contrast/saturation/hue changes to help resist lightning and white balance quirks
        # from phone cameras.
        # 5. Converts a PIL image to a PyTorch tensor shaped [C,H,W]
        # 6. Final conversion and ImageNet centering 
        Net effect is a moderate, label-safe augmentation 
    """
    resize = transforms.Resize(int(img_size * 1.14)) # make shorter side 14% bigger, to make margin for cropping
    to_tensor = transforms.ToTensor() # converts PIL to tensor 
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)   # standardize each channel using IMAGENET stats
    # Training pipeline Stochastic
    train_transforms = transforms.Compose([
        resize,    # 1      
        transforms.RandomResizedCrop(cfg.img_size),     # 2          
        transforms.RandomRotation(8),   # 3    
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),    # 4
        to_tensor,      # 5
        normalize,      # 6 
    ])
    # Validation/Test should be deterministic and comparable across epochs
    eval_transforms = transforms.Compose([
        resize,
        transforms.CenterCrop(cfg.img_size),
        to_tensor,
        normalize, 
    ])
    return train_transforms, eval_transforms

def get_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """ 
    Build training and validation loaders from training directory via a stratified 80/20 split 
    """
    train_pipeline, eval_pipeline = build_transforms(cfg.img_size)
    train_dir = cfg.TRAIN_DIR
    # Create two ImageFoldder views pointing at same files
    ds_eval = datasets.ImageFolder(str(train_dir), transform=eval_pipeline) # use determinstic eval transforms
    ds_aug = datasets.ImageFolder(str(train_dir), transform=train_pipeline) # use stochastic training transforms
    # grabs class indicies for each sample from eval view
    y = ds_eval.targets 

    # Single stratified shuffle split
    train_idx, val_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=cfg.seed)
                              .split(range(len(y)), y))
    
    # Wrap the views into subsets
    train_ds = Subset(ds_aug, train_idx)
    validation_ds = Subset(ds_eval, val_idx)

    # shared kwargs dictionary for both loaders
    common_dict = dict(
        batch_size = cfg.batch_size,
        num_workers = cfg.num_workers,
        pin_memory = cfg.pin_memory,
        persistent_workers = cfg.num_workers > 0
    )
    # wrap the subsets in DataLoaders with shared kwargs
    train_loader = DataLoader(train_ds, shuffle=True, **common_dict)
    validation_loader = DataLoader(validation_ds, shuffle=False, **common_dict)

    # map class name to index, inverts the class mapping so you can look up a human-readable class name from a numeric label
    # given: ds_eval.class_to_idx == {'cat': 0, 'dog': 1}
    # idx_to_class -> {0: 'cat', 1: 'dog'}
    idx_to_class = {v: k for k, v in ds_eval.class_to_idx.items()}

    return train_loader, validation_loader, idx_to_class

def get_test_loader(cfg: Config) -> DataLoader:
    # Build determinstic test loader from cfg.TEST_DIR
    _, eval_pipeline = build_transforms(cfg.img_size)

    test_ds = datasets.ImageFolder(str(cfg.TEST_DIR), transform=eval_pipeline)

    test_loader = DataLoader(
        test_ds,
        batch_size = cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0
    )
    return test_loader

def freeze_batchnorm_running_stats(module: nn.Module) -> None:
    """Iterate all submodules, for BatchNorm2d layers:
    - m.eval() uses stored runnng stats instead of batch stats
    - Diasable gradietns on learnable affine param (weight/bias) to keep them fixed
    """
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval() # freeze running mean/var updates
            if m.weight is not None:
                m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False

# Training and Evaluation functions

# decorator that runs the evaluate function with gradient tracking disabled
@torch.no_grad() 
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criteron: nn.Module) -> Dict[str, float]:
    """ 
    Runs a deterministic pass over the validation set with no gradient tracking, to report loss, accuracy, precision,
    recall, and F1 score. Function is called at the end of each epoch to decide early stopping, checkpointing.
    """
    model.eval()    # switch model to evaluation mode

    # Containers for per batch loss values, true labels and predicted labels
    losses:List[float] = []
    true_labels: List[int] = []
    predicted_labels: List[int] = []

    # Iterate over batches of inputs and labels
    for input, label in loader:
        input = input.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        logits = model(input)   # forward pass, raw logits of shape
        loss = criteron(logits, label) # compute batch loss
        losses.append(loss.item())   # store scalr loss value
        preds = torch.argmax(logits, dim=1) # turn logits into hard class predictions

        # accumlate lists of ground truths and predictions
        true_labels.extend(label.detach().cpu().tolist())
        predicted_labels.extend(preds.detach().cpu().tolist())

    # compute accuracy, precision, recall and F1 metrics for binary task
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary', pos_label=0)

    # Return averaged loss and metrics
    return {
        'loss': float(sum(losses) / max(1, len(losses))),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }

def run_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module,
                  optimizer: optim.Optimizer, scaler: GradScaler = None, use_amp: bool = True) -> Tuple[float, float]:
    """ 
    Main lifting of training consisting of forward, loss, backward, and optimizer step, reporting epoch level loss/accuracy
    """
    model.train()
    # accumaltors for epoch stats
    total_loss, n_samples, correct_predictions = 0.0,0,0 

    # Iterate over training batches
    for x,y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)   # clear gradients

        use_autocast = use_amp and scaler is not None and device.type in {"cuda", "mps"}

        if use_autocast:
            # autocast automatically picks the correct precision mode for the device type
            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:   # precision path if AMP disabled
            logits=model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        # sum of losses weighted by batch size
        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct_predictions += (preds == y).sum().item()
        n_samples += x.size(0)

    # Return avg loss and accuracy over whole epoch
    return total_loss / max(1, n_samples), correct_predictions / max(1, n_samples)

def restore_best_weights(model: nn.Module, cfg) -> None:
    """ 
    Restore the most recent best-weights file for the current tag. Reloading before fine-tuning.
    Ensure Phase B starts from the best head from Phase A
    """
    print("Restoring best head-only weights for fine tuning...")
    # Scan output directory for files beginning with current tag and contain 'best_weights'
    best_weights_files = [
        file for file in os.listdir(cfg.output_dir)
        if file.startswith(cfg.tag) and 'best_weights' in file
    ]
    if not best_weights_files:
        print("No best-weights file found; proceeding without restore.")
        return
    # Sort determinstic order as names have timestamps
    best_weights_files.sort()
    # pick latest
    weights_path = os.path.join(cfg.output_dir, best_weights_files[-1])
    try:
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
    except TypeError:
        # Older PyTorch
        state = torch.load(weights_path, map_location="cpu")
    # inject weights into model
    model.load_state_dict(state)
    print(f"Loaded best head weights: {weights_path}")

# Checkpoints, following functions to save model's state during training
# Save weights only (model parameters) and full checkpoint
def save_weights(model: nn.Module, path: str) -> str:
    # move tensors to cpu and write model state_dict() to path
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    # grab model's parameter dict and move all tensors to CPU
    cpu_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}   
    torch.save(cpu_state, path)
    return path

def save_full_checkpoint(model: nn.Module, optimizer=None, scheduler=None, scaler=None,
                         epoch: int = None, best_f1: float = None, extra: Dict[str, Any] = None,
                         path: str = None) -> str:
    """ 
    Writes a resumable bundle of the model's weights, optimizer, scheduler, AMP scaler, epoch, best F1,
    PyTorch version and any extra metrics
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    # Dictionary containing everything needed to resume training
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': int(epoch) if epoch is not None else None,
        'best_f1': float(best_f1) if best_f1 is not None else None,
        'extra': extra or {},
        'pytorch_version': torch.__version__
    }
    torch.save(checkpoint, path)    # write to disk
    return path

class CheckpointManager:
    """ 
    Utlity class to track best F1 and save new checkpoints when model F1 score improves.
    """
    def __init__(self, out_dir: str, tag: str):
        self.best_f1 = -1.0
        self.out_dir = out_dir
        self.tag = tag 
        self.best_weights_path = None
        self.best_full_path = None

        os.makedirs(out_dir, exist_ok=True)
    # call after each validation epoch with latest val F1 and training state
    def update(self, model: nn.Module, val_f1: float, epoch: int, optimizer=None, scheduler=None,
               scaler=None, metrics: Dict[str, float] = None):
        
        improved = (val_f1 is not None) and (val_f1 > self.best_f1)
        if improved:
            self.best_f1 = val_f1
            # timestamp for unique filenames
            ts = time.strftime("%Y%m%d-%H%M%S")
            # two paths for weights only with F1 and full checkpoint including epoch number
            weights_path = os.path.join(self.out_dir, f'{self.tag}_best_weights_f1{val_f1:.4f}_{ts}.pth')
            full_path = os.path.join(self.out_dir, f"{self.tag}_full_checkpoint_f1{val_f1:.4f}_e{epoch}_{ts}.pth")
            save_weights(model, weights_path)
            save_full_checkpoint(model, optimizer, scheduler, scaler, epoch, val_f1, metrics, full_path)
            # STORE BEST PATHS INSIDE THE MANAGER
            self.best_weights_path = weights_path
            self.best_full_path = full_path
            return True, weights_path, full_path
        
        return False, None, None
    
# ---------------------------
# VGG16
# ---------------------------
class VGG16Binary(nn.Module):
    def __init__(self, pretrained=True, dropout=0.4, out_dim=2):
        super().__init__()
        base = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        in_feats = base.classifier[6].in_features
        base.classifier[6] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feats, out_dim)
        )
        self.model = base

    def forward(self, x):
        return self.model(x)
    
def train_vgg16(cfg):
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, idx_to_class = get_dataloaders(cfg)
    num_classes = len(idx_to_class)
    model = VGG16Binary(pretrained=True, dropout=cfg.dropout, out_dim=num_classes).to(device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=cfg.use_amp and device.type == "cuda")
    ckpt = CheckpointManager(cfg.output_dir, cfg.tag)

    # Phase A: Linear probe
    for p in model.model.features.parameters():
        p.requires_grad = False
    for p in model.model.classifier.parameters():
        p.requires_grad = True

    freeze_batchnorm_running_stats(model.model.features)

    optimizer = optim.AdamW(model.model.classifier.parameters(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs_head_classifier)

    best_f1 = -1
    best_metrics: Dict[str,float] | None = None
    # store per epoch F1 for plotting
    f1_history: List[float] = []

    # Phase A 
    for epoch in range(1, cfg.epochs_head_classifier + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, device, criterion, optimizer, scaler, cfg.use_amp)
        metrics = evaluate(model, val_loader, device, criterion)
        val_loss = metrics["val_loss"]; val_acc = metrics["val_accuracy"]; val_prec = metrics["val_precision"]; val_rec = metrics["val_recall"]; val_f1 = metrics["val_f1"]
        f1_history.append(val_f1)
        improved, *_ = ckpt.update(model, metrics['val_f1'], epoch, optimizer, scheduler, scaler, metrics)
        
        
        if improved:
            best_f1 = val_f1
            best_metrics = metrics.copy()
        scheduler.step()
        print(f"[Head] Epoch {epoch:02d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} || "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} Prec {val_prec:.4f} Rec {val_rec:.4f} F1 {val_f1:.4f}")

    restore_best_weights(model, cfg)
    model.to(device)

    # ------------------------------------------------------------------------------
    # Phase B: Fine-tune last block
    for p in model.parameters():
        p.requires_grad = False
    for p in model.model.features[-10:].parameters():
        p.requires_grad = True
    for p in model.model.classifier.parameters():
        p.requires_grad = True

    freeze_batchnorm_running_stats(model.model.features)

    params = [
        {'params': model.model.features[-10:].parameters(), 'lr': cfg.lr_backbone},
        {'params': model.model.classifier.parameters(), 'lr': cfg.lr_head},
    ]
    optimizer = optim.AdamW(params, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs_fine_tuning)

    for epoch in range(1, cfg.epochs_fine_tuning + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, device, criterion, optimizer, scaler, cfg.use_amp)
        metrics = evaluate(model, val_loader, device, criterion)
        val_loss = metrics["val_loss"]; val_acc = metrics["val_accuracy"]
        val_prec = metrics["val_precision"]; val_rec = metrics["val_recall"]; val_f1 = metrics["val_f1"]
        # log f1 
        f1_history.append(val_f1)

        improved, *_ = ckpt.update(model, val_f1, epoch + cfg.epochs_head_classifier, optimizer, scheduler, scaler, metrics)
        if improved:
            best_f1 = val_f1
            best_metrics = metrics.copy()
        scheduler.step()
        print(f"[FT ] Epoch {epoch:02d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} || Val Loss {val_loss:.4f} Acc {val_acc:.4f} Prec {val_prec:.4f} Rec {val_rec:.4f} F1 {val_f1:.4f}")

    print(f"Best F1: {best_f1:.4f}")
    
    # ---- build summary for the plotting utils ----
    
    # Restore best model weights BEFORE returning
    # -----------------------------
    if ckpt.best_weights_path is not None:
        best_path = ckpt.best_weights_path
        state = torch.load(best_path, map_location=device)

        # Load depending on checkpoint structure
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
        elif isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

        print(f"Restored best student weights from: {best_path}")
    model_size = get_model_size(model)
    print(f"Model size (FP32): {model_size:.2f} MB")
    summary = {
        "acc":  best_metrics["val_accuracy"],
        "prec": best_metrics["val_precision"],
        "rec":  best_metrics["val_recall"],
        "f1":   best_metrics["val_f1"],
        "size_mb": model_size,
    }


    return model, summary, f1_history

# Configure output/tag for EfficientNet 
cfg.tag = "vgg_16_pageflip"
cfg.output_dir = (cfg.root / "models" / "vgg_16" / "checkpoints").resolve()

# Train VGG16
vgg_model, vgg_summary, vgg_history = train_vgg16(cfg)

class ResNet18Binary(nn.Module):
    def __init__(self, pretrained=True, dropout=0.0, out_dim=2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_feats = base.fc.in_features
        base.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feats, out_dim))
        self.model = base

    def forward(self, x):
        return self.model(x)
    
# Running ResNet18
def train_resnet18(cfg:Config):
    """ 
    Run a two phase training rountine for binary classification
    1. Phase A - linear probe (head only)
        Freeze the ResNet18 backbone, train just the final fully connected head for a few epochs
    2. Checkpoint & restore best head
        Save best F1 during Phase A and reload best weights before fine tuning.
    3. Phase B - Fine Tune (last block + head)
        Unfreeze layer 4 (last stage) and the head; train with discriminative learning rates
    """
    # making run reproducible, pick CPU/GPU
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Build loaders and label mapping
    train_loader, val_loader, idx_to_class = get_dataloaders(cfg)
    num_classes = len(idx_to_class)
    assert num_classes == 2, f"Expected binary classification, got {num_classes} classes: {idx_to_class}"

    # Create ResNet-18 with 2 logit head
    model = ResNet18Binary(pretrained=True, dropout=cfg.dropout, out_dim=num_classes)
    model.to(device)

    # Use cross entropy for multi class logits, and enable AMP gradient scaling
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=cfg.use_amp and device.type == 'cuda')

    # checkpoint manager to save best-F1 checkpoints
    checkpoint = CheckpointManager(cfg.output_dir, cfg.tag)

    # Phase A: Linear-Probe (freeze backbone) unfreeze final classifier, only head will train
    for p in model.model.parameters():
        p.requires_grad = False
    for p in model.model.fc.parameters():
        p.requires_grad = True
    freeze_batchnorm_running_stats(model.model.layer1)

    # Optimize the head with AdamW, cosine LR schedule over num of head-training epochs
    optimizer = optim.AdamW(model.model.fc.parameters(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs_head_classifier))

    best_validation_f1 = -1.0
    best_metrics: Dict[str,float] | None = None
    f1_history: List[float] = []
    # Train one epoch, run a no-grad validation pass and unpack metrics
    for epoch in range(1, cfg.epochs_head_classifier + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, device, criterion, optimizer, 
                                              scaler, cfg.use_amp)
        metrics = evaluate(model, val_loader, device, criterion)
        val_loss, val_acc, val_prec, val_rec, val_f1 = metrics.values()
        f1_history.append(val_f1)

        # if validation f1 improved, save weights and full checkpoint
        improved, w_path, f_path = checkpoint.update(model, val_f1, epoch, optimizer=optimizer, 
                                                     scheduler=scheduler, scaler=scaler, metrics=metrics)
        if improved: 
            best_validation_f1 = val_f1
            best_metrics = metrics.copy()
        scheduler.step()
        print(f"[Head] Epoch {epoch:02d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} || "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} Prec {val_prec:.4f} Rec {val_rec:.4f} F1 {val_f1:.4f}")
        
    # Restore best head snapshot before fine tuning
    restore_best_weights(model, cfg)
    model.to(device)

    # Phase 2: Fine-tune (unfreeze layer 4)
    for p in model.model.parameters():  # first refreeze everything
        p.requires_grad = False
    # Then unfreeze layer 4, last residual block and classifier head for fine tuning
    for m in [model.model.layer4, model.model.fc]:
        for p in m.parameters():
            p.requires_grad = True

    # call right after you unfreeze layer4 + fc:
    freeze_batchnorm_running_stats(model.model.layer4)
    
    # set discriminative learning rates, lower LR for backbone layer 4, higher for head
    param_groups = [
        {'params': model.model.layer4.parameters(), 'lr': cfg.lr_backbone},
        {'params': model.model.fc.parameters(), 'lr': cfg.lr_head},
    ]
    # new optimizer + schedule for this phase
    optimizer = optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs_fine_tuning))

    for epoch in range(1, cfg.epochs_fine_tuning + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, device, criterion, optimizer, scaler, 
                                              cfg.use_amp)
        metrics = evaluate(model, val_loader, device, criterion)
        val_loss, val_acc, val_prec, val_rec, val_f1 = metrics.values()
        f1_history.append(val_f1)

        improved, w_path, f_path = checkpoint.update(
            model, val_f1, epoch + cfg.epochs_head_classifier, optimizer = optimizer,
            scheduler=scheduler, scaler=scaler, metrics=metrics)

        if improved:
            best_validation_f1 = val_f1
            best_metrics = metrics.copy()
        scheduler.step()
        print(f"[FT ] Epoch {epoch:02d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} || "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} Prec {val_prec:.4f} Rec {val_rec:.4f} F1 {val_f1:.4f}")
    print(f"Training complete. Best Val F1: {best_validation_f1:.4f}")

    # Restore best model weights BEFORE returning
    # -----------------------------
    if checkpoint.best_weights_path is not None:
        best_path = checkpoint.best_weights_path
        state = torch.load(best_path, map_location=device)

        # Load depending on checkpoint structure
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
        elif isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

        print(f"Restored best student weights from: {best_path}")

    # ---- build summary for the plotting utils ----
    model_size = get_model_size(model)
    print(f"Model size (FP32): {model_size:.2f} MB")
    summary = {
        "acc":  best_metrics["accuracy"],
        "prec": best_metrics["precision"],
        "rec":  best_metrics["recall"],
        "f1":   best_metrics["f1"],
        "size_mb": model_size,
    } 

    return model, summary, f1_history

# Configure output/tag 
cfg.tag = 'resnet_18_pageflip'
cfg.output_dir = (cfg.root / 'models' / 'resnet_18' / 'checkpoints').resolve()

# Train ResNet18
resnet_model, resnet_summary, resnet_hist = train_resnet18(cfg)

class EfficientNetB0Binary(nn.Module):
    """ 
    Prepare a drop in EfficientNet-B0 with a swapped binary head for binary classification
    - Loads ImageNet weights by default
    - Replaces `classifier` with [Dropout, Linear(out_dim)]
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.2, out_dim: int = 2):
        super().__init__()
        base = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        # Torchvision EfficientNet classifer has a shape [Dropout, Linear]
        # Grab input size of existing linear layer
        feature_size = base.classifier[1].in_features
        # Swap ImageNet head for binary task specific
        base.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_size, out_dim)
        )
        self.model = base
    @property
    def features(self):
        # Expose the backbone to make freezing/unfreezing easier
        return self.model.features
    @property
    def classifier(self):
        # Expose the classifer head
        return self.model.classifier
    def forward(self, x):
        return self.model(x)

def train_efficientnet_b0(cfg:Config):
    """ 
    Two phase training for EfficientNet-B0
    Phase A: Linear probe (freeze backbone features, train only classifier)
    Phase B: Fine-tune (unfreeze last EfficientNet stage + classifier)
    """
    # reproducible
    set_seed(cfg.seed)
    device = get_device()
    print(f'Using device: {device}')

    # Build loaders, confirm binary task
    train_loader, val_loader, idx_to_class = get_dataloaders(cfg)
    num_classes = len(idx_to_class)
    assert num_classes == 2, f"Expected binary classification, got {num_classes} classes: {idx_to_class}"

    # Create EfficientNet-B0 with 2 logit head
    model = EfficientNetB0Binary(pretrained=True, dropout=max(cfg.dropout, 0.2), out_dim=num_classes)
    model.to(device)

    # Use cross entropy to get logits , AMP scaler 
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=cfg.use_amp and device.type == 'cuda')

    checkpoint = CheckpointManager(cfg.output_dir, cfg.tag)

    # Phase A: Linear probe, freeze backbone, train classifier head
    for p in model.features.parameters():
        p.requires_grad = False # telling autograd not to track these tensors, so no gradients are computed for backbone    
    for p in model.classifier.parameters():
        p.requires_grad = True

    freeze_batchnorm_running_stats(model.features)

    # Optimize the head with AdamW, cosine LR over num of 
    optimizer = optim.AdamW(model.classifier.parameters(), lr = cfg.lr_head, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs_head_classifier))

    best_validation_f1 = -1.0
    best_metrics: Dict[str, float] | None = None
    f1_history: Dict[float] = []

    # Train one epoch, run a no-grad validation pass and unpack metrics
    for epoch in range(1, cfg.epochs_head_classifier + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, device, criterion, optimizer, 
                                              scaler, cfg.use_amp)
        metrics = evaluate(model, val_loader, device, criterion)
        val_loss = metrics["val_loss"]; val_acc = metrics["val_accuracy"]; val_prec = metrics["val_precision"]; val_rec = metrics["val_recall"]; val_f1 = metrics["val_f1"]
        # log f1 for curves
        f1_history.append(val_f1)

        improved, w_path, f_path = checkpoint.update(
            model, val_f1, epoch,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler, metrics=metrics
        )
        if improved:
            best_validation_f1 = val_f1
            best_metrics = metrics.copy()
        scheduler.step()
        print(f"[Head] Epoch {epoch:02d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} || "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} Prec {val_prec:.4f} Rec {val_rec:.4f} F1 {val_f1:.4f}")

    restore_best_weights(model, cfg)
    model.to(device)

    # Phase B: fine tuning (unfreeze last stage and classifier)
    # First Refreeze
    for p in model.parameters():
        p.requires_grad = False
    
    # Unfreeze last 2 EfficientNet stage and classifer
    total_stages = len(model.features)  # figure out how many stages exist in features
    unfreeze_from = max(0, total_stages - 2)

    for i in range(unfreeze_from, total_stages):
        for p in model.features[i].parameters():
            p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    freeze_batchnorm_running_stats(model.features)

    ft_backbone_params: List[torch.nn.Parameter] = []
    for i in range(unfreeze_from, total_stages):
        for p in model.features[i].parameters():
            if p.requires_grad:
                ft_backbone_params.append(p)

    # set discriminative learning rates
    param_groups = [
        {'params': ft_backbone_params, 'lr': 5e-5}, # increased from 2e-5, as it increased F1 score
        {'params': model.classifier.parameters(), 'lr': cfg.lr_head},
    ]

    # New optimizer + scheduler for fine tuning
    optimizer = optim.AdamW(param_groups, weight_decay = cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max= max(1, cfg.epochs_fine_tuning))

    # train then evaluate
    for epoch in range(1, cfg.epochs_fine_tuning + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, device, criterion, optimizer, scaler, cfg.use_amp)
        metrics = evaluate(model, val_loader, device, criterion)
        val_loss = metrics["val_loss"]; val_acc = metrics["val_accuracy"]; val_prec = metrics["val_precision"]; val_rec = metrics["val_recall"]; val_f1 = metrics["val_f1"]
        f1_history.append(val_f1)

        improved, w_path, f_path = checkpoint.update(
            model, val_f1, epoch + cfg.epochs_head_classifier,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler, metrics=metrics
        )
        if improved:
            best_validation_f1 = val_f1
            best_metrics = metrics.copy()
        scheduler.step()
        print(f"[FT ] Epoch {epoch:02d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} || "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} Prec {val_prec:.4f} Rec {val_rec:.4f} F1 {val_f1:.4f}")

    print(f"Training complete. Best Val F1: {best_validation_f1:.4f}")

    # Restore best model weights BEFORE returning
    # -----------------------------
    if checkpoint.best_weights_path is not None:
        best_path = checkpoint.best_weights_path
        state = torch.load(best_path, map_location=device)

        # Load depending on checkpoint structure
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
        elif isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

        print(f"Restored best student weights from: {best_path}")
    model_size = get_model_size(model)
    print(f"Model size (FP32): {model_size:.2f} MB")

    summary = {
        "acc":  best_metrics["val_accuracy"],
        "prec": best_metrics["val_precision"],
        "rec":  best_metrics["val_recall"],
        "f1":   best_metrics["val_f1"],
        "size_mb": model_size,
    }
    return model, summary, f1_history

# Configure output/tag for EfficientNet 
cfg.tag = "efficientnet_b0_pageflip"
cfg.output_dir = (cfg.root / "models" / "efficientnet_b0" / "checkpoints").resolve()

# Train EfficientNet-B0
effb0_model, effb0_summary, effb0_hist = train_efficientnet_b0(cfg)

# model wrapper for MobileNetV2
class MobileNetV2Binary(nn.Module):
    """ 
    MobileNetV2 with a swapped binary head
    - load ImageNet weights
    - Replace 'classifier' with [Dropout, Linear(out)]
    - Expose features (backbone) and classifier (head)
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.2, out_dim: int = 2):
        super().__init__()
        # Load torchvision MobileNetV2
        base = models.mobilenet_v2(
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        # get mobilenetv2's head classifer at index 1, get in_features to size replacement linear layer
        feature_size = base.classifier[1].in_features
        # swap head
        base.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_size, out_dim)
        )
        self.model = base
    @property
    def features(self):
        return self.model.features
    @property
    def classifier(self):
        return self.model.classifier
    # foward computation
    def forward(self, x):
        return self.model(x)
    
def train_mobilenet_v2(cfg: Config):
    """ 
    Two‑phase training for MobileNetV2 using existing project utilities.
    Phase A: Linear probe (freeze backbone `features`, train only `classifier`).
    Phase B: Fine‑tune (unfreeze last ~2–3 MobileNetV2 stages + `classifier`).
    """
    # Fix random seed for reproduction, choose CPU/GPU
    set_seed(cfg.seed)
    device = get_device()
    print(f'Using device: {device}')

    # Build PyTorch DataLoaders and confirm is this is a binary task
    train_loader, val_loader, idx_to_class = get_dataloaders(cfg)
    num_classes = len(idx_to_class)
    assert num_classes == 2, f"Expected binary classification, got {num_classes} classes: {idx_to_class}"

    # Instantiate MobileNetV2Binar model with dropout value and move to device
    model = MobileNetV2Binary(pretrained=True, dropout=max(cfg.dropout, 0.2), out_dim = num_classes)
    model.to(device)

    # use cross entropy to get logits 
    criterion = nn.CrossEntropyLoss()
    # GradScaler enables loss sacling for AMP
    scaler = GradScaler(enabled=cfg.use_amp and device.type == 'cuda')

    # checkpoint help to save best so far weights
    checkpoint = CheckpointManager(cfg.output_dir, cfg.tag)

    # Phase A Linear probe, freeze backbone and train only the head
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    freeze_batchnorm_running_stats(model.features)
    
    # Optimize the head with AdamW and Cosine schedule over the number of head epochs
    optimizer = optim.AdamW(model.classifier.parameters(), lr = cfg.lr_head, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs_head_classifier))

    best_validation_f1 = -1.0
    best_metrics: Dict[str, float] | None = None
    f1_history: List[float] = []
    # Loop over head-training epochs and track the best validation F1
    for epoch in range(1, cfg.epochs_head_classifier + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, device, criterion, optimizer, 
                                              scaler, cfg.use_amp)
        metrics = evaluate(model, val_loader, device, criterion)
        val_loss = metrics["loss"]; val_acc = metrics["accuracy"]; val_prec = metrics["precision"]; val_rec = metrics["recall"]; val_f1 = metrics["f1"]
        f1_history.append(val_f1)

        improved, w_path, f_path = checkpoint.update(
            model, val_f1, epoch,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler, metrics=metrics
        )
        if improved:
            best_validation_f1 = val_f1
            best_metrics = metrics.copy()
        scheduler.step()
        print(f"[Head] Epoch {epoch:02d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} || "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} Prec {val_prec:.4f} Rec {val_rec:.4f} F1 {val_f1:.4f}")

    # Restore best head weights
    restore_best_weights(model, cfg)
    model.to(device)

    # Phase B: fine tune (unfreeze 2-3 stages)
    # Refreeze layers
    for p in model.parameters():
        p.requires_grad = False
    
    # Unfreeze last 2 MobileNet stages
    total_stages = len(model.features) # ~ 19 stages
    unfreeze_from = max(0, total_stages-2)
    # Selectively unfreeze last 2 staegs of the backbone and the head
    for i in range(unfreeze_from, total_stages):
        for p in model.features[i].parameters():
            p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    # Freeze BN running stats + affine params across the model
    freeze_batchnorm_running_stats(model.features)

    # Collect trainable backbone params to give separate learning rate
    ft_backbone_params: List[torch.nn.Parameter] = []
    for i in range(unfreeze_from, total_stages): 
        for p in model.features[i].parameters():
            if p.requires_grad:
                ft_backbone_params.append(p)
    # set discriminative learning rates
    param_groups = [
        {'params': ft_backbone_params, 'lr': cfg.lr_backbone},
        {'params': [p for p in model.classifier.parameters() if p.requires_grad], 'lr': cfg.lr_head},
    ]

    optimizer = optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs_fine_tuning))

    for epoch in range(1, cfg.epochs_fine_tuning + 1):
        train_loss, train_acc = run_one_epoch(
            model, train_loader, device, criterion, optimizer, scaler, cfg.use_amp
        )
        metrics = evaluate(model, val_loader, device, criterion)
        val_loss = metrics["loss"]; val_acc = metrics["accuracy"]
        val_prec = metrics["precision"]; val_rec = metrics["recall"]; val_f1 = metrics["f1"]
        f1_history.append(val_f1)

        improved, w_path, f_path = checkpoint.update(
            model, val_f1, epoch + cfg.epochs_head_classifier,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler, metrics=metrics
        )
        if improved:
            best_validation_f1 = val_f1
            best_metrics = metrics.copy()
        scheduler.step()
        print(f"[FT ] Epoch {epoch:02d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} || Val Loss {val_loss:.4f} Acc {val_acc:.4f} Prec {val_prec:.4f} Rec {val_rec:.4f} F1 {val_f1:.4f}")

    print(f"Training complete. Best Val F1: {best_validation_f1:.4f}")

    # Restore best model weights BEFORE returning
    # -----------------------------
    if checkpoint.best_weights_path is not None:
        best_path = checkpoint.best_weights_path
        state = torch.load(best_path, map_location=device)

        # Load depending on checkpoint structure
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
        elif isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

        print(f"Restored best student weights from: {best_path}")

    model_size = get_model_size(model)
    print(f"Model size (FP32): {model_size:.2f} MB")

    summary = {
        "acc":  best_metrics["accuracy"],
        "prec": best_metrics["precision"],
        "rec":  best_metrics["recall"],
        "f1":   best_metrics["f1"],
        "size_mb": model_size,
    }

    return model, summary, f1_history

# Configure output/tag 
cfg.tag = 'mobilenet_v2_pageflip'
cfg.output_dir = (cfg.root / 'models' / 'mobilenet_v2' / 'checkpoints').resolve()

# Train MobileNetV2
mbv2_model, mbv2_summary, mbv2_hist = train_mobilenet_v2(cfg)

model_metrics = {
    "VGG16":          vgg_summary,
    "ResNet18":       resnet_summary,
    "EfficientNet-B0":effb0_summary,
    "MobileNetV2":    mbv2_summary,
}

f1_histories = {
    "VGG16":          vgg_history,
    "ResNet18":       resnet_hist,
    "EfficientNet-B0":effb0_hist,
    "MobileNetV2":    mbv2_hist,
}

def plot_metric_bars(model_metrics: Dict[str, Dict[str, float]]) -> None:
    """Improved grouped bar chart (industry‑style) for Acc / Prec / Rec / F1.
    Cleaner visuals, consistent colors, tighter spacing, clearer labels.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    metrics_order = ["acc", "prec", "rec", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]  

    model_names = list(model_metrics.keys())
    n_models = len(model_names) # how many clusters
    n_metrics = len(metrics_order) # how many bars per cluster

    
    # obtain values for each bar for each cluster
    values = np.array([[model_metrics[m][k] for m in model_names] for k in metrics_order])
    
    # Build x axis
    x = np.arange(n_models)
    width = 0.18 # bar width

    fig, ax = plt.subplots(figsize=(12, 6))

    # grouped bars
    for i, (label, color) in enumerate(zip(metric_labels, colors)):
        # for each metric, recenters each group of bars around the model's x position
        ax.bar(
            x + (i - (n_metrics - 1) / 2) * width,
            values[i],
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Validation Metrics per Model", fontsize=14, weight="bold")
    ax.legend(frameon=False, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

plot_metric_bars(model_metrics)

def plot_f1_curves(f1_histories: Dict[str, List[float]], phase_a_epochs: int = 5) -> None:
    """Industry‑style F1 vs epoch plot with Phase A/B divider."""

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    plt.figure(figsize=(12, 6))

    # --- Phase A shaded region ---
    plt.axvspan(
        0.5,                          # slightly before epoch 1
        phase_a_epochs + 0.5,         # slightly after last Phase A epoch
        color="lightgrey",
        alpha=0.50,
        label="Phase A (Linear Probe)",
    )

    for (name, f1_list), color in zip(f1_histories.items(), colors):
        epochs = np.arange(1, len(f1_list) + 1)
        plt.plot(
            epochs, f1_list,
            marker="o", markersize=5,
            linewidth=2.2,
            color=color,
            label=name,
        )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation F1", fontsize=12)
    plt.title("Validation F1 vs Epoch per Model", fontsize=14, weight="bold")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()
    plt.show()

plot_f1_curves(f1_histories, 5)

def plot_radar_metrics(model_metrics: Dict[str, Dict[str, float]]) -> None:
    """ 
    Radar / Spider Plot where each model is represented by a polygon spreading over axes: accuracy, precision, recall, F1 and size efficiency
    """

    # Axes: Accuracy, Precision, Recall, F1, Efficiency
    base_axes = ["acc", "prec", "rec", "f1", "efficiency"]
    axis_labels = ["Accuracy", "Precision", "Recall", "F1", "F1/MB"]

    # Compute F1 per MB efficiency
    raw_eff = {
        # for each model, take its F1 score and divid by its model size MB 
        m : model_metrics[m]['f1'] / model_metrics[m]['size_mb']
        for m in model_metrics
    }

    # normalize efficiency to [0,1] standard for radar
    # find max efficiency across models
    max_eff = max(raw_eff.values())
    eff_scores = {m: raw_eff[m] / max_eff for m in raw_eff}

    # Compute the angles around the circle
    n_axes = len(base_axes)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1] # append first angle to the end to connect first and last angle

    # Creat circular plot
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    # Loop over models and build their polygons
    for (model_name, metrics), color in zip(model_metrics.items(), colors):
        values = [
            metrics.get("acc", 0),
            metrics.get("prec", 0),
            metrics.get("rec", 0),
            metrics.get("f1", 0),
            eff_scores[model_name],
        ]
        values += values[:1]

        ax.plot(angles, values, linewidth=2.2, color=color, label=model_name)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_yticklabels([])
    # Raise title higher by adjusting pad
    ax.set_title(
        "Radar Comparison of Model Performance and Efficiency", fontsize=14, weight="bold", pad=30
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), frameon=False)

    plt.tight_layout()
    plt.show()
   
plot_radar_metrics(model_metrics)

# Pareto Frontier plot
def plot_pareto_frontier(model_metrics: Dict[str, Dict[str, float]]) -> None:
    """Plot the Pareto frontier for (size_mb, F1).
    Frontier = models not dominated by any other model in both F1↑ and size↓.
    """

    model_names = list(model_metrics.keys())
    # values for 2d coords plot
    sizes = np.array([model_metrics[m]["size_mb"] for m in model_names])
    f1s = np.array([model_metrics[m]["f1"] for m in model_names])

    # Sort by size (ascending)
    order = np.argsort(sizes)
    sizes = sizes[order]
    f1s = f1s[order]
    sorted_names = [model_names[i] for i in order]

    # Compute Pareto frontier
    # walks through modesl from smallest to largest and a model is added to the frontier if its F1 is higher than previous models
    frontier_idx = []
    best_f1 = -np.inf
    for i in range(len(sizes)):
        if f1s[i] > best_f1:
            frontier_idx.append(i)
            best_f1 = f1s[i]

    plt.figure(figsize=(10, 6))

    # Scatter all points
    plt.scatter(sizes, f1s, s=160, color="#4C72B0", edgecolor="black", linewidth=0.7)
    for name, s, f in zip(sorted_names, sizes, f1s):
        plt.text(s + 0.3, f, name, fontsize=10)

    # Draw frontier line
    plt.plot(sizes[frontier_idx], f1s[frontier_idx], color="#C44E52", linewidth=2.4, label="Pareto Frontier")

    plt.xlabel("Model Size (MB)", fontsize=12)
    plt.ylabel("Validation F1", fontsize=12)
    plt.title("Pareto Frontier: F1 vs Model Size", fontsize=15, weight="bold", pad=15)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()
    plt.show()

plot_pareto_frontier(model_metrics)

def get_resnet18_teacher_ckpt_path(cfg) -> Path:
    #Return the full path to the chosen ResNet-18 teacher checkpoint.
    teacher_dir = (cfg.root / "models" / "resnet_18" / "checkpoints").resolve()
    return teacher_dir / "resnet_18_pageflip_full_checkpoint_f10.9581_e14_20251123-225022.pth"

def load_resnet18_teacher(num_classes: int, ckpt_path: str, device: torch.device):
    # Load the trained ResNet-18 model and freeze it to act as a teacher
    teacher = models.resnet18(weights=None)     # create a blank ResNet18 with no pretrained weights
    # replace classification head with task's output layer
    in_features = teacher.fc.in_features
    #teacher.fc = nn.Linear(in_features, num_classes)
    teacher.fc = nn.Sequential(
    nn.Dropout(0.0),  # Or the dropout used during training
    nn.Linear(in_features, num_classes))

    # Load saved checkpoint file
    state = torch.load(ckpt_path, map_location=device)
    # CASE 1: full checkpoint like {"model": state_dict, ...}
    if "model" in state:
        raw_state = state["model"]
    elif "model_state_dict" in state:
        raw_state = state["model_state_dict"]
    else:
        raw_state = state

    # CASE 2: keys prefixed with "model."
    cleaned_state = {}
    for k, v in raw_state.items():
        if k.startswith("model."):
            cleaned_state[k[len("model."):]] = v
        else:
            cleaned_state[k] = v

    teacher.load_state_dict(cleaned_state)  # strict=True by default


    # Move model to GPU
    teacher.to(device)
    teacher.eval()  # teacher in inference mode 
    # Freeze every parameter, no gradient flow into teacher, weights can not be modified
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher

def create_mobilenetV2_student(num_classes: int, device: torch.device):
    """ 
    Loads ImageNet pretrained weights, swaps out classsifier head to 2 classes, moves model to GPU, 
    and returns a ready to reain model to learn from ResNet18 teacher
    """
    # Load pretrained weights
    student = models.mobilenet_v2(weights = models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # find number of input features to the classifier
    in_features = student.classifier[1].in_features
    student.classifier[1] = nn.Linear(in_features, num_classes)
    student.to(device)
    return student

class DistillationLoss(nn.Module):
    """ 
    Using Hinton-style distillation loss: L_total = alpha * KL(teacher_T || student_T) * T^2 + (1 - alpha) * CE(student, y)
    Combines two learning signals: 
        - soft loss: imitate the teacher's probability distribution 
        - hard loss: match the true labels
    Blends them with a mixing weight alpha

    """
    def __init__(self, T: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.T = T
        self.alpha = alpha 
        self.kl_div = nn.KLDivLoss(reduction = 'batchmean')
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        T = self.T
        # Get soft class probabilties from teacher
        with torch.no_grad():
            p_teacher = F.softmax(teacher_logits / T, dim = 1)
        # Get log probabilties from student, need in log space for KLDivLoss
        log_p_student = F.log_softmax(teacher_logits / T, dim=1)
        # KL divergence betweeen student and teacher distributions
        loss_soft = self.kl_div(log_p_student, p_teacher) * (T * T)
        #  cross entropy with hard labels
        loss_hard = self.ce(student_logits, targets)
        # Blend two losses to make the sudent learn the shape of the teacher's uncertainity and anchor with actual labels
        return self.alpha * loss_soft + (1.0 - self.alpha) * loss_hard

def train_student_with_distillation(
    teacher, 
    student, 
    train_loader,
    val_loader,
    device,
    evaluate, 
    checkpoint,
    num_epochs: int = 10,
    lr: float = 1e-4,
    T: float = 4.0,
    alpha: float = 0.7,
    optimizer_cls = torch.optim.AdamW,
    use_amp: bool = True,
    ):
    """ 
    Train MobileNetV2 student with knowledge distillation from ResNet18
    """
    # updates only student's parameters
    optimizer = optimizer_cls(student.parameters(), lr = lr)
    criterion = DistillationLoss(T=T, alpha=alpha)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_f1 = -1
    best_metrics: Dict[str, float] | None = None
    f1_history = []

    for epoch in range(1, num_epochs + 1):  # epoch loop
        student.train() # enable dropout, batchnorm updates
        running_loss, correct, total = 0.0, 0, 0 # counters

        # training loop per batch
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # teacher forward pass, computes teacher_logits without gradients
            with torch.no_grad():
                teacher_logits = teacher(images) # teacher model gives its opnions on each image
            # zero gradients on the student optimizer
            optimizer.zero_grad()

            # Inside autocast, normal forward passs student, and KD loss
            with torch.amp.autocast(device.type, enabled=use_amp):
                student_logits = student(images)
                loss = criterion(student_logits, teacher_logits, labels)
            # scale the loss for fp16 stability, backpropagates through student only
            # update student weights, teacher remains unchanged
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # training metrics
            running_loss += loss.item() * images.size(0)
            _, preds = student_logits.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Evaluation
        student.eval()
        eval_criterion = nn.CrossEntropyLoss() # cross-entropy for evaluation 
        metrics = evaluate(student, val_loader, device, eval_criterion)
        val_loss = metrics["loss"]
        val_acc  = metrics["accuracy"]
        val_prec = metrics["precision"]
        val_rec  = metrics["recall"]
        val_f1   = metrics["f1"]
        f1_history.append(val_f1) 

        # Checkpoint saving if val_f1 improved
        improved, w_path, f_path = checkpoint.update(
            student,
            val_f1,
            epoch,
            optimizer=optimizer,
            scheduler=None,
            scaler=scaler,
            metrics=metrics,
        )
        if improved:
            best_f1 = val_f1
            best_metrics = metrics.copy()
        # logging
        print(
            f"[KD ResNet18-MBNetV2] Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, ACC: {train_acc:.4f} || "
            f"Validation Loss: {val_loss:.4f}, ACC: {val_acc:.4f}, PREC: {val_prec:.4f}, REC: {val_rec:.4f}, F1: {val_f1:.4f}"
        )
    # Restore best student weights BEFORE returning
    # -----------------------------
    best_path = getattr(checkpoint, "best_weights_path", None)
    if best_path is not None:
        state = torch.load(best_path, map_location=device)

        # Load depending on checkpoint structure
        if isinstance(state, dict) and "model" in state:
            student.load_state_dict(state["model"])
        elif isinstance(state, dict) and "model_state_dict" in state:
            student.load_state_dict(state["model_state_dict"])
        else:
            student.load_state_dict(state)

        print(f"Restored best student weights from: {best_path}")
    print(f"Training complete. Best Val F1: {best_f1:.4f}")

    summary = {
        'acc': best_metrics['accuracy'],
        'prec': best_metrics['precision'],
        'rec': best_metrics['recall'],
        'f1': best_metrics['f1']
    }

    return student, summary, f1_history

def train_mobilenetv2_kd(cfg: Config):
    """ 
    Full training pipeline for MobileNetV2 distilled from ResNet18
    """
    # Set up
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Data Loaders for train and validation
    train_loader, val_loader, idx_to_class = get_dataloaders(cfg)
    num_classes = len(idx_to_class)

    # Load ResNet18 teacher
    teacher_checkpoint = get_resnet18_teacher_ckpt_path(cfg)
    print(f"Loading teacher checkpoint: {teacher_checkpoint}")
    # Create fixed teacher ResNet-18 model for distillation
    teacher = load_resnet18_teacher(
        num_classes=num_classes, 
        ckpt_path=str(teacher_checkpoint),
        device=device,
    )

    # Create Student (MobileNetV2) with pretrained ImageNet weights
    student = create_mobilenetV2_student(
        num_classes=num_classes,
        device=device
    )

    # checkpoint manager
    checkpoint = CheckpointManager(cfg.output_dir, cfg.tag + '_KD')

    # Training Knowledge Distillation
    student, summary, f1_history = train_student_with_distillation(
        teacher = teacher, # Resnet-18
        student = student, # MobileNetV2
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        evaluate=evaluate,
        checkpoint=checkpoint,
        num_epochs=cfg.epochs_fine_tuning,
        lr=cfg.lr_head,
        T=4.0,
        alpha=0.7,
        use_amp=cfg.use_amp
    )

    # -----------------------------
    # 7. Compute final student model size
    # -----------------------------
    try:
        student_size = get_model_size(student)
    except:
        student_size = None

    summary["size_mb"] = student_size

    print(f"KD Student (MobileNetV2) Summary: {summary}")
    if student_size is not None:
        print(f"Model size: {student_size:.2f} MB")

    return student, summary, f1_history

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")
print("===== Distilling MobileNetV2 from ResNet-18 =====")

cfg.tag = "mobilenet_v2_pageflip_kd"
cfg.output_dir = (cfg.root / "models" / "mobilenet_v2_kd" / "checkpoints").resolve()

kd_model, kd_summary, kd_f1_history = train_mobilenetv2_kd(cfg)

print("KD Summary:", kd_summary)

def evaluate_on_test(model: nn.Module, cfg: Config, evaluate_fn, device=None):
    """ 
    Evaluate any trained model on the test dataset
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, eval_transform = build_transforms(cfg.img_size)
    # Build test dataset
    test_dataset = datasets.ImageFolder(str(cfg.TEST_DIR), transform=eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )

    criterion = nn.CrossEntropyLoss()

    model.eval()
    metrics = evaluate_fn(model, test_loader, device, criterion)
    metrics['size_mb'] = get_model_size(model)

    print("\n===== Test Set Evaluation =====")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"Loss     : {metrics['loss']:.4f}")
    print(f"Size     : {metrics['size_mb']:.2f} MB\n")

    return metrics

kd_test_metrics = evaluate_on_test(kd_model, cfg, evaluate_fn=evaluate)
resnet_test_metrics = evaluate_on_test(resnet_model, cfg, evaluate_fn=evaluate)
mbv2_test_metrics = evaluate_on_test(mbv2_model, cfg, evaluate_fn=evaluate)
vgg16_test_metrics = evaluate_on_test(vgg_model, cfg, evaluate_fn=evaluate)
effb0_test_metrics = evaluate_on_test(effb0_model, cfg, evaluate_fn=evaluate)

model_results = {
    "MobileNetV2_KD": kd_test_metrics,
    "MobileNetV2": mbv2_test_metrics,
    "ResNet18": resnet_test_metrics,
    "VGG16": vgg16_test_metrics,
    "EfficientNetB0": effb0_test_metrics,
}

def display_academic_results_table(models_metrics: dict):
    """
    Display an academic‑style results table for multiple models.
    """
    from tabulate import tabulate

    # Convert dictionary structure into a list of rows
    headers = ["Model", 'Loss', "Accuracy", "Precision", "Recall", "F1 Score", "Size (MB)"]
    rows = []

    for model_name, metrics in models_metrics.items():
        rows.append([
            model_name,
            f"{metrics.get('loss', 0):.4f}",
            f"{metrics.get('accuracy', 0):.4f}",
            f"{metrics.get('precision', 0):.4f}",
            f"{metrics.get('recall', 0):.4f}",
            f"{metrics.get('f1', 0):.4f}",
            f"{metrics.get('size_mb', 0):.2f}"
        ])

    # Generate table in academic formatting
    table = tabulate(rows, headers=headers, tablefmt="github", numalign="center", stralign="center")
    print("\nResults Table:\n")
    print(table)
display_academic_results_table(model_results)

from matplotlib.gridspec import GridSpec

def plot_kd_summary_figure(model_metrics: dict):
    """ 
    Knowledge Distillation Summary Figure with 3 Panels:
    Panel A: Multi-metric bar chart comparison
    Panel B: Pareto Plot (Model size vs F1 Score)
    Panel C: Radar Plot (KD vs Baseline MobileNetV2)
    """
    model_names = list(model_metrics.keys())
    # extract metrics
    loss = [model_metrics[m]['loss'] for m in model_names]
    acc = [model_metrics[m]["accuracy"] for m in model_names]
    prec = [model_metrics[m]["precision"] for m in model_names]
    rec = [model_metrics[m]["recall"] for m in model_names]
    f1 = [model_metrics[m]["f1"] for m in model_names]
    size = [model_metrics[m]["size_mb"] for m in model_names]

    # Create 3 panel figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20,12))
    gs = GridSpec(2, 2, width_ratios=[1,1.2], height_ratios=[1,1], figure=fig)

    # publication-style color palette
    pub_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Panel A Multi-metric bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(model_names))
    width = 0.13
    # plot grouped bars for each metric
    ax1.bar(x-2*width, loss, width, label='Loss')
    ax1.bar(x - width, acc, width, label="Accuracy")
    ax1.bar(x, prec,  width, label="Precision")
    ax1.bar(x + width, rec, width, label="Recall")
    ax1.bar(x + 2*width, f1, width, label="F1 Score")
    #ax1.bar(x + 3*width, size, width, label="Size (MB)")
    # subplot 1 cosmetics
    ax1.set_title("Panel A: Multi-Metric Comparison", fontsize=14, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.5)


    # Panel B Pareto Frontier Plot (Model Size vs F1)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(size, f1, s=120)
    for i, m in enumerate(model_names):
        ax2.annotate(m, (size[i], f1[i]), textcoords='offset points', xytext=(5,5))
    # Compute Pareto frontier (maxmize f1, minimize size)
    points = list(zip(size, f1, model_names))
    # sort by size 
    points.sort(key = lambda x: x[0])
    pareto = []
    best_f1 = -1
    for s, f1, n in points:
        if f1 > best_f1:
            pareto.append((s, f1))
            best_f1 = f1

    # Draw frontier line
    if len(pareto) > 1:
        pf_size, pf_f1 = zip(*pareto)
        ax2.plot(pf_size, pf_f1, linestyle="--", color='red', linewidth=2, label='Pareto Frontier')
        ax2.legend()
    ax2.set_xlabel("Model Size (MB)")
    ax2.set_ylabel("F1 Score")
    ax2.set_title('Panel B: Model Size vs F1 Score (Pareto Plot)', weight='bold')
    ax2.grid(linestyle='--', alpha=0.7)


    # Panel C Radar Plot
    # Metrics to normalize (only Loss + Efficiency)
    labels = ["Accuracy", "Precision", "Recall", "F1", "Loss (inverted)", "Efficiency (F1/Size)"]
    kd = model_metrics["MobileNetV2_KD"]
    base = model_metrics["MobileNetV2"]

    # Compute raw efficiency
    kd_eff = kd["f1"] / kd["size_mb"]
    base_eff = base["f1"] / base["size_mb"]

    # Collect raw values (keep original 0-1 metrics untouched)
    raw_kd = np.array([
        kd["accuracy"], kd["precision"], kd["recall"], kd["f1"], kd["loss"], kd_eff
    ])
    raw_base = np.array([
        base["accuracy"], base["precision"], base["recall"], base["f1"], base["loss"], base_eff
    ])

    # Build kd_norm and base_norm 
    # Keep Accuracy, Precision, Recall, F1 unchanged
    kd_norm = [raw_kd[0], raw_kd[1], raw_kd[2], raw_kd[3]]
    base_norm = [raw_base[0], raw_base[1], raw_base[2], raw_base[3]]

    # Normalize Loss (lower = better, invert to higher = better)
    vals_loss = np.array([raw_kd[4], raw_base[4]])
    min_l, max_l = vals_loss.min(), vals_loss.max()
    if max_l - min_l == 0:
        kd_loss_norm = base_loss_norm = 1.0
    else:
        kd_loss_norm = 1 - ((raw_kd[4] - min_l) / (max_l - min_l))
        base_loss_norm = 1 - ((raw_base[4] - min_l) / (max_l - min_l))

    kd_norm.append(kd_loss_norm)
    base_norm.append(base_loss_norm)

    # Normalize Efficiency (F1 / Size) 
    vals_eff = np.array([raw_kd[5], raw_base[5]])
    min_e, max_e = vals_eff.min(), vals_eff.max()
    if max_e - min_e == 0:
        kd_eff_norm = base_eff_norm = 1.0
    else:
        kd_eff_norm = (raw_kd[5] - min_e) / (max_e - min_e)
        base_eff_norm = (raw_base[5] - min_e) / (max_e - min_e)

    kd_norm.append(kd_eff_norm)
    base_norm.append(base_eff_norm)

    # Prepare radar coordinates
    num_vars = len(labels)
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()

    kd_plot = kd_norm + kd_norm[:1]
    base_plot = base_norm + base_norm[:1]
    angles += angles[:1]

    ax3 = fig.add_subplot(gs[:, 1], polar=True)  # Panel C occupies full right side  # Radar subplot (publication-ready)
    ax3.plot(angles, kd_plot, label="MobileNetV2_KD", linewidth=2)
    ax3.fill(angles, kd_plot, alpha=0.25)

    ax3.plot(angles, base_plot, label="MobileNetV2", linewidth=2)
    ax3.fill(angles, base_plot, alpha=0.25)

    ax3.set_title("Panel C: Radar Plot (KD vs Baseline) — Normalized", weight='bold')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(labels)
    ax3.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25))

    plt.tight_layout()
    plt.show()

plot_kd_summary_figure(model_results)
    
import seaborn as sns
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets

# ----------------------------------
# Helper: build test_loader from cfg
# ----------------------------------
#def get_test_loader(cfg):
#    _, eval_tfm = build_transforms(cfg.img_size)
#
#    test_ds = datasets.ImageFolder(str(cfg.TEST_DIR), transform=eval_tfm)
#
#    test_loader = DataLoader(
#        test_ds,
#        batch_size=cfg.batch_size,
#        shuffle=False,
#        num_workers=cfg.num_workers,
#        pin_memory=cfg.pin_memory,
#        persistent_workers=cfg.num_workers > 0
#    )
#    return test_loader


# --------------------------------------------------------
# 1. Loss Distribution Functions (now take cfg instead)
# --------------------------------------------------------
def get_loss_distribution(model, cfg, device):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="none")
    losses = []

    test_loader = get_test_loader(cfg)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            batch_loss = ce(logits, labels)
            losses.extend(batch_loss.cpu().numpy())

    return losses


def plot_loss_distributions(model_a, model_b, name_a, name_b, cfg, device):
    losses_a = get_loss_distribution(model_a, cfg, device)
    losses_b = get_loss_distribution(model_b, cfg, device)

    plt.figure(figsize=(8,5))
    sns.kdeplot(losses_a, label=name_a, linewidth=2)
    sns.kdeplot(losses_b, label=name_b, linewidth=2)

    plt.title("Loss Distribution KD vs Baseline MobileNetV2")
    plt.xlabel("Cross-Entropy Loss")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

device = get_device()

plot_loss_distributions(
    kd_model,
    mbv2_model,
    "MobileNetV2_KD",
    "MobileNetV2",
    cfg,
    device
)






