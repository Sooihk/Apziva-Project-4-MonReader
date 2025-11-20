# libraries to import
from __future__ import annotations
import os, random, math, time, copy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# For a clean, stratified split
from sklearn.model_selection import StratifiedShuffleSplit

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
        'val_loss': float(sum(losses) / max(1, len(losses))),
        'val_accuracy': float(accuracy),
        'val_precision': float(precision),
        'val_recall': float(recall),
        'val_f1': float(f1),
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
    for epoch in range(1, cfg.epochs_head_classifier + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, device, criterion, optimizer, scaler, cfg.use_amp)
        metrics = evaluate(model, val_loader, device, criterion)
        val_loss = metrics["val_loss"]; val_acc = metrics["val_accuracy"]; val_prec = metrics["val_precision"]; val_rec = metrics["val_recall"]; val_f1 = metrics["val_f1"]

        improved, *_ = ckpt.update(model, metrics['val_f1'], epoch, optimizer, scheduler, scaler, metrics)
        
        
        if improved:
            best_f1 = val_f1
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


        improved, *_ = ckpt.update(model, val_f1, epoch + cfg.epochs_head_classifier, optimizer, scheduler, scaler, metrics)
        if improved:
            best_f1 = val_f1
        scheduler.step()
        print(f"[FT ] Epoch {epoch:02d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} || Val Loss {val_loss:.4f} Acc {val_acc:.4f} Prec {val_prec:.4f} Rec {val_rec:.4f} F1 {val_f1:.4f}")

    print(f"Best F1: {best_f1:.4f}")
    model_size = get_model_size(model)
    print(f"Model size (FP32): {model_size:.2f} MB")
    return model

# Configure output/tag for EfficientNet 
cfg.tag = "vgg_16_pageflip"
cfg.output_dir = (cfg.root / "models" / "vgg_16" / "checkpoints").resolve()

# Train VGG16
train_vgg16(cfg)

class ResNet18Binary(nn.Module):
    def __init__(self, pretrained=True, dropout=0.0, out_dim=2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_feats = base.fc.in_features
        base.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feats, out_dim))
        self.model = base

    def forward(self, x):
        return self.model(x)
    

# ResNet-18 Head classifier swap
class ResNet18Binary(nn.Module):
    """ 
    Wrapper around torchvisions pretrained ResNet18
    - Replace fc with a new head producing logits (2 for binary)
    - Add optional Dropout
    - Swap in task specific head while retaining ImageNet-pretrained features
    """
    # create constructor with following;
    # pretrained: load ImageNet weights
    # dropout: probability for dropout before final linear layer
    # out_dim: num of output units (2)
    def __init__(self, pretrained: bool = True, dropout: float = 0.0, out_dim: int = 2):
        super().__init__()  # intialize base class
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # grab feature size feeding into the classifier (ResNet18: 512)
        feature_size = base.fc.in_features

        # Head swap
        if dropout > 0:
            base.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feature_size, out_dim)
            )
        else:
            base.fc = nn.Linear(feature_size, out_dim)

        self.model = base   # store modified backbone+head
     # Forward pass
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
    # Train one epoch, run a no-grad validation pass and unpack metrics
    for epoch in range(1, cfg.epochs_head_classifier + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, device, criterion, optimizer, 
                                              scaler, cfg.use_amp)
        metrics = evaluate(model, val_loader, device, criterion)
        val_loss, val_acc, val_prec, val_rec, val_f1 = metrics.values()

        # if validation f1 improved, save weights and full checkpoint
        improved, w_path, f_path = checkpoint.update(model, val_f1, epoch, optimizer=optimizer, 
                                                     scheduler=scheduler, scaler=scaler, metrics=metrics)
        if improved: 
            best_validation_f1 = val_f1
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

        improved, w_path, f_path = checkpoint.update(
            model, val_f1, epoch + cfg.epochs_head_classifier, optimizer = optimizer,
            scheduler=scheduler, scaler=scaler, metrics=metrics)

        if improved:
            best_validation_f1 = val_f1
        scheduler.step()
        print(f"[FT ] Epoch {epoch:02d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} || "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} Prec {val_prec:.4f} Rec {val_rec:.4f} F1 {val_f1:.4f}")
    print(f"Training complete. Best Val F1: {best_validation_f1:.4f}")
    model_size = get_model_size(model)
    print(f"Model size (FP32): {model_size:.2f} MB")
    return model

# Configure output/tag 
cfg.tag = 'resnet_18_pageflip'
cfg.output_dir = (cfg.root / 'models' / 'resnet_18' / 'checkpoints').resolve()

# Train MobileNetV2
train_resnet18(cfg)

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
    # Train one epoch, run a no-grad validation pass and unpack metrics
    for epoch in range(1, cfg.epochs_head_classifier + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, device, criterion, optimizer, 
                                              scaler, cfg.use_amp)
        metrics = evaluate(model, val_loader, device, criterion)
        val_loss = metrics["val_loss"]; val_acc = metrics["val_accuracy"]; val_prec = metrics["val_precision"]; val_rec = metrics["val_recall"]; val_f1 = metrics["val_f1"]

        improved, w_path, f_path = checkpoint.update(
            model, val_f1, epoch,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler, metrics=metrics
        )
        if improved:
            best_validation_f1 = val_f1
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
    
        improved, w_path, f_path = checkpoint.update(
            model, val_f1, epoch + cfg.epochs_head_classifier,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler, metrics=metrics
        )
        if improved:
            best_validation_f1 = val_f1
        scheduler.step()
        print(f"[FT ] Epoch {epoch:02d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} || "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} Prec {val_prec:.4f} Rec {val_rec:.4f} F1 {val_f1:.4f}")

    print(f"Training complete. Best Val F1: {best_validation_f1:.4f}")
    model_size = get_model_size(model)
    print(f"Model size (FP32): {model_size:.2f} MB")
    return model

# Configure output/tag for EfficientNet 
cfg.tag = "efficientnet_b0_pageflip"
cfg.output_dir = (cfg.root / "models" / "efficientnet_b0" / "checkpoints").resolve()

# Train EfficientNet-B0
train_efficientnet_b0(cfg)

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
    # Loop over head-training epochs and track the best validation F1
    for epoch in range(1, cfg.epochs_head_classifier + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, device, criterion, optimizer, 
                                              scaler, cfg.use_amp)
        metrics = evaluate(model, val_loader, device, criterion)
        val_loss = metrics["val_loss"]; val_acc = metrics["val_accuracy"]; val_prec = metrics["val_precision"]; val_rec = metrics["val_recall"]; val_f1 = metrics["val_f1"]

        improved, w_path, f_path = checkpoint.update(
            model, val_f1, epoch,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler, metrics=metrics
        )
        if improved:
            best_validation_f1 = val_f1
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
        val_loss = metrics["val_loss"]; val_acc = metrics["val_accuracy"]
        val_prec = metrics["val_precision"]; val_rec = metrics["val_recall"]; val_f1 = metrics["val_f1"]

        improved, w_path, f_path = checkpoint.update(
            model, val_f1, epoch + cfg.epochs_head_classifier,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler, metrics=metrics
        )
        if improved:
            best_validation_f1 = val_f1
        scheduler.step()
        print(f"[FT ] Epoch {epoch:02d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} || Val Loss {val_loss:.4f} Acc {val_acc:.4f} Prec {val_prec:.4f} Rec {val_rec:.4f} F1 {val_f1:.4f}")

    print(f"Training complete. Best Val F1: {best_validation_f1:.4f}")
    model_size = get_model_size(model)
    print(f"Model size (FP32): {model_size:.2f} MB")
    return model

# Configure output/tag 
cfg.tag = 'mobilenet_v2_pageflip'
cfg.output_dir = (cfg.root / 'models' / 'mobilenet_v2' / 'checkpoints').resolve()

# Train MobileNetV2
train_mobilenet_v2(cfg)
