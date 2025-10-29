import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any
import time

from utils.collate_fn import collate_fn
from datasets.single_class_dataset import SingleClassDetectionDataset

# ---- Model Registry ----
from models.faster_rcnn import get_faster_rcnn_model
from models.yolov11 import get_yolo11n
from models.ssd import get_ssd_vgg16
from models.retinanet import get_retinanet

MODEL_REGISTRY = {
    "fasterrcnn": lambda num_classes: get_faster_rcnn_model(num_classes),
    "yolo11n": lambda num_classes: get_yolo11n(num_classes),
    "ssd": lambda num_classes: get_ssd_vgg16(num_classes),
    "retinanet": lambda num_classes: get_retinanet(num_classes),
}


class Trainer:
    """Universal trainer for object detection models."""
    
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        device: str = None
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        
        self.model = MODEL_REGISTRY[model_name](num_classes).to(self.device)
        
        # Loss tracking
        self.train_losses = []
        self.best_loss = float('inf')
    
    def train_one_epoch(
        self, 
        dataloader: DataLoader, 
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dict of average losses
        """
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'box': 0.0,
            'cls': 0.0,
            'obj': 0.0
        }
        num_batches = len(dataloader)
        
        for batch_idx, (imgs, targets) in enumerate(dataloader):
            # Move to device
            imgs = [img.to(self.device) for img in imgs]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = self.model(imgs, targets)
            
            # Parse loss based on model type
            loss_dict = self._parse_loss(outputs)
            total_loss = loss_dict['loss']
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            # Accumulate losses
            epoch_losses['total'] += total_loss.item()
            if 'loss_box' in loss_dict:
                epoch_losses['box'] += loss_dict['loss_box'].item()
            if 'loss_cls' in loss_dict:
                epoch_losses['cls'] += loss_dict['loss_cls'].item()
            if 'loss_obj' in loss_dict:
                epoch_losses['obj'] += loss_dict['loss_obj'].item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"  Batch [{batch_idx+1}/{num_batches}] "
                      f"Loss: {total_loss.item():.4f}")
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def _parse_loss(self, outputs: Any) -> Dict[str, torch.Tensor]:
        """
        Parse model outputs to extract losses.
        
        Handles different model output formats:
        - Faster R-CNN: dict with 'loss_classifier', 'loss_box_reg', etc.
        - YOLO: dict with 'loss', 'loss_box', 'loss_cls', 'loss_obj'
        - SSD/RetinaNet: similar to Faster R-CNN
        """
        if not isinstance(outputs, dict):
            if torch.is_tensor(outputs):
                return {'loss': outputs}
            raise ValueError(f"Unsupported output type: {type(outputs)}")
        
        # Faster R-CNN style
        if 'loss_classifier' in outputs:
            total_loss = sum(loss for loss in outputs.values())
            return {
                'loss': total_loss,
                'loss_box': outputs.get('loss_box_reg', torch.tensor(0.0)),
                'loss_cls': outputs.get('loss_classifier', torch.tensor(0.0)),
                'loss_obj': outputs.get('loss_objectness', torch.tensor(0.0))
            }
        
        # YOLO style
        elif 'loss' in outputs:
            return {
                'loss': outputs['loss'],
                'loss_box': outputs.get('loss_box', torch.tensor(0.0)),
                'loss_cls': outputs.get('loss_cls', torch.tensor(0.0)),
                'loss_obj': outputs.get('loss_obj', torch.tensor(0.0))
            }
        
        # Generic: sum all losses
        else:
            total_loss = sum(loss for loss in outputs.values())
            return {'loss': total_loss}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dict of validation losses
        """
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'box': 0.0,
            'cls': 0.0,
            'obj': 0.0
        }
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for imgs, targets in dataloader:
                imgs = [img.to(self.device) for img in imgs]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # For validation, some models need to be in train mode to compute loss
                self.model.train()
                outputs = self.model(imgs, targets)
                self.model.eval()
                
                loss_dict = self._parse_loss(outputs)
                
                val_losses['total'] += loss_dict['loss'].item()
                if 'loss_box' in loss_dict:
                    val_losses['box'] += loss_dict['loss_box'].item()
                if 'loss_cls' in loss_dict:
                    val_losses['cls'] += loss_dict['loss_cls'].item()
                if 'loss_obj' in loss_dict:
                    val_losses['obj'] += loss_dict['loss_obj'].item()
        
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def save_checkpoint(self, save_path: Path, epoch: int, optimizer: torch.optim.Optimizer):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': self.train_losses,
            'best_loss': self.best_loss
        }
        torch.save(checkpoint, save_path)
        print(f"💾 Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: Path, optimizer: torch.optim.Optimizer = None):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"✅ Checkpoint loaded from {checkpoint_path}")
        return checkpoint.get('epoch', 0)


def train_model(
    model_name: str = "yolo11n",
    dataset_root: str = "data/sausage",
    num_classes: int = 2,
    img_size: int = 640,
    batch_size: int = 8,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.1,
    save_every: int = 10,
    resume: str = None
):
    """
    Universal training function for object detection.
    
    Args:
        model_name: Model architecture name
        dataset_root: Path to dataset
        num_classes: Number of classes (including background for some models)
        img_size: Input image size
        batch_size: Batch size
        epochs: Number of epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        val_split: Validation split ratio
        save_every: Save checkpoint every N epochs
        resume: Path to checkpoint to resume from
    """
    print("=" * 70)
    print(f"🚀 Training {model_name.upper()} on {dataset_root}")
    print("=" * 70)
    
    # Setup
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Dataset
    print("\n📂 Loading dataset...")
    full_dataset = SingleClassDetectionDataset(
        os.path.join(dataset_root, "images"),
        os.path.join(dataset_root, "labels"),
        img_size=img_size
    )
    
    # Train/Val split
    if val_split > 0:
        train_size = int(len(full_dataset) * (1 - val_split))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        print(f"   Train: {train_size} images | Val: {val_size} images")
    else:
        train_dataset = full_dataset
        val_dataset = None
        print(f"   Train: {len(train_dataset)} images (no validation)")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for debugging, increase for speed
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
    
    # Trainer
    print(f"\n🏗️  Building {model_name} model...")
    trainer = Trainer(model_name, num_classes)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        trainer.model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs, 
        eta_min=lr * 0.01
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if resume:
        start_epoch = trainer.load_checkpoint(Path(resume), optimizer)
    
    # Training loop
    print(f"\n🎯 Training for {epochs} epochs...")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Device: {trainer.device}")
    print("-" * 70)
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        print(f"\n📈 Epoch [{epoch+1}/{epochs}]")
        
        # Train
        train_losses = trainer.train_one_epoch(train_loader, optimizer)
        trainer.train_losses.append(train_losses)
        
        # Validate
        if val_loader:
            val_losses = trainer.validate(val_loader)
            print(f"\n   Validation Loss: {val_losses['total']:.4f}")
        
        # Print summary
        epoch_time = time.time() - epoch_start
        print(f"\n   Summary:")
        print(f"   ├─ Total Loss: {train_losses['total']:.4f}")
        print(f"   ├─ Box Loss:   {train_losses['box']:.4f}")
        print(f"   ├─ Cls Loss:   {train_losses['cls']:.4f}")
        print(f"   ├─ Obj Loss:   {train_losses['obj']:.4f}")
        print(f"   ├─ LR:         {optimizer.param_groups[0]['lr']:.6f}")
        print(f"   └─ Time:       {epoch_time:.1f}s")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = output_dir / f"{model_name}_epoch_{epoch+1}.pth"
            trainer.save_checkpoint(checkpoint_path, epoch + 1, optimizer)
        
        # Save best model
        if train_losses['total'] < trainer.best_loss:
            trainer.best_loss = train_losses['total']
            best_path = output_dir / f"{model_name}_best.pth"
            torch.save(trainer.model.state_dict(), best_path)
            print(f"   ⭐ New best model saved! Loss: {trainer.best_loss:.4f}")
    
    # Save final model
    final_path = output_dir / f"{model_name}_final.pth"
    torch.save(trainer.model.state_dict(), final_path)
    
    print("\n" + "=" * 70)
    print(f"✅ Training completed!")
    print(f"   Best loss: {trainer.best_loss:.4f}")
    print(f"   Final model: {final_path}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Universal Object Detection Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model
    parser.add_argument("--model", type=str, default="yolo11n",
                       choices=list(MODEL_REGISTRY.keys()),
                       help="Model architecture")
    
    # Data
    parser.add_argument("--dataset", type=str, default="data/sausage",
                       help="Dataset root path")
    parser.add_argument("--num_classes", type=int, default=2,
                       help="Number of classes (including background)")
    parser.add_argument("--img_size", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Validation split ratio")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    # Checkpointing
    parser.add_argument("--save_every", type=int, default=10,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        dataset_root=args.dataset,
        num_classes=args.num_classes,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        save_every=args.save_every,
        resume=args.resume
    )