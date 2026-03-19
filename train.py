import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import argparse
import os

from model import ProM3E
from dataset import ProM3EDataset

class ProM3ETrainingModule(pl.LightningModule):
    """
    PyTorch Lightning Wrapper for the ProM3E Model.
    
    This module manages the training and validation loops, optimization,
    and loss logging for the Probabilistic Multi-Modal Masked Embedding model.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize the underlying architecture
        self.model = ProM3E(
            input_dim=self.hparams.input_dim,
            embed_dim=self.hparams.embed_dim,
            depth=self.hparams.depth,
            heads=self.hparams.heads,
            mlp_dim=self.hparams.mlp_dim,
            num_modalities=self.hparams.num_modalities,
            num_register_tokens=self.hparams.num_register_tokens,
            num_cls_tokens=self.hparams.num_cls_tokens,
            masked_only=self.hparams.masked_only,
            dropout=self.hparams.dropout
        )
        
        self.lr = self.hparams.lr

    def training_step(self, batch, batch_idx):
        # Unpack batch. Note: DataLoader batch_size=1 since Dataset returns full batches.
        modalities, audio_flag = batch
        # [batch_size, 6, dim], [batch_size]
        loss = self.model(modalities[0], audio_flag[0])
        
        self.log('train_loss', loss, sync_dist=True, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        modalities, audio_flag = batch
        loss = self.model(modalities[0], audio_flag[0])
        
        self.log('val_loss', loss, sync_dist=True, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # We use AdamW as the default optimizer for transformer-based models
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            betas=(0.9, 0.98), 
            eps=1e-6
        )
        
        # Cosine Annealing with Warm Restarts for robust convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=self.hparams.scheduler_t0, 
            eta_min=1e-6
        )
        
        return [optimizer], [scheduler]

def main():
    parser = argparse.ArgumentParser(description="ProM3E: Training Script for GitHub Release")
    
    # Dataset Configuration
    parser.add_argument("--train_taxabind", type=str, required=True, help="Path to Taxabind training features (.npy)")
    parser.add_argument("--train_inat", type=str, required=True, help="Path to iNat training features (.npy)")
    parser.add_argument("--val_taxabind", type=str, required=True, help="Path to Taxabind validation features (.npy)")
    parser.add_argument("--val_inat", type=str, required=True, help="Path to iNat validation features (.npy)")
    parser.add_argument("--inat_split_size", type=int, default=80000, help="Initial split index for iNat data")
    
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=1024, help="Internal sampling size (batch size per iteration)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate for AdamW")
    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum training epochs")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader CPU workers")
    parser.add_argument("--scheduler_t0", type=int, default=20, help="T_0 parameter for CosineAnnealingWarmRestarts")
    
    # Model Architecture
    parser.add_argument("--input_dim", type=int, default=512, help="Feature dimension of each modality input")
    parser.add_argument("--embed_dim", type=int, default=1024, help="Transformer latent dimension")
    parser.add_argument("--depth", type=int, default=1, help="Number of transformer layers")
    parser.add_argument("--heads", type=int, default=8, help="Number of multi-head attention heads")
    parser.add_argument("--mlp_dim", type=int, default=2048, help="Transformer FFN hidden dimension")
    parser.add_argument("--num_modalities", type=int, default=6)
    parser.add_argument("--num_register_tokens", type=int, default=4)
    parser.add_argument("--num_cls_tokens", type=int, default=2)
    parser.add_argument("--masked_only", action="store_true", help="Toggle training on masked modalities only")
    parser.add_argument("--dropout", type=float, default=0.0)
    
    # Logging and Checkpoints
    parser.add_argument("--run_name", type=str, default="ProM3E_v1", help="Name for logging and filenames")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints", help="Folder to save model checkpoints")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    
    args = parser.parse_args()

    # Load Train/Val datasets
    train_dataset = ProM3EDataset(
        taxabind_path=args.train_taxabind,
        inat_path=args.train_inat,
        batch_size=args.batch_size,
        split='train',
        inat_split_size=args.inat_split_size
    )
    
    val_dataset = ProM3EDataset(
        taxabind_path=args.val_taxabind,
        inat_path=args.val_inat,
        batch_size=args.batch_size,
        split='val',
        inat_split_size=args.inat_split_size
    )

    # Use batch_size=1 because ProM3EDataset returns a full chunk of size args.batch_size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, 
        num_workers=args.num_workers, 
        shuffle=True, 
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        num_workers=args.num_workers, 
        shuffle=False, 
        persistent_workers=False
    )

    # Instantiate Training Module
    training_module = ProM3ETrainingModule(**vars(args))

    # Setup Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.ckpt_path,
        filename=f'prom3e-{args.run_name}' + '-{epoch:02d}-{val_loss:.4f}',
        save_last=True,
        mode='min'
    )
    
    logger = None
    if args.wandb:
        logger = WandbLogger(project="ProM3E", name=args.run_name)

    # Training logic
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        strategy='ddp_find_unused_parameters_true' if torch.cuda.device_count() > 1 else 'auto'
    )

    print(f"Starting training for {args.run_name}...")
    trainer.fit(training_module, train_loader, val_loader)

if __name__ == "__main__":
    main()
