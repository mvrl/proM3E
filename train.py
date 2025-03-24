import open_clip
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import CLIPVisionModelWithProjection
from dataset import M3EDataset
from model import M3E
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

class M3ETrainingModule(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.model = M3E(input_projection=512,
                        dim=1024,
                        depth=6,
                        heads=8,
                        dim_head=64,
                        mlp_dim=2048,
                        decoder_dim=1024,
                        decoder_depth=4,
                        decoder_heads=8,
                        decoder_dim_head=64)
                
        self.batch_size = kwargs.get('batch_size', 1)
        self.lr = kwargs.get('lr', 1e-4)

        # self.image_embeds = torch.nn.functional.normalize(torch.from_numpy(np.load('img_embeds.npy')), dim=-1)
        # self.sat_embeds = torch.nn.functional.normalize(torch.from_numpy(np.load('sat_embeds.npy')), dim=-1)
        # self.loc_embeds = torch.nn.functional.normalize(torch.from_numpy(np.load('loc_embeds.npy')), dim=-1)
        # self.env_embeds = torch.nn.functional.normalize(torch.from_numpy(np.load('env_embeds.npy')), dim=-1)
        # self.audio_embeds = torch.nn.functional.normalize(torch.from_numpy(np.load('audio_embeds_v2.npy')), dim=-1)
        # self.text_embeds = torch.nn.functional.normalize(torch.from_numpy(np.load('text_embeds.npy')), dim=-1)

        # self.mods = torch.stack((self.image_embeds, self.sat_embeds, self.loc_embeds, self.env_embeds, torch.zeros_like(self.sat_embeds), self.audio_embeds), dim=1)
        # self.mods_text = torch.stack((torch.zeros_like(self.text_embeds), torch.zeros_like(self.text_embeds), torch.zeros_like(self.text_embeds), torch.zeros_like(self.text_embeds), self.text_embeds, torch.zeros_like(self.text_embeds)), dim=1)
        # self.classes = torch.from_numpy(np.load('sp_classes.npy'))
    

    def forward(self, modalities, audio_flag=0):
        loss = self.model(modalities.float(), audio_flag)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self(batch[0][0], batch[1])
        self.log('train_loss', loss, sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self(batch[0][0], batch[1])
        self.log('val_loss', loss, sync_dist=True, prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        return loss
    
    # def on_validation_epoch_end(self):
    #     classes = self.classes.to(self.device)

    #     image_embeds = self.model.forward_inference(self.mods.to(self.device), torch.LongTensor([1, 2, 3, 4, 5]))
    #     sat_embeds = self.model.forward_inference(self.mods.to(self.device), torch.LongTensor([0, 2, 3, 4, 5]))
    #     loc_embeds = self.model.forward_inference(self.mods.to(self.device), torch.LongTensor([0, 1, 3, 4, 5]))
    #     env_embeds = self.model.forward_inference(self.mods.to(self.device), torch.LongTensor([0, 1, 2, 4, 5]))
    #     text_embeds = self.model.forward_inference(self.mods_text.to(self.device), torch.LongTensor([0, 1, 2, 3, 5]))
    #     audio_embeds = self.model.forward_inference(self.mods.to(self.device), torch.LongTensor([0, 1, 2, 3, 4]))

    #     image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1).reshape(image_embeds.shape[0], -1)
    #     sat_embeds = torch.nn.functional.normalize(sat_embeds, dim=-1).reshape(sat_embeds.shape[0], -1)
    #     loc_embeds = torch.nn.functional.normalize(loc_embeds, dim=-1).reshape(loc_embeds.shape[0], -1)
    #     env_embeds = torch.nn.functional.normalize(env_embeds, dim=-1).reshape(env_embeds.shape[0], -1)
    #     text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1).reshape(text_embeds.shape[0], -1)
    #     audio_embeds = torch.nn.functional.normalize(audio_embeds, dim=-1).reshape(audio_embeds.shape[0], -1)

    #     sims = image_embeds @ text_embeds.t()
    #     i2t = sum(torch.argmax(sims, dim=-1)==classes.to(self.device)) / sat_embeds.shape[0]

    #     sims = sat_embeds @ loc_embeds.t()
    #     s2l = sum(torch.argmax(sims, dim=-1)==torch.arange(sat_embeds.shape[0]).to(self.device)) / sat_embeds.shape[0]
    #     l2s = sum(torch.argmax(sims.t(), dim=-1)==torch.arange(sat_embeds.shape[0]).to(self.device)) / sat_embeds.shape[0]

    #     sims = sat_embeds @ audio_embeds.t()
    #     s2a = sum(classes[torch.argmax(sims, dim=-1)]==classes.to(self.device)) / sat_embeds.shape[0]
    #     a2s = sum(classes[torch.argmax(sims.t(), dim=-1)]==classes.to(self.device)) / sat_embeds.shape[0]

    #     sims = sat_embeds @ env_embeds.t()
    #     s2e = sum(classes[torch.argmax(sims, dim=-1)]==classes.to(self.device)) / sat_embeds.shape[0]
    #     e2s = sum(classes[torch.argmax(sims.t(), dim=-1)]==classes.to(self.device)) / sat_embeds.shape[0]

    #     recall = (i2t*2+s2l+l2s+s2a+a2s+s2e+e2s)/8

    #     self.log('R_1', recall, sync_dist=True, prog_bar=True, on_epoch=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=8,
                          shuffle=True,
                          persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=8,
                          shuffle=False,
                          persistent_workers=False)

    def configure_optimizers(self):
        params = self.parameters()
        self.optim = torch.optim.AdamW(params,
                                       lr=self.lr,
                                       betas=(0.9,0.98),
                                       eps=1e-6
                                    )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=self.optim,
            T_0=10,
            eta_min=1e-6
        )
        return [self.optim], [self.scheduler]   

if __name__ == '__main__':
    data_path = 'embeds/embeds_train_final.npy'
    data_path_val = 'embeds/embeds_val.npy'
    data_path_inat = 'embeds/embeds_inat.npy'

    train_dataset = M3EDataset(data_path, data_path_inat, batch_size=1024, split='train')
    val_dataset = M3EDataset(data_path_val, data_path_inat, batch_size=1024, split='val')

    print(len(train_dataset), len(val_dataset))

    #define model
    model = M3ETrainingModule(train_dataset=train_dataset, val_dataset=val_dataset)
    torch.cuda.empty_cache()

    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='m3e-prob-decoder-{epoch:02d}-{val_loss:.2f}',
        save_last=True,
        mode='min'
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true',
        devices=1, 
        max_epochs=800,
        num_nodes=1,
        callbacks=[checkpoint],
        )
    trainer.fit(model)
    