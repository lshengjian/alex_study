import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision import transforms

class AntsAndBeesDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32,mean=[0.5072, 0.4465, 0.3269],std=[0.2759, 0.2625, 0.2804]):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.num_classes = 2
        self.dims = (3, 224, 224)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageFolder(root=self.data_dir+'/train', transform=self.transform)
            self.val_dataset = ImageFolder(root=self.data_dir+'/val', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
class LitModel(L.LightningModule):
    def __init__(self, channels, h,w, num_classes, hidden_size=128, learning_rate=2e-4):
        super().__init__()
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Conv2d(channels, 12, 5), #(224-5)/1+1=220
            nn.MaxPool2d(2, 2),  #110
            nn.Dropout(0.1),
            nn.Conv2d(12, 48, 3), #(110-3)/1+1=108
            nn.MaxPool2d(2, 2),  #54
            nn.Dropout(0.1),
            nn.Conv2d(48, 96, 3), #(54-3)/1+1=52
            nn.MaxPool2d(2, 2),  #26
            nn.Dropout(0.1),            
            nn.Flatten(),
            nn.Linear(96*26*26, hidden_size*2),
            nn.Dropout(0.1),
             nn.Linear(hidden_size*2, hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=2)
        #self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

if __name__ == '__main__':
    dm = AntsAndBeesDataModule(data_dir='d:/dataset/ants-bees/', batch_size=32)
    model = LitModel(*dm.dims,dm.num_classes)
    # Init trainer
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model, dm)
    