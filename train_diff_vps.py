import os
import numpy as np
import torch 
import torch.nn as nn 
from torchvision import transforms

# Dataloader
from dataloader.augmentation_diffusion import *
from dataloader import dataloader_diffusion as db
import torch.nn.functional as F
from dataloader import box_ops
# Model
from network.diff_vps import DiffVPS  
from network.googlenet import Inception3
from light_training.trainer import Trainer
from monai.losses.dice import DiceLoss
from light_training.evaluation.metric import dice
from light_training.utils.files_helper import save_new_model_and_delete_last
from network.lr_scheduler import PolyLRScheduler

os.environ['MKL_THREADING_LAYER'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
torch.cuda.set_device(0)
import pdb


logdir = "./snapshot/logs/_test_"
model_save_path = "./snapshot/checkpoints/_test_"
os.makedirs(logdir, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)

env = "DDP"  # DDP or pytorch 
max_epoch = 30
batch_size = 16
val_every = 1
num_gpus = 2
device = "cuda:0"


class DiffTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
    
        self.model = DiffVPS()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=1e-5)
        
        self.netD = Inception3(num_classes=1, aux_logits=False, transform_input=True).cuda()
        checkpoint = torch.load('Diff_VPS/network/pretrained_weight/inception_v3_google-1a9a5a14.pth')
        pretrained_dict = {k:v for k,v in checkpoint.items()}
        self.netD.load_state_dict(pretrained_dict, strict=False)
        self.optimizerD = torch.optim.AdamW(self.netD.parameters(), lr=0.0001, weight_decay=1e-5)

        self.beta = 0.001
        self.best_mean_dice = 0.0
        self.warmup = 0.1
        self.scheduler_type = "cosine_with_warmup"
        self.t = None
        self.auto_optim = False
        
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)
        self.Class_CELoss = torch.nn.CrossEntropyLoss()
        self.Bbox_MSELoss = nn.MSELoss() 

    def training_step(self, batch):

        seq, frame, gt, reconstruct_gt, clas_gt, bbox_gt = batch['seq'].cuda(), batch['frame'].cuda(), \
            batch['mask'].cuda(), batch['frame'].cuda(), batch['clas'].cuda(), batch['bbox'].cuda()
        self.optimizer.zero_grad()
        self.optimizerD.zero_grad()
        
        mask_logit, pred_bbox, pred_clas, pred_frame = self.model(img=frame, seq=seq, gt_semantic_seg=gt)  # torch.Size([12, 1, 56, 56]
        
        D_real = self.netD(reconstruct_gt)
        D_fake = self.netD(pred_frame.detach())

        real_label_shape = D_real.squeeze(1).shape
        fake_label_shape = D_fake.squeeze(1).shape
        real_label = torch.ones(real_label_shape).float().cuda()
        fake_label = torch.ones(fake_label_shape).float().cuda()

        errD_real = self.bce(D_real.squeeze(1), real_label)
        errD_fake = self.bce(D_fake.squeeze(1), fake_label)
        
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        self.netD.zero_grad()
        D_loss = errD_real + errD_fake
        D_loss.backward()
        self.optimizerD.step()

        # (2) Update G network: maximize log(D(G(z)))
        self.model.zero_grad()
        D_fake = self.netD(pred_frame.detach())
        errG = self.bce(D_fake.squeeze(1), real_label)
        reconstruction_loss = self.mse(pred_frame, reconstruct_gt)
        reconstruction_branch_loss = reconstruction_loss + self.beta * errG
        # Segment Loss
        loss_dice = self.dice_loss(mask_logit, gt)  # torch.Size([12, 1, 224, 224]
        loss_bce = self.bce(mask_logit, gt)
        mask_logit = torch.sigmoid(mask_logit)
        loss_mse = self.mse(mask_logit, gt)
        denoise_loss = loss_dice + loss_bce + loss_mse
        
        # Multi-task Loss
        labels_loss = self.Class_CELoss(pred_clas, clas_gt)
        bboxes_loss = self.Bbox_MSELoss(pred_bbox.float(), bbox_gt.float())  

        loss = 0.5 * denoise_loss + 0.20 * bboxes_loss + 0.05 * labels_loss + 0.25 * reconstruction_branch_loss  # 50:25:5:20
        loss.backward()
        self.optimizer.step()
        
        self.log("denoise_loss", denoise_loss, step=self.global_step)
        self.log("labels_loss", labels_loss, step=self.global_step)
        self.log("bboxes_loss", bboxes_loss, step=self.global_step)
        self.log("total_loss", loss, step=self.global_step)
        self.log("D_loss", D_loss, step=self.global_step)
        self.log("reconstruction_loss", reconstruction_loss, step=self.global_step)
        self.log("reconstruction_branch_loss", reconstruction_branch_loss, step=self.global_step)
        
        return loss

    def validation_step(self, batch):
 
        frame, seq, gt = batch['frame'].cuda(), batch['seq'].cuda(), batch['mask'].cuda() 
        output = self.model(img=frame, seq=seq, gt_semantic_seg=gt, is_ddim=True)
        output = torch.sigmoid(output)
        output = (output > 0.5).float().cpu().numpy()

        target = gt.cpu().numpy()
        wt = dice(output, target)

        return wt

    def validation_end(self, mean_val_outputs):
        wt = mean_val_outputs

        print("dice", wt)
        self.log("dice", wt, step=self.epoch)

        if wt > self.best_mean_dice:
            self.best_mean_dice = wt
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{wt:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{wt:.4f}.pt"), 
                                        delete_symbol="final_model")

        print(f"wt is {wt}")


if __name__ == "__main__":

    statistics = torch.load("Diff_VPS/dataloader/statistics.pth")
    
    train_transforms = Compose_train([
        Resize_train(224, 224),
        toTensor_train(),
        Normalize_train(statistics["mean"], statistics["std"])
    ])
    valid_transforms = Compose_valid([
        Resize_valid(224, 224),
        toTensor_valid(),
        Normalize_valid(statistics["mean"], statistics["std"])
    ])
    timeclips=4
    train_dataset = db.TrainDataset(transform=train_transforms, timeclips=timeclips)
    valid_dataset = db.ValidDataset(samples_dataset='TestEasyDataset_Seen', transform=valid_transforms, timeclips=timeclips)
    
    trainer = DiffTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17759,
                            training_script=__file__)

    trainer.train(train_dataset=train_dataset, val_dataset=valid_dataset, scheduler=PolyLRScheduler(optimizer=trainer.optimizer, initial_lr=1e-4, max_steps=max_epoch))
    # trainer.train(train_dataset=train_dataset, val_dataset=valid_dataset, scheduler=torch.optim.lr_scheduler.MultiStepLR(trainer.optimizer, milestones=[8,15], gamma=0.1, last_epoch=-1))
    # trainer.train(train_dataset=train_dataset, val_dataset=valid_dataset)