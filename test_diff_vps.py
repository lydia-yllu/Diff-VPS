from PIL import Image
import time
import os
import numpy as np
import torch 

# Dataloader
from dataloader.augmentation_diffusion import *
from dataloader import dataloader_diffusion as db
from torch.utils.data import DataLoader
# Model
from network.diff_vps import DiffVPS, Diff_Temporal
from light_training.trainer import Trainer
from monai.losses.dice import DiceLoss
from light_training.evaluation.metric import dice
from light_training.utils.files_helper import save_new_model_and_delete_last

import argparse
from tqdm import tqdm
import prettytable as pt

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

 
# test生成的mask存放地址
def safe_save(img, save_path):
    os.makedirs(save_path.replace(save_path.split('/')[-1], ""), exist_ok=True)
    img.save(save_path)

logdir = "./snapshot/logs/test"
model_name = 'Diff_VPS'  # 'Diff_VPS'/ 'Diff_Temporal'
env = "pytorch"
max_epoch = 30
batch_size = 4
val_every = 1
num_gpus = 1
device = "cuda:0"


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
    
        if model_name == 'Diff_Temporal':
            self.model = Diff_Temporal()
        elif model_name == 'Diff_VPS':
            self.model = DiffVPS()
        self.t = None


    def validation_step(self, batch):
        seq, frame, gt, frame_path = batch['seq'].cuda(), batch['frame'].cuda(), batch['mask'].cuda(), batch["frame_path"][0]
        gt[gt == 255] = 1
        gt = gt.float()

        output = self.model(img=frame, seq=seq, gt_semantic_seg=gt, is_ddim=True)
        output = torch.sigmoid(output)
        output = (output > 0.5).float().cpu().numpy()

        data_root = "data/SUN-SEG"
        tag_dir = 'data/Pred/Diff_VPS'
        res = output  # torch.Size([1, 1, h, w])
        res = res.squeeze() # tensor->numpy (h, w)
        safe_save(Image.fromarray((res * 255).astype(np.uint8)),
                frame_path.replace(data_root, tag_dir).replace(".jpg", ".png").replace('Frame/',''))
        target = gt.cpu().numpy()
        
        wt = dice(output, target)
       
        return wt
    
    # Rewrite eval_engine_vps()
    def evaluation_step(self, opt, dataset):
        module_map_name = {"Smeasure": "Smeasure", "wFmeasure": "WeightedFmeasure", "MAE": "MAE",
                       "adpEm": "Emeasure", "meanEm": "Emeasure", "maxEm": "Emeasure",
                       "adpFm": "Fmeasure", "meanFm": "Fmeasure", "maxFm": "Fmeasure",
                       "meanSen": "Medical", "maxSen": "Medical", "meanSpe": "Medical", "maxSpe": "Medical",
                       "meanDice": "Medical", "maxDice": "Medical", "meanIoU": "Medical", "maxIoU": "Medical", 'wF1': 'weighted_f1_score'}
        res, metric_module = {}, {}
        
        metric_module_list = [module_map_name[metric] for metric in opt.metric_list]  # 通过缩写查找对应的指标
        metric_module_list = list(set(metric_module_list))  # 每种指标只用一次，对mean和max用同一个指标的情况
        
        val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        # define measures  相当于import metrics.Fmeasure等,注意"metrics"为metrics.py所在路径
        for metric_module_name in metric_module_list:
            metric_module[metric_module_name] = getattr(__import__("evaluation.metrics", fromlist=[metric_module_name]), metric_module_name)(length=len(val_loader))

        self.model.to(self.device)
        self.model.eval()
        
        txt_save_path = './Diff_VPS/evaluation/eval-result/{}/'.format(opt.txt_name)
        _data_name = opt.data_lst[0][0]
        os.makedirs(txt_save_path, exist_ok=True)
        filename = os.path.join(txt_save_path, '{}_eval.txt'.format(_data_name.replace('/', '-')))

        with open(filename, 'w+') as file_to_write:
            # initial settings for PrettyTable
            tb = pt.PrettyTable()
            names = ["Dataset"]
            names.extend(opt.metric_list)
            tb.field_names = names
            for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                frame, seq, gt, frame_path = batch['frame'].cuda(), batch['seq'].cuda(),batch['mask'].cuda(), batch["frame_path"][0]
                gt = gt.cpu().numpy().squeeze()
                gt = gt > 0.5

                output = self.model(img=frame, seq=seq, gt_semantic_seg=gt, is_ddim=True)
                output = torch.sigmoid(output)
                output = (output > 0.5).float().cpu().numpy()
                output = output.squeeze() # tensor->numpy 
                
                for name, module in metric_module.items():
                    module.step(pred=output, gt=gt, idx=idx)
                
            # metrics = ['Smeasure', 'maxEm', 'wFmeasure', 'maxDice', 'meanSen']
            for metric in opt.metric_list:
                module = metric_module[module_map_name[metric]]
                res[metric] = module.get_results()[metric]  # 返回字典{指标：值}
            final_score_list = ['{:.3f}'.format(value) for name, value in res.items()]
            tb.add_row([_data_name.replace('/', '-'), ] + list(final_score_list))
            print(tb)
            file_to_write.write(str(tb))
            file_to_write.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt_root', type=str, help='custom your ground-truth root',
        default='./data/SUN-SEG-Annotation/')
    parser.add_argument(
        '--pred_root', type=str, help='custom your prediction root',
        default='./data/Pred/')
    parser.add_argument(
        '--metric_list', type=list, help='set the evaluation metrics',
        default=['Smeasure', 'adpEm', 'wFmeasure', 'maxDice', 'meanSen'],
        choices=["Smeasure", "wFmeasure", "MAE", "adpEm", "meanEm", "maxEm", "adpFm", "meanFm", "maxFm",
                 "meanSen", "maxSen", "meanSpe", "maxSpe", "meanDice", "maxDice", "meanIoU", "maxIoU", 'wF1'])
    parser.add_argument(
        '--data_lst', type=str, help='set the dataset what you wanna to test',
        nargs='+', action='append',
        choices=['TestEasyDataset/Seen', 'TestHardDataset/Seen', 'TestEasyDataset/Unseen', 'TestHardDataset/Unseen'])
    parser.add_argument(
        '--txt_name', type=str, help='logging root',
        default='Diffusion')
    parser.add_argument(
        '--check_integrity', type=bool, help='whether to check the file integrity',
        default=True)
    parser.add_argument(
        '--data_root', type=str, help='loading root',
        default='./data/SUN-SEG')
    parser.add_argument(
        '--tag_dir', type=str, help='saving root',
        default='./data/Pred/Diff_VPS')
    opt = parser.parse_args()

    # txt_save_path = './Diff_VPS/evaluation/eval-result/{}/'.format(opt.txt_name)
    # os.makedirs(txt_save_path, exist_ok=True)
    
    statistics = torch.load("./Diff_VPS/dataloader/statistics.pth")
    test_transforms = Compose_test([
        Resize_test(224,224),
        toTensor_test(),
        Normalize_test(statistics["mean"], statistics["std"])
    ])
    
    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17759,
                            training_script=__file__)

    # Need to be changed
    checkpoint = './snapshot/checkpoints/_test_____/best_model_0.9064.pt'
    
    trainer.load_state_dict(checkpoint)
    dataset = db.TestDataset(dataset=opt.data_lst[0][0].replace('/','_'), transform=test_transforms, timeclips=4)
    
    # Directly Calculate
    trainer.evaluation(opt, dataset=dataset)

    # Save Image
    # v_mean_es, v_out_es = trainer.validation_single_gpu(val_dataset=dataset)    
    