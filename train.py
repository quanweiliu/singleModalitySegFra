# %%
# basic imports
import os
import json
import argparse
import numpy as np
from datetime import datetime
from thop import profile, clever_format

# from utils import load_datasets, augmentation, make_loader
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.CustomNet import CustomNet
from datasets.OSTD_Dataset import OSTD_Dataset
from losses import *

from tools.utils import train_model, evaluate_model     # train validate function
import segmentation_models_pytorch as smp
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default=None, help='name of the model as it should be saved')
parser.add_argument('--data_path', type=str, default="/home/icclab/Documents/lqw/DatasetMMF/OSTD", \
                    help='path were the input data is stored')
parser.add_argument('--patch_size', type=int, default=128, help='size of the image patches the model should be trained on')
parser.add_argument('--random_split', type=bool, default=False, \
                    help='if true, no separate valid folders are expected but train and validation in one folder, that are split randomly')
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--norm_dataset', choices=['potsdam', 'potsdam_irrg', 'floodnet', 'vaihingen', 'imagenet', None], default=None)
parser.add_argument('--model', choices=['unet', 'segformer', 'segformer-b5'], default='segformer', 
                    help="the model architecture that should be trained")
parser.add_argument('--epochs', type=int, default=100, help='epochs the model should be trained')

parser.add_argument('--train_batch', type=int, default=16, help='batch size for training data')
parser.add_argument('--val_batch', type=int, default=2, help='batch size for validation data')  # 我把 batch size 改的好大
parser.add_argument('--train_worker', type=int, default=6, help='number of workers for training data')
parser.add_argument('--val_worker', type=int, default=4, help='number of workers for validation data')
parser.add_argument('--stop_threshold', type=int, default=-1, \
                    help='number of epochs without improvement in validation loss after that the training should be stopped')

parser.add_argument('--loss_function', type=str, default='OHEMLoss', 
                    choices=['dice', 'jaccard', 'focal', 'cross-entropy', 'weighted-CE'],
                    help='loss function that should be used for training the model')
parser.add_argument('--lr', type=float, default=3e-3, help='maximum learning rate')
parser.add_argument('--lr_scheduler', type=bool, default=True, help='wether to use the implemented learning rate scheduler or not')
parser.add_argument('--use_aux_loss', type=bool, default=False, help='wether to use the use_aux_loss or not')
parser.add_argument('--num_classes', type=int, default=2, help='number of semantic classes of the dataset')
parser.add_argument('--dataset', choices=["vaihingen", 'potsdam', 'floodnet', 'dwh'], default='dwh', \
                    help='Dataset the model is applied to and trained on; argument mainly used for visualization purposes')
parser.add_argument('--encoder', type=str, default="resnet18", help='')
parser.add_argument('--encoder_weights', type=str, default="imagenet", help='')
parser.add_argument('--activation', type=str, default="softmax2d", \
                    help='could be None for logits or softmax2d for multiclass segmentation | sigmoid')
parser.add_argument('--result_dir', type=str, default='/home/icclab/Documents/lqw/Multimodal_Segmentation/singleModalitySemanticSegmentation/output', 
                    help='path to directory where the results should be stored')
parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='')

opt = parser.parse_args()


def main():

    train_dataset = OSTD_Dataset(opt, split_type='train', augmentation=True)
    val_dataset = OSTD_Dataset(opt, split_type='val', augmentation=False)
    test_dataset = OSTD_Dataset(opt, split_type='test', augmentation=False)

    train_loader = DataLoader(train_dataset, batch_size=opt.train_batch, shuffle=True, num_workers=opt.train_worker, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.val_batch, shuffle=False, num_workers=opt.val_worker, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=opt.val_batch, shuffle=False, num_workers=opt.val_worker)


#     # for (ti, tm, _), (vi, vm, _), (tei, tem, _) in zip(train_loader, val_loader, test_loader):
#     #     # 3, 256, 256 | 256, 256
#     #     # 3, 512, 512 | 512, 512
#     #     print("ti", ti.shape, "tm", tm.shape, "vi", vi.shape, "vm", vm.shape, "tei", tei.shape, "tem", tem.shape)
#     #     print(np.unique(tm), np.unique(vm), np.unique(tem))
#     #     print(torch.max(tm), torch.max(vm), torch.max(tem))
#     #     print(torch.min(tm), torch.min(vm), torch.min(tem))
#     #     break

    # create model
    model = CustomNet(opt).to(opt.device)
    

    # set loss function
    # reference : https://smp.readthedocs.io/en/latest/losses.html
    if opt.loss_function == 'jaccard':
        criterion = smp.losses.JaccardLoss('multiclass', log_loss = False, smooth=0.0)
    if opt.loss_function == 'dice':
        criterion = smp.losses.DiceLoss('multiclass', log_loss = False, smooth=0.0)
    if opt.loss_function == 'focal':
        criterion = smp.losses.FocalLoss('multiclass')
    if opt.loss_function == 'cross-entropy':
        criterion = torch.nn.CrossEntropyLoss()
    if opt.loss_function == 'weighted-CE':
        class_count = torch.zeros(opt.num_classes)
        for i in range(len(train_dataset)):
            class_count += torch.flatten(train_dataset[i][1]).bincount(minlength=opt.num_classes)
            weights = (1 / class_count).to(opt.device)
            weights = (weights / weights.sum()).to(opt.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    # ignore_index Label that indicates ignored pixels (does not contribute to loss)
    if opt.loss_function == 'UnetFormerLoss':  # (都用这个没有关系)
        # 这个损失有 ignore index 也就是在最外层补充的类别，但是专门为 长度为 2 的输出设置了辅助损失。所以用这个损失的batch size 不能是 2
        criterion = UnetFormerLoss(ignore_index=opt.num_classes)
    if opt.loss_function == 'UnetFormerLossMamba':  # (都用这个没有关系)
        # 这个来源于 mamba 库
        criterion = UnetFormerLoss_mamba(ignore_index=opt.num_classes)
        opt.use_aux_loss = True
    if opt.loss_function == 'JointLoss':
        # 这个损失有 ignore index 也就是在最外层补充的类别，但是专门为 长度为 2 的输出设置了辅助损失。所以用这个损失的batch size 不能是 2
        criterion = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=opt.num_classes),
                        DiceLoss(smooth=0.05, ignore_index=opt.num_classes), 1.0, 1.0)
    if opt.loss_function == 'OHEMLoss':
        criterion = OHEM_CELoss(thresh=0.7, ignore_index=opt.num_classes)
    if opt.loss_function == 'abcLoss':
        # criterion = functools.partial(multi_loss2)
        criterion = ABCLoss()

    # create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    if opt.lr_scheduler: 
        # lr_scheduler = OneCycleLR(optimizer, max_lr= MAX_LR, epochs = N_EPOCHS, \
        #                           steps_per_epoch = len(train_loader), 
        #                           pct_start=0.3, div_factor=10, anneal_strategy='cos')
        lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.6, patience=8, min_lr=0.000001)
    else:
        lr_scheduler = None


#     # # FLOPs and Params
#     # # input = torch.randn(2, 1, hsi_bands+sar_bands, args.patch_size, args.patch_size).to(args.device)
#     # input = torch.randn(2, 193, 128, 128).to(opt.device)
#     # print(input.shape)

#     # flops, params = profile(model, inputs=(input,))
#     # # flops, params = profile(model, inputs=(torch.randn(2, hsi_bands).to(args.device),  torch.randn(2, sar_bands).to(args.device)))

#     # flops, params = clever_format([flops, params])
#     # print('# Model FLOPs: {}'.format(flops))
#     # print('# Model Params: {}'.format(params))


    _ = train_model(opt, model, criterion, optimizer, train_loader, val_loader)



if __name__ == '__main__':

    # create folder for output if not exists
    opt.name = datetime.now().strftime("%m%d-%H%M-") + \
                opt.model + '-' + opt.loss_function + '-' + opt.encoder
    print("name: ", opt.name)

    opt.result_dir = os.path.join(opt.result_dir, opt.name)
    print("result_dir: ", opt.result_dir)

    # create folder for output if not exists
    if not os.path.exists(opt.result_dir):
        os.mkdir(opt.result_dir)
        opt.output_fig_path = os.path.join(opt.result_dir, 'figures')
        os.mkdir(opt.output_fig_path)

    with open(opt.result_dir + '/args.json', 'w') as fid:
        json.dump(opt.__dict__, fid, indent=2)

    if opt.dataset == 'dwh':
        opt.class_name = ['Water', 'oil']
    if opt.dataset == 'potsdam' or opt.dataset == 'vaihingen':
        opt.class_name = ['Impervious', 'Building', 'Vegetation', 'Tree', 'Car', 'Clutter']
    if opt.dataset == 'floodnet':
        opt.class_name = ['Background', 'Building-flooded', 'Building-non-flooded', 'Road-flooded', \
                   'Road-non-flooded', 'Water', 'Tree', 'Vehicle', 'Pool', 'Grass']
    
    main()



# 这里测试不同的数据（193， 224）和损失函数（UnetFormerLoss， OHEMLoss）
# python train.py --model unet --encoder o --loss_function UnetFormerLoss
# python train.py --model segformer --encoder o --loss_function UnetFormerLoss
# python train.py --model unetformer --encoder o --loss_function UnetFormerLoss
# python train.py --model A2FPN --encoder o --loss_function UnetFormerLoss
# python train.py --model MANet --encoder o --loss_function UnetFormerLoss
# python train.py --model BANet --encoder o --loss_function UnetFormerLoss
# python train.py --model DCSwin --encoder o --loss_function UnetFormerLoss
# python train.py --model AMSUnet --encoder resnet18 --loss_function UnetFormerLoss


# python train.py --model unet --encoder o --loss_function OHEMLoss
# python train.py --model segformer --encoder o --loss_function OHEMLoss
# python train.py --model unetformer --encoder o --loss_function OHEMLoss
# python train.py --model A2FPN --encoder o --loss_function OHEMLoss
# python train.py --model MANet --encoder o --loss_function OHEMLoss
# python train.py --model BANet --encoder o --loss_function OHEMLoss
# python train.py --model DCSwin --encoder o --loss_function OHEMLoss
# python train.py --model AMSUnet --encoder resnet18 --loss_function OHEMLoss


# python train.py --model ABCNet --encoder o --loss_function abcLoss --output_path /home/leo/Semantic_Segmentation/CNNvsTransformerHSI/weights_other
# python train.py --model mamba --encoder o --loss_function OHEMLoss --output_path /home/leo/Semantic_Segmentation/CNNvsTransformerHSI/weights_other
# python train.py --model mamba --encoder o --loss_function UnetFormerLossMamba --output_path /home/leo/Semantic_Segmentation/CNNvsTransformerHSI/weights_other

# 这里的损失函数由上面的结果决定， 测试不同的基础模型
# python train.py --model ss-Unet --encoder resnet18 --loss_function OHEMLoss --output_path /home/leo/Semantic_Segmentation/CNNvsTransformerHSI/weights_Backbone
# python train.py --model ss-UnetPlusPlus --encoder resnet18 --loss_function OHEMLoss --output_path /home/leo/Semantic_Segmentation/CNNvsTransformerHSI/weights_Backbone
# python train.py --model ss-DeepLabV3 --encoder resnet18 --loss_function OHEMLoss --output_path /home/leo/Semantic_Segmentation/CNNvsTransformerHSI/weights_Backbone
# python train.py --model ss-DeepLabV3Plus --encoder resnet18 --loss_function OHEMLoss --output_path /home/leo/Semantic_Segmentation/CNNvsTransformerHSI/weights_Backbone
# python train.py --model ss-Linknet --encoder resnet18 --loss_function OHEMLoss --output_path /home/leo/Semantic_Segmentation/CNNvsTransformerHSI/weights_Backbone


# python train.py --model ss-Unet --encoder efficientnet-b0 --loss_function OHEMLoss --output_path /home/leo/Semantic_Segmentation/CNNvsTransformerHSI/weights_Backbone
# python train.py --model ss-UnetPlusPlus --encoder efficientnet-b0 --loss_function OHEMLoss --output_path /home/leo/Semantic_Segmentation/CNNvsTransformerHSI/weights_Backbone
# python train.py --model ss-DeepLabV3 --encoder efficientnet-b0 --loss_function OHEMLoss --output_path /home/leo/Semantic_Segmentation/CNNvsTransformerHSI/weights_Backbone
# python train.py --model ss-DeepLabV3Plus --encoder efficientnet-b0 --loss_function OHEMLoss --output_path /home/leo/Semantic_Segmentation/CNNvsTransformerHSI/weights_Backbone
# python train.py --model ss-Linknet --encoder efficientnet-b0 --loss_function OHEMLoss --output_path /home/leo/Semantic_Segmentation/CNNvsTransformerHSI/weights_Backbone




# python train.py --model unetformer --encoder o --loss_function UnetFormerLoss --output_path /home/leo/Semantic_Segmentation/CNNvsTransformerHSI/weights_UFloss