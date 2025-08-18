# %%
# basic imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [0]))
print('using GPU %s' % ','.join(map(str, [0])))

import json
import argparse
import pandas as pd
from datetime import datetime
from thop import profile, clever_format

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.CustomNet import CustomNet
from datasets.OSTD_Dataset_SAR import OSTD_Dataset_SAR
from datasets.OSTD_Dataset_Image import OSTD_Dataset_Image
from datasets.ISPRS_Dataset import ISPRS_Dataset
from losses import *

from option import opts
from tools.utils import plot_training_results    # train validate function
from tools import trainer
import segmentation_models_pytorch as smp
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def main(opt):

    if opt.data_name == "OSTD_Image":
        train_dataset = OSTD_Dataset_Image(opt, split_type='train', augmentation=True)
        val_dataset = OSTD_Dataset_Image(opt, split_type='val', augmentation=False)

    elif opt.data_name == "OSTD_SAR":
        train_dataset = OSTD_Dataset_SAR(opt, split_type='train', augmentation=True)
        val_dataset = OSTD_Dataset_SAR(opt, split_type='val', augmentation=False)

    elif opt.data_name == "vaihingen":
        train_dataset = ISPRS_Dataset(opt, split_type='train', augmentation=True)
        val_dataset = ISPRS_Dataset(opt, split_type='val', augmentation=False)

    train_loader = DataLoader(train_dataset, batch_size=opt.train_batch, shuffle=True, 
                              num_workers=opt.train_worker, prefetch_factor=4, 
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.val_batch, shuffle=False, 
                            num_workers=opt.val_worker, prefetch_factor=4, 
                            pin_memory=True, drop_last=True)
    
    for image, label, _ in train_dataset:
        print("image", image.shape, image.dtype, image.max(), image.min())
        print("label", label.shape, label.dtype, label.max(), label.min())
        break

    for image, label, _ in val_dataset:
        print("image", image.shape, image.dtype, image.max(), image.min())
        print("label", label.shape, label.dtype, label.max(), label.min())
        break

    # create model
    model = CustomNet(opt, bands=opt.bands).to(opt.device)
    
    # set loss function
    # reference : https://smp.readthedocs.io/en/latest/losses.html
    if opt.loss_function == 'jaccard':
        criterion = smp.losses.JaccardLoss('multiclass', log_loss = False, smooth=0.0)
    elif opt.loss_function == 'dice':
        criterion = smp.losses.DiceLoss('multiclass', log_loss = False, smooth=0.0)
    elif opt.loss_function == 'focal':
        criterion = smp.losses.FocalLoss('multiclass')
    elif opt.loss_function == 'cross-entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif opt.loss_function == 'weighted-CE':
        class_count = torch.zeros(opt.num_classes)
        for i in range(len(train_dataset)):
            class_count += torch.flatten(train_dataset[i][1]).bincount(minlength=opt.num_classes)
            weights = (1 / class_count).to(opt.device)
            weights = (weights / weights.sum()).to(opt.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    # ignore_index Label that indicates ignored pixels (does not contribute to loss)
    elif opt.loss_function == 'UnetFormerLoss':  # (都用这个没有关系)
        # 这个损失有 ignore index 也就是在最外层补充的类别，但是专门为 长度为 2 的输出设置了辅助损失。所以用这个损失的batch size 不能是 2
        criterion = UnetFormerLoss(ignore_index=opt.num_classes)
    elif opt.loss_function == 'UnetFormerLossMamba':  # (都用这个没有关系)
        # 这个来源于 mamba 库
        criterion = UnetFormerLoss_mamba(ignore_index=opt.num_classes)
        opt.use_aux_loss = True
    elif opt.loss_function == 'JointLoss':
        # 这个损失有 ignore index 也就是在最外层补充的类别，但是专门为 长度为 2 的输出设置了辅助损失。所以用这个损失的batch size 不能是 2
        criterion = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=opt.num_classes),
                        DiceLoss(smooth=0.05, ignore_index=opt.num_classes), 1.0, 1.0)
    elif opt.loss_function == 'OHEMLoss':
        criterion = OHEM_CELoss(thresh=0.7, ignore_index=opt.num_classes)
    elif opt.loss_function == 'abcLoss':
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


    results = trainer.train_model(opt, model, train_loader, val_loader, criterion, optimizer, lr_scheduler)
    results = pd.DataFrame(results)
    plot_training_results(results, opt, savefig_path=False)


if __name__ == '__main__':
    opt = opts.get_options()

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

    if opt.data_name == 'OSTD_Image':
        opt.class_name = ['Water', 'oil']
        opt.bands = 193
    elif opt.data_name == 'OSTD_SAR':
        opt.class_name = ['Water', 'oil']
        opt.bands = 3
    elif opt.data_name == 'potsdam' or opt.data_name == 'vaihingen':
        opt.class_name = ['Impervious', 'Building', 'Vegetation', 'Tree', 'Car', 'Clutter']
        opt.bands = 3
    elif opt.data_name == 'floodnet':
        opt.class_name = ['Background', 'Building-flooded', 'Building-non-flooded', 'Road-flooded', \
                   'Road-non-flooded', 'Water', 'Tree', 'Vehicle', 'Pool', 'Grass']
    
    main(opt)


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