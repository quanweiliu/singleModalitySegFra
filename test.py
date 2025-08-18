# basic imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [0]))
print('using GPU %s' % ','.join(map(str, [0])))

import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ttach as tta
from datasets.OSTD_Dataset_SAR import OSTD_Dataset_SAR
from datasets.OSTD_Dataset_Image import OSTD_Dataset_Image
from datasets.ISPRS_Dataset import ISPRS_Dataset
from losses import *
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import multiprocessing.pool as mpp
import multiprocessing as mp
from models.CustomNet import CustomNet   # 神奇，就算不调用，也要import一下，否则会报错

# plot test predictions
from option import opts
from tools.utils import visualize_predictions, plot_training_results, img_writer, create_resultstable
from tools.tester import tester   # evaluation function


opt = opts.get_options()

if opt.data_name == 'OSTD_Image':
	opt.class_name = ['Water', 'oil']
	opt.bands = 193
elif opt.data_name == 'OSTD_SAR':
	opt.class_name = ['Water', 'oil']
	opt.bands = 3
elif opt.data_name == 'potsdam' or opt.data_name == 'vaihingen':
	opt.class_name = ['Impervious', 'Building', 'Vegetation', 'Tree', 'Car', 'Clutter']
elif opt.data_name == 'floodnet':
	opt.class_name = ['Background', 'Building-flooded', 'Building-non-flooded', 'Road-flooded', \
				'Road-non-flooded', 'Water', 'Tree', 'Vehicle', 'Pool', 'Grass']
        

models_dir = '/home/icclab/Documents/lqw/Multimodal_Segmentation/singleModalitySegFra/output'
model1_name = "0818-1659-unet-OHEMLoss-resnet18"


savefig_path = os.path.join(models_dir, model1_name)
opt.output_fig_path = os.path.join(savefig_path, 'figures')

# load model
MODEL_PATH = os.path.join(models_dir, model1_name, model1_name +'_best.pt')
print('Loading model from:', MODEL_PATH)

checkpoint = torch.load(MODEL_PATH, weights_only=False)
model = checkpoint['model'].to(opt.device)


if opt.data_name == "OSTD_Image":
    test_dataset = OSTD_Dataset_Image(opt, split_type='test', augmentation=False)

elif opt.data_name == "OSTD_SAR":
    test_dataset = OSTD_Dataset_SAR(opt, split_type='test', augmentation=False)

elif opt.data_name == "vaihingen":
    test_dataset = ISPRS_Dataset(opt, split_type='test', augmentation=False)

test_loader = DataLoader(test_dataset, batch_size=opt.val_batch, shuffle=False, 
                            num_workers=opt.val_worker, drop_last=False)

# plot_training_results
results_df1 = pd.DataFrame(checkpoint['results'])
plot_training_results(results_df1, opt, savefig_path=False)


# # visualize_predictions 
# num_test_samples = 2
# seed = 2345
# norm_dataset = False
# fig, axes1 = plt.subplots(num_test_samples, 4, figsize=(4*6, num_test_samples * 4))
# visualize_predictions(opt, model, test_dataset, axes1, \
#                       numTestSamples = num_test_samples, \
#                       seed = seed, norm_dataset = norm_dataset, \
#                       model_label = opt.model)
# fig.savefig(os.path.join(savefig_path, \
#             opt.model + '_noSamples' + str(num_test_samples) + '_seed' + str(seed)), \
#             bbox_inches='tight')


if opt.loss_function == 'jaccard':
    criterion = smp.losses.JaccardLoss('multiclass', log_loss = False, smooth=0.0)
elif opt.loss_function == 'dice':
    criterion = smp.losses.DiceLoss('multiclass', log_loss = False, smooth=0.0)
elif opt.loss_function == 'focal':
    criterion = smp.losses.FocalLoss('multiclass')
elif opt.loss_function == 'cross-entropy':
    criterion = torch.nn.CrossEntropyLoss()
# elif opt.loss_function == 'weighted-CE':
#     class_count = torch.zeros(opt.num_classes)
#     for i in range(len(train_dataset)):
#         class_count += torch.flatten(train_dataset[i][1]).bincount(minlength=opt.num_classes)
#         weights = (1 / class_count).to(opt.device)
#         weights = (weights / weights.sum()).to(opt.device)
#     criterion = torch.nn.CrossEntropyLoss(weight=weights)
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


tic2 = time.time()
metrics_test = tester(opt, model, test_loader)
test_time = time.time() - tic2


print(f'{opt.model} ({len(results_df1)} epochs)')

f = open((os.path.join(savefig_path, opt.model) + '.txt'), 'a+')
str_results = "\nEpoch: " + str(len(results_df1)) + \
    " \nclassAcc 0  \t" + str(round(metrics_test.class_precision()[0][0]*100, 2)) + \
    " \nclassAcc 1  \t" + str(round(metrics_test.class_precision()[0][1]*100, 2)) + \
    " \nclassAcc 2  \t" + str(round(metrics_test.class_precision()[0][2]*100, 2)) + \
    " \nclassAcc 3  \t" + str(round(metrics_test.class_precision()[0][3]*100, 2)) + \
    " \nclassAcc 4  \t" + str(round(metrics_test.class_precision()[0][4]*100, 2)) + \
    " \nOA = \t\t\t" + str(round(np.nanmean(metrics_test.OA())*100, 2)) + \
    " \nPrecision = \t" + str(round(np.nanmean(metrics_test.Precision())*100, 2)) + \
    " \nRecall = \t\t" + str(round(np.nanmean(metrics_test.Recall())*100, 2)) + \
    " \nF1 = \t\t\t" + str(round(np.nanmean(metrics_test.F1())*100, 2)) + \
    " \nKappa = \t\t" + str(round(np.nanmean(metrics_test.Kappa())*100, 2)) + \
    " \nmIOU = \t\t\t" + str(round(np.nanmean(metrics_test.Intersection_over_Union())*100, 2)) + \
    ' \ntest time = \t' + str(round(test_time, 2)) + \
    " \n"
f.write(str_results)
f.close()


# t0 = time.time()
# mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
# t1 = time.time()
# img_write_time = t1 - t0
# print('images writing spends: {} s'.format(img_write_time))
