# basic imports
import os
import math
import cv2
import time
import tifffile
import numpy as np
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.dates import DateFormatter
from collections import namedtuple
from PIL import Image
# DL library imports

import torch
import torch.nn as nn
from torchvision import transforms
# import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
# from torch.optim.lr_scheduler import _LRScheduler

# import multiprocessing.pool as mpp
# import multiprocessing as mp

from .metrics import Evaluator
###################################
# FILE CONSTANTS
###################################

# Convert to torch tensor and normalize images using Imagenet values
preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
                ])

norms = {
    'imagenet': {'mean':(0.485, 0.456, 0.406), 'std':(0.229, 0.224, 0.225)},
    'potsdam': {'mean':(0.349, 0.371, 0.347), 'std':(0.1196, 0.1164, 0.1197)},
    'potsdam_irrg': {'mean':(0.3823, 0.3625, 0.3364), 'std':(0.1172, 0.1167, 0.1203)},
    'floodnet': {'mean':(0.4159, 0.4499, 0.3466), 'std':(0.1297, 0.1197, 0.1304)},
    'vaihingen': {'mean':(0.4731, 0.3206, 0.3182), 'std':(0.1970, 0.1306, 0.1276)},
}


# when using torch datasets we defined earlier, the output image
# is normalized. So we're defining an inverse transformation to 
# transform to normal RGB format
def inverse_transform(dataset):
    if dataset in norms.keys():
        return transforms.Compose([
            transforms.Normalize(-np.array(norms[dataset]['mean']) / np.array(norms[dataset]['std']), \
                                 1/np.array(norms[dataset]['std']))
    ])
    else:
        return transforms.Compose([
            transforms.Normalize((-0.485/0.229, -0.56/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
    ])


###################################

# calculate recall, precision , f1 from confusion matrix
def create_resultstable(metrices, classes):
    # get matrix from results
    # results = results_df.metrices[len(results_df)-1] 
    results = metrices
    matrix = results['matrix']
    # calculate recall and respective values for wrong segmentations
    matrix_per = matrix / matrix.sum(axis = 1)[np.newaxis].T # transpose sum to apply divison to rows
    recall = np.diag(matrix_per)
    
    # other metrices
    precision = np.diag(matrix / matrix.sum(axis = 0))
    f1 = 2*precision*recall / (precision+recall)
    
    # classes as captions for table
    CLASSES = classes
    
    # make df and add column names and ids
    df = pd.DataFrame(matrix_per)
    df.columns = [s + '_pred' for s in CLASSES]
    df.index = [s + '_tr' for s in CLASSES]
    # add metrices
    df.loc['Precision'] = precision
    df.loc['Recall'] = recall
    df.loc['f1'] = f1
    df.loc['IoU'] = results['classwise_iou']
    df.loc['(overall next)'] = '-'
    df.loc['Accuracy'] = results['accuracy']
    df.loc['mIoU'] = results['miou']
    
    # print('Metrices')
    print(f'Accuracy: {results["accuracy"]}, MeanIoU: {results["miou"]}') 
          #, f1 score mean: {results["f1_mean"]}')
    # print('"row predicted as column"')
    # display 函数是 IPython 的一个内置函数，
    # 它用于在 Jupyter Notebook 环境中显示 Python 对象的图形化表示或其他格式化输出，
    # 例如图像、音频、视频、HTML 等。
    # display(df)
    return df

###################################
# FUNCTION TO PLOT TRAINING, VALIDATION CURVES
###################################

# myFmt = DateFormatter("%M:%S")
def plot_training_results(df, opt, savefig_path=False):
    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.set_ylabel('trainLoss', color='tab:red')
    ax1.plot(df['epoch'].values, df['trainLoss'].values, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()  
    ax2.set_ylabel('validationLoss', color='tab:blue')
    ax2.plot(df['epoch'].values, df['validationLoss'].values, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    # ax3 = ax1.twinx()  
    # ax3.set_ylabel('trainingTime(sec)', color='tab:orange', labelpad=-32)
    # # ax3.yaxis.set_major_formatter(myFmt)
    # ax3.tick_params(axis="y",direction="in", pad=-23)
    # ax3.plot(df['epoch'].values, df['duration_train'].dt.total_seconds(), color='tab:orange')
    # ax3.tick_params(axis='y', labelcolor='tab:orange')

    ax4 = ax1.twinx()  
    ax4.set_ylabel('kappa', color='tab:orange', labelpad=-232)
    ax4.tick_params(axis="y",direction="in", pad=-203)
    ax4.plot(df['epoch'].values, df['Kappa'].values, color='tab:orange')
    ax4.tick_params(axis='y', labelcolor='tab:orange')

    # ax5 = ax1.twinx()  
    # ax5.set_ylabel('mIou', color='tab:green', labelpad=-82)
    # ax5.tick_params(axis="y",direction="in", pad=-83)
    # ax5.plot(df['epoch'].values, df['mIOU'].values, color='tab:green')
    # ax5.tick_params(axis='y', labelcolor='tab:green')

    plt.suptitle(f'{opt.name} Training, Validation Curves')
    if savefig_path:
        plt.savefig(os.path.join(opt.result_dir, "metric"), bbox_inches='tight')
    else:
        plt.show()

def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 255]
    return mask_rgb


def img_writer(inp):
    (mask,  mask_id, rgb) = inp
    print("mask", mask.shape)
    if rgb:
        mask_name_tif = mask_id + '.png'
        # print("mask_png 1", mask_png.shape)
        mask_tif = label2rgb(mask)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        # print("mask_png 2", mask_png.shape)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


###################################
# FUNCTION TO VISUALIZE MODEL PREDICTIONS
###################################
def train_id_to_color(classes):
    # a tuple with name, train_id, color, more pythonic and more easily readable. 
    Label = namedtuple( "Label", [ "name", "train_id", "color"])
    # print(len(classes))
    if len(classes) == 2:
        # print("into here")
        drivables = [ 
            Label(classes[0], 0, (255, 255, 0)), 
            Label(classes[1], 1, (0, 0, 255)),
            # Label(classes[2], 2, (255, 0, 0))
        ]
    elif len(classes) == 3:
        # print("into here")
        drivables = [ 
            Label(classes[0], 0, (255, 255, 0)), 
            Label(classes[1], 1, (0, 0, 255)),
            Label(classes[2], 2, (255, 0, 0))
        ]
    elif len(classes) == 6:
        drivables = [ 
            Label(classes[0], 0, (255, 255, 255)), 
            Label(classes[1], 1, (0, 0, 255)), 
            Label(classes[2], 2, (0, 255, 255)), 
            Label(classes[3], 3, (0, 255, 0)), 
            Label(classes[4], 4, (255, 255, 0)), 
            Label(classes[5], 5, (255, 0, 0))
        ]
    elif len(classes) == 10:
        drivables = [ 
            Label(classes[0], 0, (0, 0, 0)), 
            Label(classes[1], 1, (255, 0, 0)), 
            Label(classes[2], 2, (200, 90, 90)), 
            Label(classes[3], 3, (130, 130, 0)), 
            Label(classes[4], 4, (150, 150, 150)), 
            Label(classes[5], 5, (0, 255, 255)),
            Label(classes[6], 6, (0, 0, 255)), 
            Label(classes[7], 7, (255, 0, 255)), 
            Label(classes[8], 8, (250, 250, 0)), 
            Label(classes[9], 9, (0, 255, 0)) 
        ]
    else:
        return
    # print("drivables", drivables)
    id_to_color = [c.color for c in drivables if (c.train_id != -1 and c.train_id != 255)]
    id_to_color = np.array(id_to_color)
    # print("id_to_color", id_to_color)
    
    legend_elements = []
    for i, c in enumerate(classes):
        # A **Patch** is a 2D artist with a face color and an edge color.If any of edgecolor, facecolor, linewidth, 
        # or antialiased are None, they default to their rc params setting.
        # 个人理解就是这个类是其他2D 图形类的支持类，用以控制各个图形类的公有属性。
        legend_elements.append(Patch(facecolor = id_to_color[i]/255, label=c))
        
    return id_to_color, legend_elements


diff_legend = [
    Patch(facecolor='#00fa00', label='True'), 
    Patch(facecolor='#c80000', label='False'), 
]

def visualize_predictions(model : torch.nn.Module,
                          dataSet : Dataset,
                          axes,
                          device :torch.device,
                          numTestSamples : int,
                          # id_to_color : np.ndarray = train_id_to_color, 
                          seed : int = None, 
                          norm_dataset = 'own', 
                          # rgb = True, 
                          classes=None,
                          model_label=""   # just a name
                         ):
    """Function visualizes predictions of input model on samples from the provided dataset.
    Shows input image, next to ground truth, prediction and their difference.

    Args:
        model (torch.nn.Module): model whose output we're to visualize
        dataSet (Dataset): dataset to take samples from
        device (torch.device): compute device as in GPU, CPU etc
        numTestSamples (int): number of samples to plot
        id_to_color (np.ndarray) : array to map class to colormap
        seed (int) : random seed to control the selected samples
        norm_dataset (String) : select between one of 'imagenet', 'potsdam', 'potsdam_irrg', 
            'floodnet', 'vaihingen' to apply respective normalization to the images; 
            default 'own' applies (false) imagenet normalization
        classes : array with classes of the dataset; currently implemented ISPRS and 
            FloodNet datasets with 6 and 10 classes respectively
        model_label (String) : text that should be added to the figure title
    """
    model.to(device=device)
    model.eval()

    rgcmap = colors.ListedColormap(['green','red'])
    np.random.seed(seed)

    # predictions on random samples
    # print(f"Candiditure visualizing samples of {len(dataSet)}")
    testSamples = np.random.choice(len(dataSet), numTestSamples).tolist()
    # print(testSamples)
    # _, axes = plt.subplots(numTestSamples, 3, figsize=(3*6, numTestSamples * 4))
    
    # legend_elements 这个只记录了 label 信息，提供了标注信息

    # print("classes", classes)
    id_to_color, legend_elements = train_id_to_color(classes)
    for handle in legend_elements:
        if handle.get_label() == 'Impervious':
            handle.set_edgecolor("gray")
    id_to_rg = np.array([[200, 0, 0], [0, 250, 0]])
    
    # 逐个预测并绘制图片
    # for i, sampleID in enumerate(testSamples):
    for i, sampleID in enumerate(testSamples):
        # print(dataSet[sampleID].keys())
        inputImage, gt, _ = dataSet[sampleID]
        print(inputImage.shape, gt.shape)
        # print(dataSet[sampleID]['img'].shape, dataSet[sampleID]['gt_semantic_seg'].shape)
        # 224, 128, 128 / 128, 128
        # inputImage, gt = dataSet[sampleID], dataSet[sampleID]

        # input rgb image   
        inputImage = inputImage.to(device)

        # 为什么这里要用 inverse_transform？因为绘图的时候不要用归一化的数据！！！
        if norm_dataset: 
            inv_norm = inverse_transform(norm_dataset)
            landscape = inv_norm(inputImage).permute(1, 2, 0)[:, :, 16:19].cpu().detach().numpy()
        else: 
            landscape = inputImage.permute(1, 2, 0)[:, :, 16:19].cpu().detach().numpy()
        # print(landscape.shape)
            
        axes[i, 0].imshow(landscape)
        axes[i, 0].set_title(dataSet.get_name(sampleID))

        # groundtruth label image
        label_class = gt.cpu().detach().numpy()
        # print(label_class)
        axes[i, 1].imshow(id_to_color[label_class])
        axes[i, 1].set_title("Groundtruth Label")

        # predicted label image
        y_pred = torch.argmax(model(inputImage.unsqueeze(0)), dim=1).squeeze(0)
        label_class_predicted = y_pred.cpu().detach().numpy()    
        axes[i, 2].imshow(id_to_color[label_class_predicted])
        axes[i, 2].legend(handles=legend_elements, loc = 'upper left', bbox_to_anchor=(-0.7, 0.9))
        axes[i, 2].set_title("Prediction " + model_label)

        # difference groundtruth and prediction
        # very smart
        diff = label_class == label_class_predicted
        axes[i, 3].imshow(id_to_rg[diff*1]) #, cmap = rgcmap) # make int to map 0 and 1 to cmap, otherwise a 
        axes[i, 3].legend(handles=diff_legend)
        axes[i, 3].set_title("Correctness " + model_label)
        # print(diff * 1)
        # issue (solved?): if the whole image is predicted wrong, 
        # it is visualized green (probably because imshow simply takes first color from cmap?)
    for ax in axes.reshape(-1): 
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    

def compare_models_onOneImage(model1 : torch.nn.Module, 
                              model2 : torch.nn.Module, 
                              dataset: Dataset,
                              im_name : str,  
                              device : torch.device, 
                              normalization = True, 
                              classes = None,
                              norm_dataset : str = 'own',
                              model1_label : str = 'U-Net',
                              model2_label : str = 'SegFormer',
                              plot_title : str = None
                             ):
    """Function visualizes predictions of two input models on one sample from the provided dataset.
    Shows input image, ground truth and the prediction and difference per model. 

    Args:
        model1 (torch.nn.Module): first model whose output we're to visualize
        model2 (torch.nn.Module): second model whose output we're to visualize
        dataSet (Dataset): dataset to take samples from
        im_name (str) :  name of the image the models are be applied to
        device (torch.device): compute device as in GPU, CPU etc
        classes : array with classes of the dataset; currently implemented ISPRS and 
            FloodNet datasets with 6 and 10 classes respectively
        norm_dataset (String) : select between one of 'imagenet', 'potsdam', 'potsdam_irrg', 
            'floodnet', 'vaihingen' to apply respective normalization to the images; 
            default 'own' applies (false) imagenet normalization
        model_label1 (String) : text that should be added to the figure title for the first model
        model_label2 (String) : text that should be added to the figure title for the second model
        plot_title (String) : title of the whole figure
    """
    _, axes = plt.subplots(2, 3, figsize=(4*5, 3 * 3))
    
    if plot_title:
        _.suptitle(plot_title, fontsize=20)
        
    plt.rcParams.update({'font.size': 12})
    if len(classes) == 10:
        plt.rcParams.update({'font.size': 11})
    
    model1.to(device=device)
    model1.eval()
    
    model2.to(device=device)
    model2.eval()
    
    #################
    # testSamples = np.random.choice(len(dataSet), numTestSamples).tolist()
    # get image from the dataset by its file name
    imId = dataset.get_id_by_name(im_name)
    inputImage, gt = dataset[imId]
    
    # set colors and legend
    id_to_color, legend_elements = train_id_to_color(classes)
    for handle in legend_elements:
        if handle.get_label() == 'Impervious':
            handle.set_edgecolor("gray")
    id_to_rg = np.array([[200, 0, 0], [0, 250, 0]])

    # input rgb image   
    inputImage = inputImage.to(device)
    if norm_dataset: 
        inv_norm = inverse_transform(norm_dataset)
        landscape = inv_norm(inputImage).permute(1, 2, 0).cpu().detach().numpy()
    else: 
        landscape = inputImage.permute(1, 2, 0).cpu().detach().numpy()
    axes[0, 0].imshow(landscape)
    axes[0, 0].set_title('Input Image')

    # groundtruth label image
    label_class = gt.cpu().detach().numpy()
    axes[1, 0].imshow(id_to_color[label_class])
    axes[1, 0].set_title("Groundtruth Label")

    # predicted label image
    y_pred1 = torch.argmax(model1(inputImage.unsqueeze(0)), dim=1).squeeze(0)
    label_class_predicted1 = y_pred1.cpu().detach().numpy()    
    axes[0, 1].imshow(id_to_color[label_class_predicted1])
    axes[0, 1].set_title("Prediction "+model1_label)

    # difference groundtruth and prediction
    diff = label_class == label_class_predicted1
    axes[0, 2].imshow(id_to_rg[diff*1])#, cmap = rgcmap) # make int to map 0 and 1 to cmap, otherwise a 
    axes[0, 2].set_title("Correctness "+model1_label)
    
    # predicted label image
    y_pred2 = torch.argmax(model2(inputImage.unsqueeze(0)), dim=1).squeeze(0)
    label_class_predicted2 = y_pred2.cpu().detach().numpy()    
    axes[1, 1].imshow(id_to_color[label_class_predicted2])
    axes[1, 1].legend(handles=legend_elements, loc = 'upper left', bbox_to_anchor=(-0.6, 1.4))
    axes[1, 1].set_title("Prediction "+model2_label)

    # difference groundtruth and prediction
    diff = label_class == label_class_predicted2
    axes[1, 2].imshow(id_to_rg[diff*1])#, cmap = rgcmap) # make int to map 0 and 1 to cmap, otherwise a 
    axes[1, 2].legend(handles=diff_legend, loc = 'upper left', bbox_to_anchor=(-0.5, 1.2))
    axes[1, 2].set_title("Correctness "+model2_label)
    
    for ax in axes.reshape(-1): 
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    
    return _


#######################################
# Data processing utils
#######################################
# Now replace RGB to integer values to be used as labels.
# Find pixels with combination of RGB for the above defined arrays...
# if matches then replace all values in that pixel with a specific integer
def rgb_to_2D_label(label):
    """
    Suply our label masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    Impervious = [255, 255, 255]
    Building = [0, 0, 255]
    Vegetation = [0, 255, 255]
    Tree = [0, 255, 0]
    Car = [255, 255, 0]
    Clutter = [255, 0, 0]

    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label==Impervious,axis=-1)] = 0
    label_seg [np.all(label==Building,axis=-1)] = 1
    label_seg [np.all(label==Vegetation,axis=-1)] = 2
    label_seg [np.all(label==Tree,axis=-1)] = 3
    label_seg [np.all(label==Car,axis=-1)] = 4
    label_seg [np.all(label==Clutter,axis=-1)] = 5

    # label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg

