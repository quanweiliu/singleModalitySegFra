# basic imports
import os
import math
from datetime import datetime
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
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
from torch.optim.lr_scheduler import _LRScheduler


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

# augmentation = torch.nn.Sequential(
#     # transforms.ToTensor(),
#     transforms.ColorJitter(
#         brightness=0.5, 
#         contrast=1, 
#         saturation=0.1, 
#         hue=0.5
#     )
# )

# def normalize_images(dataset):
#     if dataset in norms.keys():
#         return transforms.Compose([
#                     transforms.Normalize(mean=np.array(norms[dataset]['mean']), \
#                                          std=np.array(norms[dataset]['std'])) # Image Net mean and std
#                 ])
#     else:
#         return transforms.Compose([
#                     transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225)) # Image Net mean and std
#                 ])


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
# METRIC CLASS DEFINITION
# intersection = np.logical_and(target, prediction) 
# union = np.logical_or(target, prediction) 
# iou_score = np.sum(intersection) / np.sum(union)

# 含义：模型对 **某一类** 别预测结果和真实值的交集与并集的比值
# 混淆矩阵计算：
# 以求二分类：正例（类别1）的IoU为例
# 交集：TP，并集：TP、FP、FN求和
# IoU = TP / (TP + FP + FN)
###################################
        
class IoU:
    """ Class to find the mean IoU using confusion matrix approach """    
    def __init__(self, num_classes):
        self.iou_metric = 0.0
        self.num_classes = num_classes
        # placeholder for confusion matrix on entire dataset | IoU 需要混淆矩阵？？
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        
    def _fast_hist(self, label_true, label_pred):
        """ Function to calculate confusion matrix on single batch """
        # mask only valid labels (this step should be irrelevant usually)
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # calculate correctness of segementation by assigning numbers and count them
        # e.g. for 6 classes [0:5], 
            # 7 is a class 2 pixel segemented correctly (6*1+1)
            # 16 is a class 3 pixel segmented as class 5 (6*2+4)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, y_preds, labels):
        """ 
            Function finds the IoU for the input batch
            and add batch metrics to overall metrics 
        """
        predicted_labels = torch.argmax(y_preds, dim=1)
        batch_confusion_matrix = self._fast_hist(labels.numpy().flatten(), \
                                                 predicted_labels.numpy().flatten())
        self.confusion_matrix += batch_confusion_matrix
    
    def compute(self, matrix = None):
        """ Computes overall meanIoU metric from confusion matrix data """ 
        hist = self.confusion_matrix
        # if a matrix is given as argument to the function, compute the metrices based on that matrix 
        if matrix:
            hist = matrix
        # divide number of pixels segmented correctly (area of overlap) 
        # by number of pixels that were segmented in this class and that should have been segmented in this class (hist.sum(axis=1) + hist.sum(axis=0))
        # minus 1 time the pixels segmented correctly in the denominator as they are in both sums
        # IoU = TP / (TP + FP + FN)
        # TP = np.diag(hist); FP = hist.sum(axis=0) - np.diag(hist); FN = hist.sum(axis=1) - np.diag(hist) ?
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)) 
        # calculate mean of IoU per class
        mean_iu = np.nanmean(iu)
        # calculate accuracy
        accuracy = np.diag(hist).sum() / hist.sum().sum()
        # class_accuracy = (np.diag(hist) + (hist.sum().sum() - hist.sum(axis=1) - hist.sum(axis=0) + np.diag(hist))) / (hist.sum().sum())
        # calculate dice coefficient / f1 score
        f1 = 2*np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0))
        meanf1 = np.nanmean(f1)
        # return {'hist' : hist, 'accuracy' : accuracy, 'classwise_accuracy' : class_accuracy, 'miou' : mean_iu, 'classwise_iou' : iu}
        return {'accuracy' : accuracy, 'miou' : mean_iu, \
                'classwise_iou' : iu, 'classwise_f1': f1, \
                'f1_mean': meanf1, 'matrix': hist}

    def reset(self):
        self.iou_metric = 0.0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        

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
    print('"row predicted as column"')
    # display 函数是 IPython 的一个内置函数，
    # 它用于在 Jupyter Notebook 环境中显示 Python 对象的图形化表示或其他格式化输出，
    # 例如图像、音频、视频、HTML 等。
    # display(df)
    return df

###################################
# FUNCTION TO PLOT TRAINING, VALIDATION CURVES
###################################

# myFmt = DateFormatter("%M:%S")
def plot_training_results(df, model_name):
    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.set_ylabel('trainLoss', color='tab:red')
    ax1.plot(df['epoch'].values, df['trainLoss'].values, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()  
    ax2.set_ylabel('validationLoss', color='tab:blue')
    ax2.plot(df['epoch'].values, df['validationLoss'].values, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    ax3 = ax1.twinx()  
    ax3.set_ylabel('trainingTime(sec)', color='tab:orange', labelpad=-32)
    # ax3.yaxis.set_major_formatter(myFmt)
    ax3.tick_params(axis="y",direction="in", pad=-23)
    ax3.plot(df['epoch'].values, df['duration_train'].dt.total_seconds(), color='tab:orange')
    ax3.tick_params(axis='y', labelcolor='tab:orange')

    plt.suptitle(f'{model_name} Training, Validation Curves')
    plt.show()




###################################
# FUNCTION TO EVALUATE MODEL ON DATALOADER
###################################

def evaluate_model(
        model : torch.nn.Module, 
        dataloader : torch.utils.data.DataLoader,   # Dataloader
        criterion, 
        metric_class, # class IoU
        num_classes : int, 
        device : torch.device
        ):
    """Evaluate a model on given data

    Args:
        model (torch.nn.Module): Model to train; either of class UNet or segformer
        criterion (): loss function, e.g. smp.losses.JaccardLoss
        metric_class (_type_): metrics to evaluate the model
        dataloader (torch.utils.data.Dataloader): dataloader for test data
        num_classes (int): number of semantic classes
        device (torch.device): device to train on; e.g. "cuda:0" or "cpu"

    Returns:
        _type_: evaluation metrics
    """
    
    model.eval()
    total_loss = 0.0
    metric_object = metric_class(num_classes)

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            inputs, labels = batch['img'], batch['gt_semantic_seg']
            inputs = inputs.to(device)
            labels = labels.to(device)      
            # print("val inputs", inputs.shape)   # 2, 3, 256, 256
            # print("val labels", labels.shape)   # 2, 256, 256          
            y_preds = model(inputs)
            # print("val y_preds", y_preds.shape)   # 2, 6, 256, 256

            # calculate loss
            # print("test", y_preds.shape, labels.shape)   # 2, 6, 256, 256
            loss = criterion(y_preds, labels)
            total_loss += loss.item()

            # update batch metric information            
            metric_object.update(y_preds.cpu().detach(), labels.cpu().detach())
            
    print(len(dataloader))
    print(total_loss)

    evaluation_loss = total_loss / len(dataloader)
    evaluation_metric = metric_object.compute()
    return evaluation_loss, evaluation_metric


###################################
# FUNCTION TO TRAIN, VALIDATE MODEL ON DATALOADER
###################################

def train_validate_model(
        model : torch.nn.Module, 
        num_epochs : int, 
        model_name : str, 
        criterion, 
        optimizer : torch.optim, 
        device : torch.device, 
        dataloader_train : torch.utils.data.DataLoader, 
        dataloader_valid : torch.utils.data.DataLoader, 
        metric_class, 
        num_classes : int, 
        lr_scheduler = None,
        output_path : str = '.', 
        early_stop : int = -1,
        opt = None
        ):
    """Train and validate a model

    Args:
        model (torch.nn.Module): Model to train; either of class UNet or segformer
        num_epochs (int): number of epochs to train
        model_name (str): name to save the model
        criterion (): loss function, e.g. smp.losses.JaccardLoss
        optimizer (torch.optim): Optimizer, e.g. Adam
        device (torch.device): device to train on; e.g. "cuda:0" or "cpu"
        dataloader_train (torch.utils.data.Dataloader): dataloader for training data
        dataloader_valid (torch.utils.data.Dataloader): dataloader for validation data
        metric_class (_type_): metrics to evaluate the model
        num_classes (int): number of semantic classes
        lr_scheduler (_type_, optional): learning rate scheduler; 
            e.g. torch.optim.lr_scheduler.OneCycleLR . Defaults to None.
        output_path (str, optional): Directory to save the model at. Defaults to '.'.
        early_stop (int, optional): Number of epochs for an early stopping of the training. 
            I.e. after the number of epochs given here without an improvement in the validation loss, 
            the training is stopped. Defaults to -1.

    Returns:
        pd.Dataframe: evaluation metrics
    """
    early_stop_threshold = early_stop
    
    # initialize placeholders for running values    
    results = []
    min_val_loss = np.Inf
    len_train_loader = len(dataloader_train)
    

    # 自动加载权重
    model_folder = os.path.join(output_path, model_name)
    lastmodel_path = f"{model_folder} / {model_name}_last.pt"
    # print(lastmodel_path)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    else:
        if os.path.exists(lastmodel_path):
            print('model already exists. load last states..')
            checkpoint = torch.load(lastmodel_path)
            model.load_state_dict(checkpoint['model'].state_dict())
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            if lr_scheduler:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'].state_dict())
            results = checkpoint['results']
            
    # 保存结果
    if results:
        epochs_trained = results[-1]['epoch']+1
        # get minimum validation loss from previous training
        min_val_loss = min(results, key=lambda x:x['validationLoss'])['validationLoss'] 
        best_epoch = min(results, key=lambda x:x['validationLoss'])['epoch'] 
        print(f"Best epoch: {best_epoch+1}")

        if epochs_trained >= num_epochs:
            print(f"Existing model already trained for at least {num_epochs} epochs")
            return  # terminate the training loop
    else:
        epochs_trained = 0
        best_epoch = -1
    
    # move model to device
    model.to(device)
    
    for epoch in range(epochs_trained, num_epochs):
        # epoch = epoch + epochs_trained
        
        print(f"Starting {epoch + 1} epoch ...")
        starttime = datetime.now()
        
        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(dataloader_train, total=len_train_loader):
            inputs, labels = batch['img'], batch['gt_semantic_seg']
            inputs = inputs.to(device)
            labels = labels.to(device) 
            # print("train inputs", inputs.shape)   # 1, 3, 256, 256
            # print("train labels", labels.shape)   # 1, 256, 256
            
            # Forward pass
            y_preds = model(inputs)
            # print("train y_preds", y_preds.shape)  # 4, 6, 256, 256
            loss = criterion(y_preds, labels)
            train_loss += loss.item()
              
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # adjust learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()
            
        # compute per batch losses, metric value
        train_loss = train_loss / len(dataloader_train)

        endtime_train = datetime.now()
        validation_loss, validation_metric = evaluate_model(
                        model, dataloader_valid, criterion, metric_class, num_classes, device)
        # endtime_val = datetime.now()
        duration_training = endtime_train - starttime
        
        print(f'Epoch: {epoch+1}, trainLoss:{train_loss:6.5f}, validationLoss:{validation_loss:6.5f}, validation_metrices: {validation_metric}, trainingDuration {duration_training}')
        if opt.result_dir:
            f = open(os.path.join(opt.result_dir, opt.name) + '.txt', 'a+')
            str_results = "\nEpoch: " + str(epoch + 1) + \
                " trainLoss = " + str(round(train_loss, 5)) + \
                " validationLoss = " + str(round(validation_loss, 5)) + \
                " accuracy = " + str(validation_metric["accuracy"].round(5)) + \
                " miou = " + str(validation_metric["miou"].round(5)) + \
                " f1_mean = " + str(validation_metric["f1_mean"].round(5)) + \
                " duration_training = " + str(duration_training)
            f.write(str_results)
            f.close()
        # store results
        results.append({'epoch': epoch, 
                        'trainLoss': train_loss, 
                        'validationLoss': validation_loss, 
                        'metrices': validation_metric,
                        'duration_train': duration_training,
                       })
        
        torch.save({
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer,
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler,
            # 'scheduler_state_dict': lr_scheduler.state_dict(),
            'min_val_loss': min_val_loss,
            'results': results,
            'epoch': epoch,
        }, f"{output_path}/{model_name}/{model_name}_last.pt")
        
        # if validation loss has decreased, save model and reset variable
        if validation_loss <= min_val_loss:
            min_val_loss = validation_loss
            best_epoch = epoch
            torch.save({
                'model': model,
                # 'model_state_dict': model.state_dict(),
                'optimizer': optimizer,
                # 'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler,
                # 'scheduler_state_dict': lr_scheduler.state_dict(),
                'min_val_loss': min_val_loss,
                'results': results,
                'epoch': epoch,
            }, f"{output_path}/{model_name}/{model_name}_best.pt")
            print('best model saved')
        elif early_stop_threshold != -1:
            if epoch - best_epoch > early_stop_threshold:
                # stop training if validation_loss did not improve for early_stop_threshold epochs
                print(f"Early stopped training at epoch {epoch} because loss did not improve for {early_stop_threshold} epochs")
                break  # terminate the training loop
        
    # plot results
    results = pd.DataFrame(results)
    plot_training_results(results, model_name)
    return results


###################################
# FUNCTION TO VISUALIZE MODEL PREDICTIONS
###################################


def train_id_to_color(classes):
    # a tuple with name, train_id, color, more pythonic and more easily readable. 
    Label = namedtuple( "Label", [ "name", "train_id", "color"])
    # print(len(classes))
    if len(classes) == 3:
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


# legend_elements = [
#     Patch(facecolor=train_id_to_color[0]/255, label=drivables[0].name),  
#     Patch(facecolor=train_id_to_color[1]/255, label=drivables[1].name),
#     Patch(facecolor=train_id_to_color[2]/255, label=drivables[2].name),
#     Patch(facecolor=train_id_to_color[3]/255, label=drivables[3].name),
#     Patch(facecolor=train_id_to_color[4]/255, label=drivables[4].name),
#     Patch(facecolor=train_id_to_color[5]/255, label=drivables[5].name),
#                   ]

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
        # inputImage, gt = dataSet[sampleID]
        # print(dataSet[sampleID]['img'].shape, dataSet[sampleID]['gt_semantic_seg'].shape)
        # 224, 128, 128 / 128, 128
        inputImage, gt = dataSet[sampleID]['img'], dataSet[sampleID]['gt_semantic_seg']

        # input rgb image   
        inputImage = inputImage.to(device)

        # 为什么这里要用 inverse_transform？因为绘图的时候不要用归一化的数据！！！
        if norm_dataset: 
            inv_norm = inverse_transform(norm_dataset)
            landscape = inv_norm(inputImage).permute(1, 2, 0)[:, :, 16:19].cpu().detach().numpy()
        else: 
            landscape = inputImage.permute(1, 2, 0)[:, :, 16:19].cpu().detach().numpy()
        print(landscape.shape)
            
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


# basic imports
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


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




