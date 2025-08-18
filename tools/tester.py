import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from .metrics import Evaluator

def tester(
        opt,
        model : torch.nn.Module, 
        dataloader : torch.utils.data.DataLoader,   # Dataloader
        ):
    """Evaluate a model on given data

    Args:
        model (torch.nn.Module): Model to train; either of class UNet or segformer
        metric_class (_type_): metrics to evaluate the model
        dataloader (torch.utils.data.Dataloader): dataloader for test data
        num_classes (int): number of semantic classes
        device (torch.device): device to train on; e.g. "cuda:0" or "cpu"

    Returns:
        _type_: evaluation metrics
    """
    
    model.eval()
    index = 0
    metrics_val = Evaluator(opt.num_classes)
    # id_to_color, legend_elements = train_id_to_color(opt.class_name)

    results = []
    with torch.no_grad():
        for inputs, labels, image_ids in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device)      
            y_preds = model(inputs)
            # print("test", y_preds.shape, labels.shape)   # 2, 128 ,128

            predicted_labels2 = nn.Softmax(dim=1)(y_preds)
            # print("predicted_labels2 1", predicted_labels2.shape)   # torch.Size([8, 128, 128]) 
            predicted_labels2 = predicted_labels2.argmax(dim=1)
            for i in range(y_preds.shape[0]):
                # print("image_ids", image_ids)
                metrics_val.add_batch(labels[i].cpu().numpy(), predicted_labels2[i].cpu().numpy())
                pred = predicted_labels2[i].cpu().numpy()
                mask_name = image_ids[i]
                # print("mask_name", mask_name)   
                # results.append((pred, opt.output_fig_path, True))

                # print("pred", pred.shape)
                # pred = pred.reshape((256, 256))
                # cv2.imwrite(os.path.join(opt.output_fig_path, str(mask_name) + '.png'), id_to_color[pred])
                index += 1
    return  metrics_val
