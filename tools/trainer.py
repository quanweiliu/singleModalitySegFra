import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from .metrics import Evaluator

###################################
# FUNCTION TO EVALUATE MODEL ON DATALOADER
###################################

def evaluate_model(
        opt,
        model : torch.nn.Module, 
        dataloader : torch.utils.data.DataLoader,   # Dataloader
        criterion, 
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
    index = 0
    metrics_val = Evaluator(opt.num_classes)
    # id_to_color, legend_elements = train_id_to_color(opt.class_name)

    results = []
    with torch.no_grad():
        for inputs, labels, image_ids in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device)      
            # print("val inputs", inputs.shape)   # [8, 193, 128, 128]
            # print("val labels", labels.shape)   # [8, 128, 128]     
            y_preds = model(inputs)
            # print("val y_preds", y_preds.shape)   # 2, 6, 256, 256

            # calculate loss
            # print("test", y_preds.dtype, labels.dtype)   # 2, 128 ,128
            loss = criterion(y_preds, labels)
            total_loss += loss.item()

            predicted_labels2 = nn.Softmax(dim=1)(y_preds)
            # print("predicted_labels2 1", predicted_labels2.shape)   # torch.Size([8, 128, 128]) 
            predicted_labels2 = predicted_labels2.argmax(dim=1)
            # print("predicted_labels2 2", predicted_labels2.shape)   # torch.Size([2, 128]) 
            # output_path = r'C:\Users\jc962911\Project\Semantic_Segmentation\CNNvsTransformerHSI\fig_results'
            for i in range(y_preds.shape[0]):
                # print("image_ids", image_ids)
                metrics_val.add_batch(labels[i].cpu().numpy(), predicted_labels2[i].cpu().numpy())
                pred = predicted_labels2[i].cpu().numpy()
                mask_name = image_ids[i]
                # print("mask_name", mask_name)   
                results.append((pred, opt.output_fig_path, True))

                # print("pred", pred.shape)
                # pred = pred.reshape((256, 256))
                # cv2.imwrite(os.path.join(opt.output_fig_path, str(mask_name) + '.png'), id_to_color[pred])
                index += 1
            
    # print(len(dataloader))
    # print(total_loss)
    evaluation_loss = total_loss / len(dataloader)
    # evaluation_metric = metric_object.compute()
    return evaluation_loss, metrics_val, results


###################################
# FUNCTION TO TRAIN, VALIDATE MODEL ON DATALOADER
###################################

def train_model(
        opt,
        model : torch.nn.Module, 
        criterion, 
        optimizer : torch.optim, 
        dataloader_train : torch.utils.data.DataLoader, 
        dataloader_valid : torch.utils.data.DataLoader, 
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
    # initialize placeholders for running values    
    results = []
    max_kappa = 0
    min_val_loss = np.inf
    len_train_loader = len(dataloader_train)
    

    # 自动加载权重
    model_folder = os.path.join(opt.result_dir)
    # print("model_folder:", output_path, model_folder)
    lastmodel_path = f"{model_folder}/{opt.name}_last.pt"
    # print(lastmodel_path)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    else:
        if os.path.exists(lastmodel_path):
            print('model already exists. load last states..')
            checkpoint = torch.load(lastmodel_path)
            model.load_state_dict(checkpoint['model'].state_dict())
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            # if lr_scheduler:
            #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'].state_dict())
            results = checkpoint['results']
            
    # 保存结果
    if results:
        epochs_trained = results[-1]['epoch']+1
        # get minimum validation loss from previous training
        min_val_loss = min(results, key=lambda x:x['validationLoss'])['validationLoss'] 
        max_kappa = min(results, key=lambda x:x['Kappa'])['Kappa'] 
        best_epoch = min(results, key=lambda x:x['validationLoss'])['epoch'] 
        print(f"Best epoch: {best_epoch+1}")

        if epochs_trained >= opt.epochs:
            print(f"Existing model already trained for at least {opt.epochs} epochs")
            return  # terminate the training loop
    else:
        epochs_trained = 0
        best_epoch = -1
    

    #  TRAINING LOOP
    for epoch in range(epochs_trained, opt.epochs):
        
        print(f"Starting {epoch + 1} epoch ...")
        starttime = datetime.now()
        
        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels, _ in tqdm(dataloader_train, total=len_train_loader):
            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device) 
            # print("train inputs", inputs.shape)   # 1, 3, 256, 256
            # print("train labels", labels.shape)   # 1, 256, 256
            # print("train labels", type(labels))   # 1, 256, 256
            # Forward pass
            y_preds = model(inputs)
            # print("train y_preds", y_preds[0].dtype, labels.dtype)  
            # print("train y_preds", y_preds.shape, labels.shape)  
            loss = criterion(y_preds, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # compute per batch losses, metric value
        train_loss = train_loss / len(dataloader_train)

        duration_training = datetime.now() - starttime

        # VALIDATION LOOP
        validation_loss, metrics_val, _  = evaluate_model(opt, model, dataloader_valid, criterion)


        print(f'Epoch: {epoch+1}, trainLoss:{train_loss:6.5f}, valLoss:{validation_loss:6.5f}, trainingDuration {duration_training}')
        if opt.result_dir:
            f = open(os.path.join(opt.result_dir, opt.name) + '.txt', 'a+')
            str_results = "\nEpoch: " + str(epoch + 1) + \
                " trLoss = " + str(round(train_loss, 4)) + \
                " valLoss = " + str(round(validation_loss, 4)) + \
                " OA = " + str(round(np.nanmean(metrics_val.OA())*100, 2)) + \
                " Pre = " + str(round(np.nanmean(metrics_val.Precision())*100, 2)) + \
                " Re = " + str(round(np.nanmean(metrics_val.Recall())*100, 2)) + \
                " F1 = " + str(round(np.nanmean(metrics_val.F1())*100, 2)) + \
                " Kappa = " + str(round(np.nanmean(metrics_val.Kappa())*100, 2)) + \
                " miou = " + str(round(np.nanmean(metrics_val.Intersection_over_Union())*100, 2)) + \
                " du_training = " + str(duration_training)               
            f.write(str_results)
            f.close()

        # store results
        results.append({'epoch': epoch, 
                        'trainLoss': train_loss, 
                        'validationLoss': validation_loss, 
                        'duration_train': duration_training,
                        "Kappa": np.nanmean(metrics_val.Kappa()),
                        "mIOU": np.nanmean(metrics_val.Intersection_over_Union()),
                       })
        
        # if validation loss has decreased, save model and reset variable
        kappa = np.nanmean(metrics_val.Intersection_over_Union())
        
        if kappa >= max_kappa:
        # if validation_loss <= min_val_loss:
            max_kappa = kappa
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model': model,
                'optimizer': optimizer,
                'min_val_loss': min_val_loss,
                'max_kappa': max_kappa,
                'results': results,
            }, f"{opt.result_dir}/{opt.name}_best.pt")
            print('best model saved')

        elif opt.stop_threshold != -1:
            if epoch - best_epoch > opt.stop_threshold:
                # stop training if validation_loss did not improve for early_stop_threshold epochs
                print(f"Early stopped training at epoch {epoch} because loss did not improve for {opt.stop_threshold} epochs")
                break  # terminate the training loop


    torch.save({
        'epoch': epoch,
        'model': model,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer,
        'optimizer_state_dict': optimizer.state_dict(),
        'min_val_loss': min_val_loss,
        'max_kappa': max_kappa,
        'results': results,
    }, f"{opt.result_dir}/{opt.name}_last.pt")

    # plot results

    return results
