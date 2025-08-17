import os
import cv2
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

# 神奇，就算不调用，也要import一下，否则会报错
from models.CustomNet import CustomNet, ConvBN   
from datasets.ct_dwh import load_datasets, make_loader

import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
from tools.metrics import Evaluator

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    # arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("--rgb", help="whether output rgb images", action='store_true')
    arg('--normalize', type=bool, default=True)
    arg('--norm_dataset', choices=['potsdam', 'potsdam_irrg', 'floodnet', 'vaihingen', 'imagenet', None], default=None)
    arg('--dataset', choices=["vaihingen", 'potsdam', 'floodnet', 'dwh'], default='dwh', help='Dataset the model is applied to and trained on; argument mainly used for visualization purposes')
    arg('--random_split', type=bool, default=False, help='if true, no separate valid folders are expected but train and validation in one folder, that are split randomly')
    arg('--train_batch', type=int, default=4, help='batch size for training data')
    arg('--val_batch', type=int, default=4, help='batch size for validation data')  # 我把 batch size 改的好大
    arg('--train_worker', type=int, default=0, help='number of workers for training data')
    arg('--val_worker', type=int, default=0, help='number of workers for validation data')
    arg('--patch_size', type=int, default=128, help='size of the image patches the model should be trained on')
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 255]
    return mask_rgb


def img_writer(inp):
    (mask,  mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label2rgb(mask)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


def main():
    seed_everything(42)
    args = get_args()
    # config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)


    models_dir = r'C:\Users\jc962911\Project\Semantic_Segmentation\CNNvsTransformerHSI\weights\\'
    model1_name = '0626-2224-unet-UnetFormerLoss-o'
    savefig_path = os.path.join(models_dir, model1_name)
    MODEL1_PATH = os.path.join(models_dir, model1_name, model1_name +'_best.pt')
    print('Loading model 1 from:', MODEL1_PATH)
    
    checkpoint1 = torch.load(MODEL1_PATH)
    model = checkpoint1['model']


    # model = Supervision_Train.load_from_checkpoint(\
    #         os.path.join(config.weights_path, \
    #                      config.test_weights_name+'.ckpt'), \
    #                      config=config)
    
    # print("successful here"*10)
    model.cuda()
    model.eval()
    evaluator = Evaluator(num_class=2)
    evaluator.reset()
    if args.tta == "lr":
        transforms = tta.Compose(
            [   tta.HorizontalFlip(),
                tta.VerticalFlip()])
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [   tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[90]),
                tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)])
        model = tta.SegmentationTTAWrapper(model, transforms)


    train_dataset, \
        val_dataset, \
            test_dataset = load_datasets(data_dir = r"C:\Users\jc962911\Project\datasets\MMF\OSD_H", 
                                            random_split = args.random_split, \
                                            normalize = args.normalize, \
                                            augmentation = None, \
                                            classes = args.dataset, \
                                            patch_size=args.patch_size, \
                                            dataset=args.norm_dataset)
    train_loader, \
        val_loader, \
            test_loader = make_loader(train_dataset, val_dataset, test_dataset,\
                                        args.train_batch, \
                                        args.val_batch, \
                                        args.train_worker, \
                                        args.val_worker)


    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            # num_workers=4,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        results = []

        flag = 0
        for inputs, labels in tqdm(test_loader, total=len(test_loader)):
            raw_predictions = inputs.cuda()
            masks_true = labels.cuda()
        # for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            # raw_predictions = model(input['img'].cuda())

            # image_ids = input["img_id"]
            image_ids = str(flag)
            flag += 1
            # masks_true = input['gt_semantic_seg']

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                gt=masks_true[i].cpu().numpy()

                # print(mask.shape, gt.shape)   # (128, 128) (128, 128)

                evaluator.add_batch(pre_image=mask, gt_image=gt)
                mask_name = image_ids[i]
                results.append((mask, str(args.output_path / mask_name), args.rgb))

    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    OA = evaluator.OA()

    # 为什么没有保存 test 的结果
    for class_name, class_iou, class_f1 in zip(2, iou_per_class, f1_per_class):
        print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
    print('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class[:-1]), np.nanmean(iou_per_class[:-1]), OA))
    
    # 下面是保存图片
    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))


if __name__ == "__main__":
    main()


# python test.py -o fig_results/dwh/unetformer




