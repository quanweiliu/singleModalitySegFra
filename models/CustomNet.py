import argparse


import torch
from torch import nn
import segmentation_models_pytorch as smp

from .unet import UNet
from .MANet import MANet
from .A2FPN import A2FPN
from .ABCNet import ABCNet
from .BANet import BANet
from .UNetFormer import UNetFormer
from .DCSwin import dcswin_base, dcswin_small, dcswin_tiny
from .FTUNetFormer import FTUNetFormer, ft_unetformer, ft_unetformer_hsi

from .segformer import segformer_mit_b3 as segformer
from .AMSUnet import AMSUnet

from .config import get_config

# from unet import UNet
# from MANet import MANet
# from A2FPN import A2FPN
# from ABCNet import ABCNet
# from BANet import BANet
# from UNetFormer import UNetFormer
# from DCSwin import dcswin_base, dcswin_small, dcswin_tiny
# from FTUNetFormer import FTUNetFormer, ft_unetformer, ft_unetformer_hsi

# from segformer import segformer_mit_b3 as segformer
# from AMSUnet import AMSUnet

# from vision_mamba import MambaUnet, MambaSwin, MambaRes, Mambamamba
# from config import get_config


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                    dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            nn.BatchNorm2d(out_channels))


class CustomNet(nn.Module):
    def __init__(self, opt, bands=3):
        super(CustomNet, self).__init__()

        self.bands = bands
        self.pre_conv = ConvBN(bands, 3, kernel_size=3)
        
        # create model
        if opt.model == 'unet':
            self.model = UNet(in_channels=3, out_channels=opt.num_classes, layer_channels=[64, 128, 256, 512]).to(opt.device)
        elif opt.model == 'segformer':
            self.model = segformer(in_channels=3, num_classes=opt.num_classes).to(opt.device)
        elif opt.model == 'segformer-b5':
            self.model = segformer(in_channels=3, num_classes=opt.num_classes, depths=(3, 6, 40, 3)).to(opt.device)
        elif opt.model == 'unetformer':
            self.model = UNetFormer(num_classes=opt.num_classes, window_size=4).to(opt.device)
        elif opt.model == 'A2FPN':
            self.model = A2FPN(class_num=opt.num_classes).to(opt.device)
        elif opt.model == 'ABCNet':
            self.model = ABCNet(n_classes=opt.num_classes).to(opt.device)
        elif opt.model == 'BANet':
            self.model = BANet(num_classes=opt.num_classes, weight_path="/home/leo/Semantic_Segmentation/CNNvsTransformerHSI/pretrain_weights/rest_lite.pth").to(opt.device)
        elif opt.model == 'DCSwin':
            self.model = dcswin_tiny(num_classes=opt.num_classes, weight_path="/home/leo/Semantic_Segmentation/CNNvsTransformerHSI/pretrain_weights/stseg_tiny.pth").to(opt.device)
            # self.model = dcswin_small(num_classes=opt.num_classes, weight_path="/home/leo/Semantic_Segmentation/CNNvsTransformerHSI/pretrain_weights/stseg_small.pth").to(opt.device)
            # self.model = dcswin_base(num_classes=opt.num_classes, weight_path="/home/leo/Semantic_Segmentation/CNNvsTransformerHSI/pretrain_weights/stseg_base.pth").to(opt.device)
        elif opt.model == 'FTUNetFormer':
            # self.model = FTUNetFormer(num_classes=opt.num_classes, window_size=4).to(opt.device)
            # net = ft_unetformer(num_classes=1, weight_path="/home/leo/Semantic_Segmentation/CNNvsTransformerHSI/pretrain_weights/stseg_base.pth").to(opt.device)
            self.model = ft_unetformer_hsi(num_classes=opt.num_classes, weight_path="/home/leo/Semantic_Segmentation/CNNvsTransformerHSI/pretrain_weights/stseg_base.pth").to(opt.device)
        elif opt.model == 'MANet':
            self.model = MANet(num_classes=opt.num_classes).to(opt.device)
        elif opt.model == 'AMSUnet':
            self.model = AMSUnet(in_channels=3, num_classes=opt.num_classes, base_c=32).to(opt.device)

        elif opt.model == "ss-Unet":
            self.model = smp.Unet(encoder_name=opt.encoder, encoder_weights=opt.encoder_weights, classes=opt.num_classes, activation=opt.activation).to(opt.device)
        elif opt.model == "ss-UnetPlusPlus":
            self.model = smp.UnetPlusPlus(encoder_name=opt.encoder, encoder_weights=opt.encoder_weights, classes=opt.num_classes, activation=opt.activation).to(opt.device)
        elif opt.model == "ss-FPN":
            self.model = smp.FPN(encoder_name=opt.encoder, encoder_weights=opt.encoder_weights, classes=opt.num_classes, activation=opt.activation).to(opt.device)
        elif opt.model == "ss-PSPNet":
            self.model = smp.PSPNet(encoder_name=opt.encoder, encoder_weights=opt.encoder_weights, classes=opt.num_classes, activation=opt.activation).to(opt.device)
        elif opt.model == "ss-DeepLabV3":
            self.model = smp.DeepLabV3(encoder_name=opt.encoder, encoder_weights=opt.encoder_weights, classes=opt.num_classes, activation=opt.activation).to(opt.device)
        elif opt.model == "ss-DeepLabV3Plus":
            self.model = smp.DeepLabV3Plus(encoder_name=opt.encoder, encoder_weights=opt.encoder_weights, classes=opt.num_classes, activation=opt.activation).to(opt.device)
        elif opt.model == "ss-Linknet":
            self.model = smp.Linknet(encoder_name=opt.encoder, encoder_weights=opt.encoder_weights, classes=opt.num_classes, activation=opt.activation).to(opt.device)
        elif opt.model == "ss-MAnet":
            self.model = smp.MAnet(encoder_name=opt.encoder, encoder_weights=opt.encoder_weights, classes=opt.num_classes, activation=opt.activation).to(opt.device)
        elif opt.model == "ss-PAN":
            self.model = smp.PAN(encoder_name=opt.encoder, encoder_weights=opt.encoder_weights, classes=opt.num_classes, activation=opt.activation).to(opt.device)
        else:
            raise ValueError(f"Unknown model: {opt.model}")

    def forward(self, x):

        if self.bands > 3:
            x = self.pre_conv(x)
        return self.model(x)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None, help='name of the model as it should be saved')
    parser.add_argument('--model', type=str, default='mamba', help="model architecture that should be trained")
    # | train_images, train_masks, test_images, test_masks, valid_images, valid_masks
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='')
    parser.add_argument('--num_classes', type=int, default=3, help='class number')

    opt = parser.parse_args()
    # opt = Namespace(model='unet', num_classes=16)

    model = CustomNet(opt)
    # print(model)



