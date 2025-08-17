import torch
import argparse

def get_options():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, default=None, help='name of the model as it should be saved')
	parser.add_argument('--data_name', type=str, default='OSTD_Image', 
						choices=['OSTD_SAR', 'OSTD_Image', 'vaihingen', "potsdam"],
						help='Dataset the model is applied to and trained on')
	parser.add_argument('--data_path', type=str, default="/home/icclab/Documents/lqw/DatasetMMF/OSTD", \
						help='path were the input data is stored')
	parser.add_argument('--patch_size', type=int, default=128, help='size of the image patches the model should be trained on')
	parser.add_argument('--model', choices=['unet', 'segformer', 'segformer-b5', 'AMSUnet', "MANet"], 
						default='unet', help="the model architecture that should be trained")
	
	parser.add_argument('--epochs', type=int, default=10, help='epochs the model should be trained')
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
	parser.add_argument('--encoder', type=str, default="resnet18", help='')
	parser.add_argument('--activation', type=str, default="softmax2d", \
						help='could be None for logits or softmax2d for multiclass segmentation | sigmoid')
	parser.add_argument('--result_dir', type=str, default='/home/icclab/Documents/lqw/Multimodal_Segmentation/singleModalitySemanticSegmentation/output', 
						help='path to directory where the results should be stored')
	parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='')

	# opt = parser.parse_args()
	opt = parser.parse_args('')

	return opt
