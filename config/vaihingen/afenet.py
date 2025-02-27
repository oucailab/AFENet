from torch.utils.data import DataLoader
from afeseg.losses import *
from afeseg.datasets.vaihingen_dataset import *
from afeseg.models.AFENet import AFENet
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 100
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "AFENet-res18-e{}".format(max_epoch)
weights_path = "model_weights/potsdam/{}".format(weights_name)
test_weights_name = "AFENet-res18-e{}".format(max_epoch)
log_name = 'potsdam/{}'.format(weights_name)

monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = AFENet(num_classes=num_classes)

# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)


train_dataset = VaihingenDataset(data_root='data/vaihingen/train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(data_root='data/vaihingen/val',transform=val_aug)
test_dataset = VaihingenDataset(data_root='data/vaihingen/test',
                                transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=8,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

