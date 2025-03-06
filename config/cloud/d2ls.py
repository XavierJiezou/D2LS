from torch.utils.data import DataLoader
from network.losses import *
from network.datasets.cloud_dataset import *
from network.models.d2ls import DynamicDictionaryLearning 
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 100
ignore_index = 255
train_batch_size = 4
val_batch_size = 4
lr = 1e-4
weight_decay = 0.01
backbone_lr = 0.001
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
token_length = num_classes 
classes = CLASSES

weights_name = "d2ls"
weights_path = "checkpoints/cloud/{}".format(weights_name)
test_weights_name = weights_name
log_name = 'cloud/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = [0]  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None #"checkpoints/cloud/delta-0817l0.6/delta-0817l0.6.ckpt"  # whether continue training with the checkpoint, default None
strategy = None

#  define the network
net = DynamicDictionaryLearning(
    model="convnext_base",
    token_length=token_length,
    l=3,
)
# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)

use_aux_loss = True

# define the dataloader

train_dataset = CloudDataset(data_root='data/cloud', img_dir='img_dir', mask_dir='ann_dir',
                             mode='train', mosaic_ratio=0.25, transform=train_aug, img_size=(512, 512))

val_dataset = CloudDataset(data_root='data/cloud', img_dir='img_dir', mask_dir='ann_dir',
                             mode='test', mosaic_ratio=0.25, transform=train_aug, img_size=(512, 512))

test_dataset = CloudDataset(data_root='data/cloud', img_dir='img_dir', mask_dir='ann_dir',
                             mode='test', mosaic_ratio=0.25, transform=train_aug, img_size=(512, 512))

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
base_optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

