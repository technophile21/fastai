# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

from fastai.conv_learner import *

PATH = Path("../../../data/cifar/")
os.makedirs(PATH, exist_ok=True)
#torch.cuda.set_device(1)

# +
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))

num_workers = num_cpus()//2
bs = 256
sz = 32
# -

tfms = tfms_from_stats(stats, sz=sz, aug_tfms=[RandomFlip()], pad=sz//8)
data = ImageClassifierData.from_paths(PATH, bs=bs, tfms=tfms, val_name='test')


def conv_layer(ni, nf, stride=1, ks=3):
    return nn.Sequential(
            nn.Conv2d(ni, nf, ks, stride, padding=ks//2, bias=False),
            nn.BatchNorm2d(nf, momentum=0.01),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )


class ResLayer(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.conv1 = conv_layer(ni, ni//2, ks=1)
        self.conv2 = conv_layer(ni//2, ni, ks=3)
    
    def forward(self, x):
        return x.add(self.conv2(self.conv1(x)))


def make_group_layer(ni, num_blocks, stride=1):
        return [conv_layer(ni, ni*2, stride=stride)
               ] + [ResLayer(ni*2) for i in range(num_blocks)]


class DarkNet(nn.Module):    
    def __init__(self, num_blocks, num_classes, nf=32):
        super().__init__()
        layers = [conv_layer(3, nf, ks=3, stride=1)]
        for i,nb in enumerate(num_blocks):
            layers += make_group_layer(ni=nf, num_blocks=nb, stride=2-(i==1))
            nf *= 2
        layers += [nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(nf, num_classes)]
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


m = DarkNet([1,2,4,6,3], num_classes=10, nf=32)
m = nn.DataParallel(m, device_ids=None)

learn = ConvLearner.from_model_data(m, data)
learn.critic = nn.CrossEntropyLoss()
learn.metrics = [accuracy]
wd = 1e-4
lr=1.3

# %time learn.fit(lr, 1, wds=wd, cycle_len=2, use_clr_beta=(20, 20, 0.95, 0.85))


