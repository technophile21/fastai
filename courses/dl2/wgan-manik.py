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
from fastai.dataset import *
import gzip
torch.cuda.set_device(0)

PATH = Path('../../../data/lsun')
PATH_TRAIN = PATH/'bedroom'
PATH_CSV = PATH/'train_sample.csv'
PATH_TMP = PATH/'tmp'
os.makedirs(PATH_TMP, exist_ok=True)

# +
files = PATH_TRAIN.glob('**/*.jpg')

cnt = 0
with PATH_CSV.open('w') as fo:
    for f in files:
        cnt += 1
        fo.write(f'{f.relative_to(PATH_TRAIN)},0\n')
        #For sample of data
        #if random.random()<0.1: fo.write(f'{f.relative_to(PATH_TRAIN)},0\n')
print(cnt)


# -

class ConvBlock(nn.Module):
    def __init__(self, ni, no, ks, stride, pad, bn=True):
        super().__init__()
        self.bn = bn
        self.conv = nn.Conv2d(ni, no, kernel_size=ks, stride=stride, padding=pad, bias=False)
        self.bn2d = nn.BatchNorm2d(no) if bn else None
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn2d(x) if self.bn else x
        return self.relu(x)


class DCGAN_D(nn.Module):
    def __init__(self, ni, isize, ndf, extra_layers=0):
        super().__init__()
        assert isize%16 == 0
        
        self.initial = ConvBlock(ni, ndf, ks=4, stride=2, pad=1, bn=False)
        csize,cndf = isize/2,ndf
        self.extra = nn.Sequential(*[ConvBlock(cndf, cndf, ks=3, stride=1, pad=1) for i in range(extra_layers)])
        
        pyr_layers=[]
        while csize > 4:
            pyr_layers.append(ConvBlock(cndf, cndf*2, ks=4, stride=2, pad=1))
            cndf *= 2
            csize /= 2
            
        self.pyramid = nn.Sequential(*pyr_layers)
        self.final = nn.Conv2d(cndf, 1, kernel_size=4, stride=1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.initial(x)
        x = self.extra(x)
        x = self.pyramid(x)
        return self.final(x).mean(0).view(1)
        #return self.sigmoid(x)


class DeConvBlock(nn.Module):
    def __init__(self, ni, no, ks, stride, pad=0, bn=True):
        super().__init__()
        
        self.bn = bn
        self.deconv = nn.ConvTranspose2d(ni, no, kernel_size=ks, stride=stride, padding=pad, bias=False)
        self.bn2d = nn.BatchNorm2d(no) if bn else None
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn2d(x) if self.bn else x
        return self.relu(x)


class DCGAN_G(nn.Module):
    def __init__(self, nz, nc, ngf, isize, extra_layers=0):
        super().__init__()
        
        assert isize%16 == 0
        
        cngf, tisize = ngf//2, 4
        while tisize!=isize: cngf*=2; tisize*=2
            
        layers = [DeConvBlock(nz, cngf, ks=4, stride=1)]
        
        #pyr_layers = []
        csize, cndf = 4, cngf
        while csize < isize//2:
            layers.append(DeConvBlock(cngf, cngf//2, ks=4, stride=2, pad=1))
            cngf = cngf // 2
            csize *= 2
            
#        self.pyramid = nn.Sequential(*pyr_layers)
        
        layers += [DeConvBlock(cngf, cngf, ks=3, stride=1, pad=1) for t in range(extra_layers)]
        layers.append(nn.ConvTranspose2d(cngf, nc, kernel_size=4, stride=2, padding=1, bias=False))
        self.features = nn.Sequential(*layers)
        #self.tanh = nn.Tanh()
        
    def forward(self, x):
        return F.tanh(self.features(x))



bs, sz, nz = 64,64,100

tfms = tfms_from_stats(inception_stats, sz)
md = ImageClassifierData.from_csv(PATH, 'bedroom', PATH_CSV, bs=128, tfms=tfms, skip_header=False, continuous=True)
len(md.trn_dl)

md = md.resize(128)

x,_ = next(iter(md.val_dl))

plt.imshow(md.trn_ds.denorm(x)[0])

netG = DCGAN_G(nz, 3, 64, sz, 1).cuda()
netD = DCGAN_D(3, sz, 64, 1).cuda()

netG

netD


def create_noise(b): return V(torch.zeros(b, nz, 1, 1).normal_(0, 1))


# +
preds = netG(create_noise(4))
preds_ims = md.trn_ds.denorm(preds)

fig, axes = plt.subplots(2, 2, figsize=(6,6))
for i,ax in enumerate(axes.flat):
    ax.imshow(preds_ims[i])


# -

def gallery(x, nc=3):
    n,h,w,c = x.shape
    nr = n//nc
    assert n == nr*nc
    return (x.reshape(nr, nc, h, w, c)
              .swapaxes(1,2)
              .reshape(h*nr, w*nc, c))


optimizerD = optim.RMSprop(netD.parameters(), lr = 1e-4)
optimizerG = optim.RMSprop(netG.parameters(), lr = 1e-4)


def train(niter, first=True):
    gen_iter = 0
    for epoch in range(niter):
        netG.train()
        netD.train()
        data_iter = iter(md.trn_dl)
        i,n = 0,len(md.trn_dl)
        print(n)
        with tqdm(total=n) as pbar:
            while i < n:
                set_trainable(netG, False)
                set_trainable(netD, True)
                d_iter = 100 if (first and (gen_iter < 25) or (gen_iter%500 == 0)) else 5
                j = 0
                while (j < d_iter) and (i < n):
                    j += 1
                    i += 1
                    for p in netD.parameters(): p.data.clamp_(-0.01, 0.01)
                    real = V(next(data_iter)[0])
                    real_loss = netD(real)
                    fake = netG(create_noise(real.size(0)))
                    fake_loss = netD(V(fake.data))
                    netD.zero_grad()
                    lossD = real_loss - fake_loss
                    lossD.backward()
                    optimizerD.step()
                    pbar.update()

                set_trainable(netD, False)
                set_trainable(netG, True)
                netG.zero_grad()
                lossG = netD(netG(create_noise(bs))).mean(0).view(1)
                lossG.backward()
                optimizerG.step()
                gen_iter += 1
            
        print(f'Loss_D {to_np(lossD)}; Loss_G {to_np(lossG)}; '
            f'D_real {to_np(real_loss)}; Loss_D_fake {to_np(fake_loss)}')



torch.backends.cudnn.benchmark=True

train(1, False)

fixed_noise = create_noise(bs)

set_trainable(netD, True)
set_trainable(netG, True)
optimizerD = optim.RMSprop(netD.parameters(), lr = 1e-5)
optimizerG = optim.RMSprop(netG.parameters(), lr = 1e-5)

train(1, False)

# +
netD.eval(); netG.eval();
fake = netG(fixed_noise).data.cpu()
faked = np.clip(md.trn_ds.denorm(fake),0,1)

plt.figure(figsize=(9,9))
plt.imshow(gallery(faked, 8));
# -


