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

# +
from fastai.conv_learner import *
from fastai.dataset import *

from pathlib import Path
import json
from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects


# -

PATH = Path('../../../data/pascal')
list(PATH.iterdir())

trn_data = json.load((PATH/'pascal_train2007.json').open())

trn_data.keys()

IMAGES, ANNO, CAT = ['images', 'annotations', 'categories']
trn_data[IMAGES][:5]

BBOX, IMAGE_ID, CAT_ID, FILE_NAME = ['bbox', 'image_id', 'category_id', 'file_name'] 
trn_data[ANNO][:2]

trn_data[CAT][:5]

ID, NAME = ['id', 'name']
trn_cat = {o[ID]:o[NAME] for o in trn_data[CAT]}
trn_cat[1]

trn_fnm = {o[ID]:o[FILE_NAME] for o in trn_data[IMAGES]}
trn_fnm[12]

trn_ids = [o[ID] for o in trn_data[IMAGES]]
trn_ids[:5]


# +
def hw_bb(bb): return np.array([bb[1], bb[0], bb[3]+bb[1]-1, bb[2]+bb[0]-1])

trn_anno = collections.defaultdict(lambda:[])
for o in trn_data[ANNO]:
    if not o['ignore']:
        bb = o[BBOX]
        bb = hw_bb(bb)
        trn_anno[o[IMAGE_ID]].append((bb, o[CAT_ID]))
len(trn_anno)
# -

img_path = PATH/'VOCdevkit/VOC2007'
list(img_path.iterdir())

img_path = img_path/'JPEGImages'

trn_anno[12]

trn_cat[7]


def bb_hw(a): return np.array([a[1],a[0],a[3]-a[1]+1,a[2]-a[0]+1])


def show_img(img, figsize=None, axes=None):
    if not axes:
        fig, axes = plt.subplots(figsize=figsize)
    axes.imshow(img)
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    return axes


def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])


def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)


def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)


im = open_image(img_path/trn_data[IMAGES][0][FILE_NAME])

ax = show_img(im)
img_id = trn_data[IMAGES][0][ID]
bbox, cat_id = trn_anno[img_id][0]
bbox = bb_hw(bbox)
#print(trn_anno[img_id][1])
# cat = trn_cat[cat_id]
draw_text(ax, (bbox[0], bbox[1]), cat)
draw_rect(ax, bbox)


def draw_im(im, ann):
    ax = show_img(im, figsize=(16,8))
    for b,c in ann:
        b = bb_hw(b)
        draw_rect(ax, b)
        draw_text(ax, b[:2], trn_cat[c], sz=16)


def draw_idx(i):
    im_a = trn_anno[i]
    im = open_image(img_path/trn_fnm[i])
    print(im.shape)
    draw_im(im, im_a)


draw_idx(17)


def get_lrg(b):
    if not b: raise Exception()
    b = sorted(b, key=lambda x: np.product(x[0][-2:]-x[0][:2]), reverse=True)
    return b[0]


#Now we have a dictionary from image id to a single bounding box - the largest for that image.
trn_lrg_anno = {a: get_lrg(b) for a,b in trn_anno.items()}

img_name = trn_fnm[23]
ax = show_img(open_image(img_path/img_name), figsize=(15, 8))
(b, c) = trn_lrg_anno[23]
b = bb_hw(b)
draw_rect(ax, b)
draw_text(ax, b[:2], trn_cat[c], 16)

(PATH/'tmp').mkdir(exist_ok = True)
CSV = PATH/'tmp/lrg.csv'

df = pd.DataFrame({'fn': [trn_fnm[o] for o in trn_ids],                  
                   'cat': [trn_cat[trn_lrg_anno[o][1]] for o in trn_ids]}, columns=['fn', 'cat'])
df.to_csv(CSV, index=False)

f_model = resnet34
sz = 224
bs = 64
JPEGS = 'VOCdevkit/VOC2007/JPEGImages'

tfm_model = tfms_from_model(f_model, sz, aug_tfms=transforms_side_on, crop_type=CropType.NO)
md = ImageClassifierData.from_csv(PATH, JPEGS, CSV, tfms=tfm_model, bs=bs)

md

#get the iterator from data loader and extract the next batch
x,y = next(iter(md.val_dl))


#x is normalized (done during application of transforms)
# we use md.val_ds to denormalize as we want to use the same stats for normalization as used for normalization
show_img(md.val_ds.denorm(to_np(x))[0]);

learn = ConvLearner.pretrained(f_model, md, metrics=[accuracy])
learn.opt_fn = optim.Adam

learn.lr_find(1e-5, 100)

learn.sched.plot()

#Generally sched.plot skips 10 values at start and 5 at end. Below code will decrease skips 
learn.sched.plot(n_skip=5, n_skip_end=1)

lr = 2e-2

learn.fit(lr, 1, cycle_len=1)

lrs = np.array([lr/1000,lr/100,lr])

learn.freeze_to(-2)

lrf=learn.lr_find(lrs/1000)
learn.sched.plot(1)

learn.fit(lrs/5, 1, cycle_len=1)

learn.unfreeze()

learn.fit(lrs/5, 1, cycle_len=2)

learn.save('clas_one')

learn.load('clas_one')

x,y = next(iter(md.val_dl))
probs = F.softmax(predict_batch(learn.model, x), -1)
x,preds = to_np(x),to_np(probs)
preds = np.argmax(preds, -1)

fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for i,ax in enumerate(axes.flat):
    ima=md.val_ds.denorm(x)[i]
    b = md.classes[preds[i]]
    ax = show_img(ima, axes=ax)
    draw_text(ax, (0,0), b)
plt.tight_layout()

BB = PATH/'tmp/bb.csv'

bb = np.array([trn_lrg_anno[o][0] for o in trn_ids])
bbs = [' '.join(str(p) for p in o) for o in bb]
df = pd.DataFrame({'fn': [trn_fnm[o] for o in trn_ids],
                  'bbox': bbs}, columns=['fn', 'bbox'])
df.to_csv(BB, index=False)

f_model = resnet34
sz = 224
bs = 64

augs = [RandomFlip(),
       RandomRotate(30),
       RandomLighting(0.1, 0.1)]

tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, aug_tfms=augs)
md = ImageClassifierData.from_csv(PATH, JPEGS, BB, bs=4, tfms=tfms, continuous=True)

idx=3
fig,axes = plt.subplots(3,3, figsize=(9,9))
for i,ax in enumerate(axes.flat):
    x,y=next(iter(md.aug_dl))
    ima=md.val_ds.denorm(to_np(x))[idx]
    b = bb_hw(to_np(y[idx]))
    print(b)
    show_img(ima, axes=ax)
    draw_rect(ax, b)

tfm_y = TfmType.COORD
augs = [RandomFlip(tfm_y=tfm_y),
       RandomRotate(30, tfm_y=tfm_y),
       RandomLighting(0.1, 0.1, tfm_y=tfm_y)]

tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, tfm_y=TfmType.COORD, aug_tfms=augs)
md = ImageClassifierData.from_csv(PATH, JPEGS, BB, bs=4, continuous=True, tfms=tfms)

idx=3
fig,axes = plt.subplots(3,3, figsize=(9,9))
for i,ax in enumerate(axes.flat):
    x,y=next(iter(md.aug_dl))
    ima=md.val_ds.denorm(to_np(x))[idx]
    b = bb_hw(to_np(y[idx]))
    print(b)
    show_img(ima, axes=ax)
    draw_rect(ax, b)

# +
tfm_y = TfmType.COORD
augs = [RandomFlip(tfm_y=tfm_y),
        RandomRotate(3, p=0.5, tfm_y=tfm_y),
        RandomLighting(0.05,0.05, tfm_y=tfm_y)]

tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, tfm_y=tfm_y, aug_tfms=augs)
md = ImageClassifierData.from_csv(PATH, JPEGS, BB, tfms=tfms, bs=bs, continuous=True)
# -

512*7*7

head_reg = nn.Sequential(Flatten(), nn.Linear(25088, 4))
learn = ConvLearner.pretrained(f_model, md, custom_head=head_reg)
learn.optim_fn = optim.Adam
learn.crit = nn.L1Loss()

learn.summary()

learn.lr_find(1e-5,100)
learn.sched.plot(5)

lr = 2e-3

learn.fit(lr, 2, cycle_len=1, cycle_mult=2)

lrs = np.array([lr/100,lr/10,lr])

learn.freeze_to(-2)

lrf=learn.lr_find(lrs/1000)
learn.sched.plot(1)

learn.fit(lrs, 2, cycle_len=1, cycle_mult=2)

learn.freeze_to(-3)

learn.fit(lrs, 1, cycle_len=2)

learn.save('reg4')

learn.load('reg4')

x,y = next(iter(md.val_dl))
learn.model.eval()
preds = to_np(learn.model(VV(x)))

fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for i,ax in enumerate(axes.flat):
    ima=md.val_ds.denorm(to_np(x))[i]
    b = bb_hw(preds[i])
    ax = show_img(ima, axes=ax)
    draw_rect(ax, b)
plt.tight_layout()

#  ## Single Object Detection


