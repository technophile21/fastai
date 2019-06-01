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

torch.cuda.set_device(0)
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

x,y = next(iter(md.val_dl))


show_img(md.val_ds.denorm(to_np(x))[0]);

learn = ConvLearner.pretrained(f_model, md, metrics=[accuracy])
learn.opt_fn = optim.Adam

lrf=learn.lr_find(1e-5,100)

learn.sched.plot()

learn.sched.plot(n_skip=5, n_skip_end=1)

lr = 2e-2

learn.fit(lr, 1, cycle_len=1)


