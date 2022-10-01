import numpy as np
import skimage
import matplotlib.pyplot as plt
orig_img = skimage.data.astronaut()
skimage.io.imsave("ast.jpg", orig_img)
# 要透過pillow格式進行讀取
from PIL import Image
# 以pillow函數讀取讀檔
orig_img = Image.open("ast.jpg")
from pathlib import Path
import numpy as np
import torchvision.transforms as T
# 定義繪圖函數
def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # make 2d grid if there is only 1d
        imgs = [imgs]
    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows = num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if with_orig:
        axs[0,0].set(title="origin_pic")
        axs[0,0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])
    plt.tight_layout()
    plt.show()
# resize(圖片放大縮小)
resized_img = [T.Resize(size=size)(orig_img) for size in (30, 50, 100, orig_img.size)]
# plot(resized_img)

# 從中心做裁減
centers_crops = [T.CenterCrop(size=size)(orig_img) for size in (30, 50, 100, orig_img.size)]
# plot(centers_crops)

# 以不同點為中心，裁減五張圖片
(top_left, top_right, bottom_left, bottom_right, center) = T.FiveCrop(size=(100, 100))(orig_img)
# plot([top_left, top_right, bottom_left, bottom_right, center])

# 轉成灰階
gray_img = T.Grayscale()(orig_img)
# plot([gray_img], cmap='gray')

# 周圍補0(看遍框來觀察變化)
paddings_img = [T.Pad(padding=padding)(orig_img) for padding in (3, 10, 30,50)]
plot(paddings_img)