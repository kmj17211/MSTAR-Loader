#%%
import os
import struct
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage

os.chdir(os.path.dirname(__file__))
"""
MSTAR LOADER for raw img
mag and phase both, no quantification
shape [2 , w, h].  0 for mag, 1 for phase
"""

def load_parm(header, string):
    start = header.find(string) + len(string)
    end = header.find(b'\n', start)
    parm = int(header[start:end])
    return parm

def read(file):
    with open(file, 'rb') as f:
        header = f.read(512)
        head = load_parm(header, b'PhoenixHeaderLength= ')
        # print(head)

        w = load_parm(header, b'NumberOfColumns= ')
        h = load_parm(header, b'NumberOfRows= ')
        # print(w, h)
        f.seek(0, 0)
        f.read(head)

        data = struct.unpack_from('>'+str(w*h*2)+'f', f.read(), 0)
        # print(data)
        data = np.reshape(np.array(data), (2, h, w))
        # real = data[0] * np.cos(data[1])
        # imag = data[0] * np.sin(data[1])

        # return data[1]
        return data[0]
    
def center_crop(img):
    y, x = img.shape
    cropx = 128
    cropy = 128
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def data_fix(img):
    zero = np.zeros_like(img)
    zero[:,0:64] = img[:,64:128]
    zero[:,64:128] = img[:,0:64]
    return zero

#%%
state = r'mstar2raw.exe '
ind = ' 0'

#IMAGE FOLDER
folder = r'/home/minjun/Code/Data/SAR Data/MSTAR/10-class (17-15)/17_DEG/ZIL131'
#SAVE FOLDER
save_folder = folder.replace('10-class (17-15)', '10-class (17-15)_save')
# if os.path.exists(save_folder):
#     raise FileExistsError("YOU KNOW WHY, IDIOT")
os.makedirs(save_folder, exist_ok=True)
files = os.listdir(folder)
i = 0
to_pil = ToPILImage()
# data_zip = []
for filename in files:
    # POSTFIX
    if '.025' in filename:
        i += 1
        file = os.path.join(folder, filename)
        data = read(file)
        img = center_crop(data)
        # img = abs(data)
        # img = img / img.max()
        # img = np.log10(1000 * img + 1) / np.log10(1000 + 1)

        img = to_pil(np.float32(img))
        img.save(os.path.join(save_folder, filename.replace('.025', '.tif')))
        # np.save(os.path.join(save_folder, filename.replace('.', '_')), data)
        # data_zip.append(data)
        # print(filename)

# data_zip = abs(np.array(data_zip))
# data_flat = data_zip.flatten()
# plt.hist(data_flat, bins = 100, log = True)
# plt.show()
# plt.imshow(data_zip[1,...])
# plt.show()
# # plt.savefig('a.png')
# print(i)

