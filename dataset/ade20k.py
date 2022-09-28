from torch.utils import data
import os.path as osp
import numpy as np
import random
import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

from PIL import Image
import os
from torchvision import transforms


class ADETrainSet(data.Dataset):
    def __init__(self, root, max_iters=None, crop_size=(512, 1024), scale=True, mirror=True, ignore_label=-1):
        self.root = root
        self.crop_h, self.crop_w = crop_size
        self.is_scale = scale
        self.is_mirror = mirror
        self.ignore_label = ignore_label

        img_folder = os.path.join(root, 'images/training')
        mask_folder = os.path.join(root, 'annotations/training')
        

        self.files = []
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                maskname = basename + '.png'
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(maskpath):
                    self.files.append({
                    "img": imgpath,
                    "label": maskpath,
                    "name": filename
                    })
                else:
                    print('cannot find the mask:', maskpath)
        
        if max_iters:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))
            self.files = self.files[:max_iters]

        print('{} training images are loaded!'.format(len(self.files)))

        self.num_class = 150

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 15) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        
        size = image.shape

        name = datafiles["name"]
        if self.is_scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image = image - np.array([104.00698793, 116.66876762, 122.67891434])
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label
        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
        label = label - 1
        return image.copy(), label.copy(), name


class ADEDataValSet(data.Dataset):
    def __init__(self, root, ignore_label=-1):
        self.root = root
        self.ignore_label = ignore_label
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

        self.files = [] 
        
        img_folder = os.path.join(root, 'images/validation')
        mask_folder = os.path.join(root, 'annotations/validation')
        
        self.files = []
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                maskname = basename + '.png'
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(maskpath):
                    self.files.append({
                    "img": imgpath,
                    "label": maskpath,
                    "name": filename
                    })
                else:
                    print('cannot find the mask:', maskpath)
                    
            
        print('{} validation images are loaded!'.format(len(self.files)))
        

        self.num_class = 150


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)

        size = image.shape

        name = osp.splitext(osp.basename(datafiles["img"]))[0]
        image = np.asarray(image, np.float32)
        image = image - np.array([104.00698793, 116.66876762, 122.67891434])
        image = image.transpose((2, 0, 1))
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        label = label - 1

        return  image.copy(), label.copy(), (datafiles["img"], name)