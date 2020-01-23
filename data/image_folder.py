"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path
import pydicom
import skimage
import numpy as np


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_dicom_file(filename):
    return any(filename.endswith(extension) for extension in "dcm")


'''
qh: ajouter ici if_dicom_file
def is_dicom_file(filename):
    return any(filename.endswith(extension) for extension in "dcm")
'''


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname) or is_dicom_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]
'''
 14 Janvier 2020 QH: on peut changer ici pour ajouter le code à lire image DICOM
 for ex; 
     def dicom_loader(path)::
         return pydicom.open(path).convert('gray')
         
attention: PIL.Image.open return an Image Objet, but not Np array
penser à injecter pixel data de pydicom à PIL image objet, puis donner à CNN

OK: images[] peut returner les fichers en dcm.
         

'''
def dicomImageOpen(path):
    ds=pydicom.dcmread(path)
    return Image.fromarray(ds.pixel_array).convert('I')
    
    '''array_buffer=ds.pixel_array.tobytes()
    imgA=Image.new("I",ds.Pixel_array.T.shape)
    return imgA.frombytes(array_buffer,'raw',"I;16")
       #32-bit signed integer pixels conversion
    '''
       
    
    

def default_loader(path):
    if is_dicom_file(path):
        return dicomImageOpen(path) # ajouter ici le code pour lire dicom file
    else:
        return Image.open(path).convert('RGB')
#

class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
