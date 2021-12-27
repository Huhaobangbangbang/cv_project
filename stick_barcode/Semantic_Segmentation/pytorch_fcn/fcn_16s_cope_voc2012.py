"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 参考博客：https://www.cnblogs.com/zou0929/p/14709976.html
 @date   2021/10/24 8:59 PM
"""
import os
from datetime import datetime
import cv2
import numpy as np
import torch
import torchvision.transforms as tfs
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn, log_softmax
from torch.autograd import Variable
from torch.utils import model_zoo
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
voc_root = '/Users/huhao/Documents/VOC2012'

def read_image(root=voc_root, train=True):
    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')

    with open(txt_fname, 'r') as f:
        images = f.read().split()

    # images = np.loadtxt(txt_fname, delimiter=' ', dtype=np.int)
    # print(images)

    data = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i + '.png') for i in images]
    # images=data[0]
    # print(images)
    # img = cv2.imread(images)
    # cv2.imshow("Image", img)
    # cv2.waitKey()
    return data, label


# 图片随机剪裁重写
class RdCrop(tfs.RandomCrop):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        # super(RdCrop, self).__init__(self)
        tfs.RandomCrop.__init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant")

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = F._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), [i, j, h, w]


# read_image()
def rand_crop(data, label, height, width):
    '''
    data is PIL.Image object
    ImagesSet is PIL.Image object
    '''
    # print(data)
    data, rect = RdCrop((height, width))(data)
    # data, rect = RdCrop((100, 100))(data)
    # ImagesSet = tfs.FixedCrop(*rect)(ImagesSet)
    label = F.crop(label, rect[0], rect[1], rect[2], rect[3])
    # print(type(data))
    # print(data)
    # print(rect[0])
    # print(ImagesSet)
    return data, label


# data, ImagesSet = read_image(voc_root)

# print(data[0])
# img = cv2.imread(data[0])
# data1=Image.open(data[0])
# cropped_image = tfs.RandomCrop(100,100)(image)
# label1=Image.open(ImagesSet[0])
# data,ImagesSet=rand_crop(data,ImagesSet,100,100)
# cv2.imshow("output", cropped_image)
# cv2.waitKey(0)


classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']

# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]

# print(len(classes), len(colormap))
cm2lbl = np.zeros(256 ** 3)  # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i  # 建立索引


def image2label(im):
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')  # 根据索引得到 ImagesSet 矩阵


# data, ImagesSet = read_image(voc_root)
#
# label_im = Image.open(ImagesSet[1]).convert('RGB')
# ImagesSet = image2label(label_im)
# plt.imshow(ImagesSet)
# plt.show()
# print(ImagesSet)

def img_transforms(im, label, crop_size):
    im, label = rand_crop(im, label, *crop_size)

    im_tfs = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    im = im_tfs(im)
    label = image2label(label)
    label = torch.from_numpy(label)
    return im, label


class VOCSegDataset(Dataset):
    '''
    voc dataset
    '''

    def __init__(self, train, crop_size, transforms):
        self.crop_size = crop_size
        self.transforms = transforms
        data_list, label_list = read_image(train=train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
        print('Read ' + str(len(self.data_list)) + ' images')

    def _filter(self, images):  # 过滤掉图片大小小于 crop 大小的图片
        return [im for im in images if (Image.open(im).size[1] >= self.crop_size[0] and
                                        Image.open(im).size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        img = self.data_list[idx]
        img1 = img
        # print(img)
        label = self.label_list[idx]
        label1 = label
        img = Image.open(img)
        # #img = Image.open("G:\\dataset\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\SegmentationClass/2007_000032.png")
        # print(img)
        #
        # plt.imshow(img)
        # plt.show()
        label = Image.open(label).convert('RGB')

        img, label = self.transforms(img, label, self.crop_size)

        return img, label

    def __len__(self):
        return len(self.data_list)


input_shape = (320, 480)
voc_train = VOCSegDataset(True, input_shape, img_transforms)
voc_test = VOCSegDataset(False, input_shape, img_transforms)

train_data = DataLoader(voc_train, 64, shuffle=True, num_workers=4)
valid_data = DataLoader(voc_test, 64, num_workers=4)


def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


# x = Image.open('G:/dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2007_005210.jpg')
# x = np.array(x)
# plt.imshow(x)
# print(x.shape)

# pretrained_net = model_zoo.resnet34(pretrained=True)
pretrained_net = torchvision.models.resnet34(pretrained=True)
num_classes = len(classes)


class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn, self).__init__()

        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])  # 第一段
        self.stage2 = list(pretrained_net.children())[-4]  # 第二段
        self.stage3 = list(pretrained_net.children())[-3]  # 第三段

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)  # 使用双线性 kernel

        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

    def forward(self, x):
        x = self.stage1(x)
        s1 = x  # 1/8

        x = self.stage2(x)
        s2 = x  # 1/16

        x = self.stage3(x)
        s3 = x  # 1/32

        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3

        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2

        s = self.upsample_8x(s2)
        return s


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


class ScheduledOptim(object):
    '''A wrapper class for learning rate scheduling'''

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']
        self.current_steps = 0

    def step(self):
        "Step by the inner optimizer"
        self.current_steps += 1
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def learning_rate(self):
        return self.lr


# from mxtorch.trainer import ScheduledOptim

def train1():
    net = fcn(num_classes)
    net.cpu()

    criterion = nn.NLLLoss2d()
    basic_optim = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)
    optimizer = ScheduledOptim(basic_optim)
    for e in range(80):
        if e > 0 and e % 50 == 0:
            optimizer.set_learning_rate(optimizer.learning_rate * 0.1)
        train_loss = 0
        train_acc = 0
        train_acc_cls = 0
        train_mean_iu = 0
        train_fwavacc = 0

        prev_time = datetime.now()
        net = net.train()
        for data in train_data:
            im = Variable(data[0].cuda())
            label = Variable(data[1].cuda())
            # im = data[0]
            # ImagesSet = data[1]
            # forward
            out = net(im)
            out = log_softmax(out, dim=1)  # (b, n, h, w)
            loss = criterion(out, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            label_pred = out.max(dim=1)[1].data.cpu().numpy()
            label_true = label.data.cpu().numpy()
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
                train_acc += acc
                train_acc_cls += acc_cls
                train_mean_iu += mean_iu
                train_fwavacc += fwavacc

        net = net.eval()
        eval_loss = 0
        eval_acc = 0
        eval_acc_cls = 0
        eval_mean_iu = 0
        eval_fwavacc = 0
        for data in valid_data:
            im = Variable(data[0].cuda(), volatile=True)
            label = Variable(data[1].cuda(), volatile=True)
            # forward
            # im = data[0]
            # ImagesSet = data[1]
            out = net(im)
            out = log_softmax(out, dim=1)
            loss = criterion(out, label)
            eval_loss += loss.item()

            label_pred = out.max(dim=1)[1].data.cpu().numpy()
            label_true = label.data.cpu().numpy()
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
                eval_acc += acc
                eval_acc_cls += acc_cls
                eval_mean_iu += mean_iu
                eval_fwavacc += fwavacc

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f}, \
    Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} '.format(
            e, train_loss / len(train_data), train_acc / len(voc_train), train_mean_iu / len(voc_train),
               eval_loss / len(valid_data), eval_acc / len(voc_test), eval_mean_iu / len(voc_test)))
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print(epoch_str + time_str + ' lr: {}'.format(optimizer.learning_rate))


if __name__ == '__main__':
    train1()