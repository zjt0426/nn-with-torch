import torch
import torch.nn as nn
import os
import numpy as np
import random
import torchvision.transforms as tfs
from torchvision import models
from torch.utils import data
from PIL import Image
import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd.variable import Variable
import cv2



voc_root = "/media/server/segmentation/fcn.pytorch-master/fcn.pytorch-master/data/VOCdevkit/VOC2012"



def read_images(root=voc_root, train=True):
    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i + '.png') for i in images]
    return data, label


def rand_crop(data, label, height, width):
    '''
    data is PIL.Image object
    label is PIL.Image object
    '''
    x = random.uniform(0, data.size[0] - width)
    x = int(x)
    y = random.uniform(0, data.size[1] - height)
    y = int(y)

    box = (x, y, x + width, y + height)
    data = data.crop(box)
    label = label.crop(box)
    return data, label


classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']


colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


cm2lbl = np.zeros(256 ** 3)  
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i  


def image2label(im):
    data = np.array(im, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')  


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


class VOCSegDataset(data.Dataset):
    '''
    voc dataset
    '''

    def __init__(self, train, crop_size, transforms):
        self.crop_size = crop_size
        self.transforms = transforms
        data_list, label_list = read_images(train=train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
        print('Read ' + str(len(self.data_list)) + ' images')

    def _filter(self, images):  
        return [im for im in images if (Image.open(im).size[1] >= self.crop_size[0] and
                                        Image.open(im).size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.transforms(img, label, self.crop_size)
        return img, label

    def __len__(self):
        return len(self.data_list)



input_shape = (320, 480)
voc_train = VOCSegDataset(True, input_shape, img_transforms)
voc_test = VOCSegDataset(False, input_shape, img_transforms)

train_data = data.DataLoader(voc_train, 4, shuffle=True, num_workers=4)
valid_data = data.DataLoader(voc_test, 4, num_workers=4)

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

'''
pretrained_net = models.vgg16(pretrained=True)
num_classes = len(classes)
'''
pretrained_net = models.resnet34(pretrained=True)
num_classes = len(classes)
'''
class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn, self).__init__()

        self.features = pretrained_net.features

        self.conv1 = nn.Conv2d(512, 4096, 1)
        self.conv2 = nn.Conv2d(4096, 21, 1)

        self.relu = nn.ReLU(inplace=True)

        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upsample32x = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)

    def forward(self, x):
        s = self.features(x)
        s = self.conv1(s)
        s = self.relu(s)
        s = self.conv2(s)
        s = self.relu(s)
        s = self.upsample32x(s)
        return s
        '''

class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn, self).__init__()

        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4]) 
        self.stage2 = list(pretrained_net.children())[-4]
        self.stage3 = list(pretrained_net.children())[-3] 
        
        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)
        
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16) 
        
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4) 
        
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)   
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4) 

        
    def forward(self, x):
        x = self.stage1(x)
        s1 = x # 1/8
        
        x = self.stage2(x)
        s2 = x # 1/16
        
        x = self.stage3(x)
        s3 = x # 1/32
        
        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3
        
        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2

        s = self.upsample_8x(s2)
        return s


net = fcn(num_classes)
net.cuda()

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

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)
# basic_optim = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)
# optimizer = ScheduledOptim(basic_optim)

def acc_simu(label_true, label_pred):

    sum = len(voc_train.data_list) * label_true.shape[1] * label_true.shape[2]

    cnt = 0.
    check = label_true == label_pred

    for i in range(0, label_pred.shape[0]):
        for j in range(0, label_pred.shape[1]):
            for k in range(0, label_pred.shape[2]):
                if check[i][j][k]:
                    cnt = cnt + 1
    return 100. * cnt / sum
    
def predict(im, label): 
    im = Variable(im.unsqueeze(0)).cuda()
    out = net(im)
    pred = out.max(1)[1].squeeze().cpu().data.numpy()
    pred = cm[pred] # this can find predict matrix's color in cm
    return pred, cm[label.numpy()]   # first is pred image and secode is truth image

for e in range(10):

    # train_loss = 0
    # train_acc = 0
    
    train_loss = 0
    train_acc = 0
    train_acc_cls = 0
    train_mean_iu = 0
    train_fwavacc = 0
    
    prev_time = datetime.datetime.now()
    net = net.train()
    for data in train_data:
        im = data[0].cuda()
        label = data[1].cuda()
        # forward
        out = net(im)
        out = F.log_softmax(out, dim=1)  # (b, n, h, w)
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
        # acc = acc_simu(label_true, label_pred)
        # train_acc += acc
        # print(train_acc, '%')

    net = net.eval()
    # eval_loss = 0
    # eval_acc = 0
    eval_loss = 0
    eval_acc = 0
    eval_acc_cls = 0
    eval_mean_iu = 0
    eval_fwavacc = 0
    for data in valid_data:
        im = data[0].cuda()
        label = data[1].cuda()
        # forward
        with torch.no_grad():
            out = net(im)
            out = F.log_softmax(out, dim=1)
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
            
        # acc = acc_simu(label_true, label_pred)
        # eval_acc += acc

    cur_time = datetime.datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f}, \
Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} '.format(
        e, train_loss / len(train_data), train_acc / len(voc_train), train_mean_iu / len(voc_train),
        eval_loss / len(valid_data), eval_acc / len(voc_test), eval_mean_iu / len(voc_test)))
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(epoch_str + time_str)
    
    # epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, \
# Valid Loss: {:.5f}, Valid Acc: {:.5f} '.format(
        # e, train_loss / len(train_data), train_acc,
           # eval_loss / len(valid_data), eval_acc))
    # time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    # print(epoch_str + time_str)
    torch.save(net,'/media/server/segmentation/fcn.pytorch-master/fcn.pytorch-master/model.pth')

    cm = np.array(colormap).astype('uint8')
    
    _, figs = plt.subplots(6, 3, figsize=(12, 10))
    for i in range(6):
        test_data, test_label = voc_test[i]
        pred, label = predict(test_data, test_label)
        imageroot = "/media/server/segmentation/fcn.pytorch-master/fcn.pytorch-master/results/myruns/" + str(i)+ ".jpg"
        cv2.imwrite(imageroot, pred)
        
        figs[i, 0].imshow(Image.open(voc_test.data_list[i]))
        figs[i, 0].axes.get_xaxis().set_visible(False)
        figs[i, 0].axes.get_yaxis().set_visible(False)
        figs[i, 1].imshow(label)
        figs[i, 1].axes.get_xaxis().set_visible(False)
        figs[i, 1].axes.get_yaxis().set_visible(False)
        figs[i, 2].imshow(pred)
        figs[i, 2].axes.get_xaxis().set_visible(False)
        figs[i, 2].axes.get_yaxis().set_visible(False)





