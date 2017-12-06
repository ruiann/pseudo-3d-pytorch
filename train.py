from __future__ import print_function
import pdb

from p3d_model import *
import torch
import random
import os
import numpy as np
from PIL import Image

train_list = 'ucfTrainTestlist/trainlist01.txt'
test_list = 'ucfTrainTestlist/testlist01.txt'
class_index = 'ucfTrainTestlist/classInd.txt'
data_dir = 'data'
batch_size = 10
step = 1000000

# below parameters are fixed
frame_size = 16
height = 160
width = 160


def read_train_list():
    files = []
    labels = []
    for line in open(train_list):
        file, label = line.replace('.avi', '').replace('\r\n', '').split(' ')
        files.append(file)
        labels.append(int(label) - 1)
    return files, labels


def read_test_list():
    classes = []
    files = []
    labels = []
    for line in open(class_index):
        classes.append(line.replace('\r\n', '').split(' ')[1])
    for line in open(test_list):
        file = line.replace('.avi', '').replace('\r\n', '')
        clazz = file.split('/')[0]
        files.append(file)
        labels.append(classes.index(clazz))
    return files, labels


def read_sample(video):
    video_path = '{}/{}'.format(data_dir, video)
    frames = range(len(os.listdir(video_path)))
    random.shuffle(frames)
    frames = sorted(frames[0: frame_size])
    tensor = []
    for i in frames:
        file_name = '{0}/{1:0>3}.jpg'.format(video_path, i + 1)
        pic = np.asarray(Image.open(file_name))
        tensor.append(pic)
    tensor = np.array(tensor)
    # from lhwc to clhw
    tensor = np.rollaxis(tensor, 3, 0)
    return tensor


def read_batch(files, labels):
    samples = []
    sample_labels = []
    limit = len(files)
    for i in range(batch_size):
        index = random.randint(0, limit - 1)
        samples.append(read_sample(files[index]))
        sample_labels.append(labels[index])
    return torch.from_numpy(np.array(samples)), torch.from_numpy(np.array(sample_labels))


files, labels = read_train_list()
test_files, test_labels = read_test_list()

model = P3D199(num_classes=101)
model = model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train():
    data, data_label = read_batch(files, labels)
    out = model(torch.autograd.Variable(data.float()).cuda())
    loss = criterion(out, torch.autograd.Variable(data_label, requires_grad=False).cuda())
    print(i, loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test():
    test_data, test_label = read_batch(test_files, test_labels)
    out = model(torch.autograd.Variable(test_data.float(), volatile=True).cuda())
    _, predict_label = torch.max(out, dim=1)
    eq = torch.eq(predict_label, torch.autograd.Variable(test_label).cuda())
    print(eq)


i = 1
while i < step:
    if i % 100 == 0:
        test()
    else:
        train()
    i = i + 1
