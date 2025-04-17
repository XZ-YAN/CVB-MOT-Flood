# coding=UTF-8
import argparse, os, time, shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision
from model import Net

parser = argparse.ArgumentParser("Train on Reid Dataset")
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument('--resume', '-r', action='store_true')
args = parser.parse_args()

# device
device = "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available(): cudnn.benchmark = True

# data loading
root = 'data'
train_dir = os.path.join(root, "bbox_train")
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((128, 64), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(train_dir, transform=transform_train),
    batch_size=64, shuffle=True)

num_classes = len(trainloader.dataset.classes)

# net definition
start_epoch = 0
net = Net(num_classes=num_classes)
if args.resume:
    print('Loading from checkpoint')
    checkpoint = torch.load("./checkpoint/ckpt.t7")
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
net.to(device)

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
best_acc = 0.

# lr decay
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))

# train function for each epoch
def train(epoch):
    print("\nEpoch : %d" % (epoch+1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = 20
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print
        if (idx+1) % interval == 0: #100.*(idx+1)/len(trainloader) == 100.0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(trainloader), end-start, training_loss /
                interval, correct, total, 100.*correct/total))
            training_loss = 0.
            start = time.time()

    # saving checkpoint in each epoch
    acc = 100.*correct/total
    print('Saving parameters to checkpoint/osnet_x0_25_cmot-'+str(epoch+1)+'.pt')
    checkpoint = {'net_dict': net.state_dict(), 'acc': acc, 'epoch': epoch}
    torch.save(checkpoint, './checkpoint/osnet_x0_25_cmot-'+str(epoch+1)+'.pt')
    return train_loss/len(trainloader), 1. - correct/total

def main(total_epoch):
    if not os.path.isdir('checkpoint'): os.mkdir('checkpoint')
    total_train_loss,  total_train_err = [], []
    for epoch in range(start_epoch, start_epoch + total_epoch):
        train_loss, train_err = train(epoch)
        if (epoch+1) % 20 == 0: lr_decay()
        total_train_loss.append(train_loss)
        total_train_err.append(train_err)
  
if __name__ == '__main__':
    #total_epoch = 10
    #main(total_epoch)
    shutil.copy2('./checkpoint/osnet_x0_25_cmot-10.pt', '../MOT_StrongSORT_box/weights/osnet_x0_25_cmot-10.pt')
    
