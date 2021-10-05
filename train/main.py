import torch
import imageio
import glob
import numpy as np
import math
from torch.optim import lr_scheduler
import torch.optim as optim

from train_dataset import *
from train_loss import *
from train_model import *

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feat_thresh = 0.05
dist_thresh = 2
margin = 100
save_name = "./locnet_descriptor"

def load_data(num_data):
    tensor_dataset = []
    poses = [] # r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz

    image_paths = 'C:\\Users\\Haeyeon Kim\\Desktop\\lego_loam_result\\train_image\\'

    ## For KITTI Data
    # gt_path = "/home/haeyeon/Cocel/slam_ws/dataset/poses/00.txt" 
    # f = open(gt_path, "r")
    # while True:
    #     line = f.readline()
    #     if not line: break
    #     data = line.split()
    #     xy_poses = [float(data[3]), float(data[7])]
    #     poses.append(xy_poses)
    # f.close()

    ## For VLP 16 Data
    gt_path = "../gt/1528/key_gt.txt" 
    f = open(gt_path, "r")
    while True:
        line = f.readline()
        if not line: break

        if line[0] == 'x':
            data = line.split()
            x = float(data[-1])
            next_line = f.readline()
            data = next_line.split()
            y = float(data[-1])
            xy_poses = [x, y]
            poses.append(xy_poses)

    f.close()

    
    for i in range(num_data):
        range_i = imageio.imread(image_paths + 'range%d.png'%(i+1))
        # delta_range_i = imageio.imread(image_paths + 'delta_range/%d.png'%(i+1))
        # data = np.stack([range_i, delta_range_i])
        data = np.stack(range_i)
        #data_reshape = np.reshape(data, (1,data.shape[0],data.shape[1],data.shape[2])) # N x C x H x W
        
        Tensor = torch.tensor(data).float()
        tensor_dataset.append(Tensor)
    return tensor_dataset, poses

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)

        val_loss = test_epoch(val_loader, model, loss_fn, cuda, epoch)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, val_loss)
        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval):
    model.train()
    losses = []
    total_loss = 0
    total = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        predicted = (outputs[0] - outputs[1]).pow(2).sum(1) < feat_thresh
        total += batch_size
        correct += sum(predicted == target[0])

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    print("train accuracy: %f (%d/%d)"%(correct/total, correct, total))
    return total_loss

def test_epoch(val_loader, model, loss_fn, cuda, epoch):
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target
            
            predicted = (outputs[0] - outputs[1]).pow(2).sum(1) < feat_thresh
            total += batch_size
            correct += sum(predicted == target[0])


            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

    print("test accuracy: %f (%d/%d)"%(correct/total, correct, total))
    if epoch % 10 == 0:
        traced_script_module = torch.jit.trace(embedding_model, data[0], check_trace=False)
        traced_script_module.save(save_name + str(epoch)+".pt")


    return val_loss

if __name__ == '__main__':
    dataset, poses = load_data(319)
    train_dataset = SiameseDataset(dataset, True, poses)
    test_dataset = SiameseDataset(dataset, False, poses)

    batch_size = 64
    kwargs = {'num_workers': 2, 'pin_memory': False} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    embedding_model = LocNet()
    model = SiameseNet(embedding_net=embedding_model) 
    
    loss_fn = ContrastiveLoss(margin)
    
    if cuda:
        model.cuda()
    
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 100000
    log_interval = 100    
    fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

    # # with torch.no_grad() :
    # if cuda:
    #     input_data = dataset[0].reshape([1,2,64, 80])
    #     input_data.to(device)
    #     # torch.jit.save(torch.jit.trace(embedding_model), save_name)
 