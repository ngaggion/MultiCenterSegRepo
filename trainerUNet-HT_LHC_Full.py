import os
import torch
import torch.nn.functional as F
import argparse
import random

from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from utils.utils import CrossVal
from utils.dataset_full import LandmarksDataset, ToTensorSeg, RandomScale, AugColor, Rotate

from models.unet import UNet, OneClassDiceLoss
from torch.nn import CrossEntropyLoss, BCELoss

from medpy.metric.binary import dc
import time
import numpy as np

def evalImageMetricsL(output, target):
    dcp = dc(output == 1, target == 1)
    return dcp

def evalImageMetricsLH(output, target):
    dcp = dc(output == 1, target == 1)
    dcc = dc(output == 2, target == 2)
    return dcp, dcc

def evalImageMetricsLHC(output, target):
    dcp = dc(output == 1, target == 1)
    dcc = dc(output == 2, target == 2)
    dccla = dc(output == 3, target == 3)
    return dcp, dcc, dccla

def trainer(train_datasets, val_datasets, model, config):
    torch.manual_seed(420)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)

    train_dataloaders = []
    val_dataloaders = []

    for dataset in train_datasets:
        tloader = torch.utils.data.DataLoader(dataset, batch_size = config['batch_size'], shuffle = True, num_workers = 0)
        train_dataloaders.append(tloader)
    
    for dataset in val_datasets:
        vloader = torch.utils.data.DataLoader(dataset, batch_size = config['val_batch_size'], shuffle = True, num_workers = 0)
        val_dataloaders.append(vloader)
        
    optimizer = torch.optim.Adam(params = model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])

    train_loss_avg = []
    val_loss_avg = []
    val_dicelungs_avg = []
    val_diceheart_avg = []
    val_dicecla_avg = []

    tensorboard = "Training"
        
    folder = os.path.join(tensorboard, config['name'])

    try:
        os.mkdir(folder)
    except:
        pass 

    writer = SummaryWriter(log_dir = folder)  

    best = 0
    suffix = ".pt"
    
    print('Training ...')
    
    dice_loss = OneClassDiceLoss().to(device)
    ce_loss = BCELoss().to(device)
    
    scheduler = StepLR(optimizer, step_size=config['stepsize'], gamma=config['gamma'])
    
    for epoch in range(config['epochs']):
        model.train()

        train_loss_avg.append(0)
        num_batches = 0
        
        t = time.time()
        
        for j in range(0, 100):
            coin = np.random.uniform(0, 1)

            if coin > (2/3):
                SAMPLE = 2
                sample_batched = iter(train_dataloaders[2]).next()
            elif coin > (1/3):
                SAMPLE = 1
                sample_batched = iter(train_dataloaders[1]).next()
            else:
                SAMPLE = 0
                sample_batched = iter(train_dataloaders[0]).next()

            image, target = sample_batched['image'].to(device), sample_batched['seg'].to(device)
            
            out = model(image)
            sigmoid = torch.sigmoid(out)

            # backpropagation
            optimizer.zero_grad()
            
            if SAMPLE == 2:
                loss_cla = dice_loss(sigmoid[:,2,:,:], (target == 3).float()) + ce_loss(sigmoid[:,2,:,:], (target == 3).float())
                loss_heart = dice_loss(sigmoid[:,1,:,:], (target == 2).float()) + ce_loss(sigmoid[:,1,:,:], (target == 2).float())
            elif SAMPLE == 1:
                loss_cla = 0
                loss_heart = dice_loss(sigmoid[:,1,:,:], (target == 2).float()) + ce_loss(sigmoid[:,1,:,:], (target == 2).float())
            else:
                loss_cla = 0
                loss_heart = 0            

            loss_lungs = dice_loss(sigmoid[:,0,:,:], (target == 1).float()) + ce_loss(sigmoid[:,0,:,:], (target == 1).float())

            loss = loss_lungs + loss_heart + loss_cla

            train_loss_avg[-1] += loss.item()

            loss.backward()
            optimizer.step()

            num_batches += 1
        
        t2 = time.time()
        
        print('Training epoch took %.3f seconds' %(t2-t))

        train_loss_avg[-1] /= num_batches

        num_batches_l = 0
        num_batches_h = 0
        num_batches_c = 0

        model.eval()
        val_loss_avg.append(0)
        val_dicelungs_avg.append(0)
        val_diceheart_avg.append(0)
        val_dicecla_avg.append(0)
        
        t = time.time()
        with torch.no_grad():
            for j in range(0, 3):
                for sample_batched in val_dataloaders[j]:
                    image, target = sample_batched['image'].to(device), sample_batched['seg'].to(device)

                    out = model(image)
                    sigmoid = torch.sigmoid(out)
                    
                    seg = torch.zeros([1024,1024])
                    seg[sigmoid[0,0,:,:] > 0.5] = 1
                    seg[sigmoid[0,1,:,:] > 0.5] = 2
                    seg[sigmoid[0,2,:,:] > 0.5] = 3
                    
                    if j == 0:
                        dcl = evalImageMetricsL(seg.cpu().numpy(), target[0,:,:].cpu().numpy())
                        dch = 0
                        dcc = 0
                        num_batches_l += 1
                        val_loss_avg[-1] += dcl

                    if j == 1:
                        dcl, dch = evalImageMetricsLH(seg.cpu().numpy(), target[0,:,:].cpu().numpy())
                        dcc = 0
                        num_batches_l += 1
                        num_batches_h += 1
                        val_loss_avg[-1] += (dcl+dch)/2

                    elif j == 2:
                        dcl, dch, dcc = evalImageMetricsLHC(seg.cpu().numpy(), target[0,:,:].cpu().numpy())
                        num_batches_l += 1
                        num_batches_h += 1
                        num_batches_c += 1
                        val_loss_avg[-1] += (dcl+dch+dcc)/3
                    
                    val_dicelungs_avg[-1] += dcl
                    val_diceheart_avg[-1] += dch
                    val_dicecla_avg[-1] += dcc                                        
            
        val_loss_avg[-1] /= num_batches_l
        val_dicelungs_avg[-1] /= num_batches_l
        val_diceheart_avg[-1] /= num_batches_h
        val_dicecla_avg[-1] /= num_batches_c
        
        t2 = time.time()
        
        print('Epoch [%d / %d] validation Dice: %.3f, took %.3f seconds' % (epoch+1, config['epochs'], val_loss_avg[-1], t2-t))
        print('Dice Lungs %.3f. Dice Heart %.3f, Dice Clavicles %.3f' % (val_dicelungs_avg[-1], val_diceheart_avg[-1], val_dicecla_avg[-1]))
        
        writer.add_scalar('Train/Loss', train_loss_avg[-1], epoch)
        writer.add_scalar('Validation/Dice', val_loss_avg[-1], epoch)
        writer.add_scalar('Validation/Dice Lungs', val_dicelungs_avg[-1], epoch)
        writer.add_scalar('Validation/Dice Heart', val_diceheart_avg[-1], epoch)
        writer.add_scalar('Validation/Dice Cla', val_dicecla_avg[-1], epoch)
        
        if val_loss_avg[-1] > best:
            best = val_loss_avg[-1]
            print('Model Saved Dice')
            out = "bestDice.pt"
            torch.save(model.state_dict(), os.path.join(folder, out))

        scheduler.step()

        print('')
        
    torch.save(model.state_dict(), os.path.join(folder, "final.pt"))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", type=str)    
    parser.add_argument("--load", help="enter the folder where the weights are saved", default = "None", type=str)
    parser.add_argument("--inputsize", default = 1024, type=int)
    parser.add_argument("--epochs", default = 1000, type = int)
    parser.add_argument("--lr", default = 1e-4, type = float    )
    parser.add_argument("--stepsize", default = 3000, type = int)
    parser.add_argument("--gamma", default = 0.1, type = float)
    
    ## 5-fold Cross validation fold
    parser.add_argument("--fold", default = 1, type = int)
    parser.add_argument('--organs', type=str, default = 'LHC')
    
    # Define the output: only lungs, or lungs and heart by default
    parser.add_argument('--original', dest='original', action='store_true')
    parser.set_defaults(original = False)

    config = parser.parse_args()
    config = vars(config)

    inputSize = config['inputsize']

    images_lungs = open("train_images_lungs.txt",'r').read().splitlines()
    images_heart = open("train_images_heart.txt",'r').read().splitlines()

    images_jsrt = [image for image in images_heart if "JP" in image]
    images_pad = [image for image in images_heart if not "JP" in image]
    
    print("Lungs:", len(images_lungs))
    random.Random(13).shuffle(images_lungs)
    print("Heart:", len(images_pad))
    random.Random(13).shuffle(images_pad)
    print("Cla:", len(images_jsrt))
    random.Random(13).shuffle(images_jsrt)
        
    print('Fold %s'%config['fold'], 'of 5')
    images_train_L, images_val_L = CrossVal(images_lungs, config['fold'])
    images_train_H, images_val_H = CrossVal(images_pad, config['fold'])
    images_train_C, images_val_C = CrossVal(images_jsrt, config['fold'])

    train_dataset_lungs = LandmarksDataset(images=images_train_L,
                                     img_path="../Chest-xray-landmark-dataset/Images",
                                     label_path="../Chest-xray-landmark-dataset/landmarks",
                                     organ = 'L',
                                     transform = transforms.Compose([
                                                 RandomScale(),
                                                 Rotate(3),
                                                 AugColor(0.40),
                                                 ToTensorSeg()])
                                     )

    val_dataset_lungs = LandmarksDataset(images=images_val_L,
                                     img_path="../Chest-xray-landmark-dataset/Images",
                                     label_path="../Chest-xray-landmark-dataset/landmarks",
                                     organ = 'L',
                                     transform = ToTensorSeg()
                                     )

    train_dataset_heart = LandmarksDataset(images=images_train_H,
                                     img_path="../Chest-xray-landmark-dataset/Images",
                                     label_path="../Chest-xray-landmark-dataset/landmarks",
                                     organ = 'LH',
                                     transform = transforms.Compose([
                                                 RandomScale(),
                                                 Rotate(3),
                                                 AugColor(0.40),
                                                 ToTensorSeg()])
                                     )

    val_dataset_heart = LandmarksDataset(images=images_val_H,
                                     img_path="../Chest-xray-landmark-dataset/Images",
                                     label_path="../Chest-xray-landmark-dataset/landmarks",
                                     organ = 'LH',
                                     transform = ToTensorSeg()
                                     )
                                                                       
 
    train_dataset_cla = LandmarksDataset(images=images_train_C,
                                     img_path="../Chest-xray-landmark-dataset/Images",
                                     label_path="../Chest-xray-landmark-dataset/landmarks",
                                     organ = "LHC",
                                     transform = transforms.Compose([
                                                 RandomScale(),
                                                 Rotate(3),
                                                 AugColor(0.40),
                                                 ToTensorSeg()])
                                     )

    val_dataset_cla = LandmarksDataset(images=images_val_C,
                                     img_path="../Chest-xray-landmark-dataset/Images",
                                     label_path="../Chest-xray-landmark-dataset/landmarks",
                                     organ = "LHC",
                                     transform = ToTensorSeg()
                                     )


    train_datasets = [train_dataset_lungs,train_dataset_heart,train_dataset_cla]
    val_datasets = [val_dataset_lungs,val_dataset_heart,val_dataset_cla]
                
    config['latents'] = 64
    config['batch_size'] = 4
    config['val_batch_size'] = 1
    config['weight_decay'] = 1e-5
    
    n_classes = len(config['organs'])

    model = UNet(n_classes = n_classes)    
    trainer(train_datasets, val_datasets, model, config)