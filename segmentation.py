import argparse
import torch
import segmentation_models_pytorch.utils as utils
from src.dataset import SegmentationDataset
import wandb

from src.unet import UNet
from src.accuracy import AccuracyTracker

from tqdm import tqdm
import numpy as np
import albumentations as A
import cv2
from src.dice_score import dice_loss
from src.dice_score import dice_coeff
import torch.nn.functional as F

class_set = wandb.Classes(
    [
        {"name": "peau", "id": 0},
        {"name": "maladie", "id": 1},
    ]
)
class_labels = {0: "peau", 1: "maladie"}
IMG_SIZE = (512, 512)

def train(model, args, train_loader):
    model.train()
    model.to(args.device)

    running_loss = 0
    iteration = 0

    loop = tqdm(train_loader)
    
    for batch_idx, (inputs, true_masks) in enumerate(loop):
        iteration+=1
        
        inputs, true_masks = inputs.to(args.device), true_masks.to(args.device)

        masks_pred=model(inputs)
        
        loss = args.criterion(masks_pred,true_masks)
        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float())
        
        args.optimizer.zero_grad()
        loss.backward()
        args.optimizer.step()

        loop.set_postfix(loss=loss.item())
        
        running_loss += loss.item()

    train_loss = running_loss/iteration
    
    # args.sched.step()
    print('Train Loss: %.3f'%(train_loss))
    return(train_loss)

def eval(model, args, val_loader):
    model.eval()

    running_loss = 0
    dice_score = 0

    saved_images = [None, None, None]
    iteration = 0

    loop = tqdm(val_loader)

    with torch.no_grad():
        for batch_idx, (inputs, true_masks) in enumerate(loop):
            iteration+=1
            
            inputs, true_masks = inputs.to(args.device), true_masks.to(args.device)
            masks_pred=model(inputs)

            torch.cuda.synchronize()

            loss = args.criterion(masks_pred,true_masks)
            
            running_loss += loss.item()

            masks_pred = masks_pred.argmax(dim=1)
            true_masks = true_masks.argmax(dim=1)

            masks_pred.to(torch.int)
            true_masks.to(torch.int)
            
            dice_score += dice_coeff(masks_pred, true_masks, reduce_batch_first=False)
            
            if(batch_idx == 0):
                saved_images[0] = inputs[0].cpu().numpy().transpose(1, 2, 0)
                
                #Visualizing the output
                output = masks_pred[0].cpu().numpy()
                label = true_masks[0].cpu().numpy()
                
                image = np.zeros((512, 512), dtype=np.uint8)
                for i in range(2):
                    image[output == i] = i
                saved_images[1] = image
                
                image = np.zeros((512, 512), dtype=np.uint8)
                for i in range(2):
                    image[label == i] = i
                saved_images[2] = image

                cv2.imwrite(('true_masks.png'), saved_images[0])
                cv2.imwrite(('masks_pred.png'), saved_images[1])

    val_loss = running_loss/iteration
    dice_score = dice_score/iteration

    print('Eval Loss: %.3f'%(val_loss))
    return(val_loss, dice_score, saved_images)

def main():
    torch.cuda.empty_cache()
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Segmentation')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dev', default="cuda:1")
    parser.add_argument('--momentum-sgd', type=float, default=None, metavar='M', help='Momentum')
    parser.add_argument('--name', default='RUN')
    args = parser.parse_args()

    args.device = torch.device(args.dev)
    if args.dev != "cpu":
        torch.cuda.set_device(args.device)

    model = UNet(nclass=2, in_chans=3).to(args.device)

    aug_data = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=[-60, 60], p=0.8, interpolation=cv2.INTER_NEAREST),
        # A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.2], contrast_limit=0.2, p=0.3),
    ], p=1)
    
    train_dataset = SegmentationDataset(
        datapath='/home/infres/jsun-22/Documents/IMA205/Medical_classification/Dataset/Train_seg',
        # augmentation=aug_data,
        train=True
    )
    
    val_dataset = SegmentationDataset(
        datapath='/home/infres/jsun-22/Documents/IMA205/Medical_classification/Dataset/Test_seg',
        # transform=transform, 
        train=False
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    args.criterion = torch.nn.BCEWithLogitsLoss()
    
    args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #args.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(args.optimizer, 'max', patience=5) 
    
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(args.optimizer, gamma=0.1)
    #args.sched = torch.optim.lr_scheduler.OneCycleLR(args.optimizer, 0.001, total_steps=len(train_loader)*args.epochs)
    # args.scaler = torch.cuda.amp.GradScaler()
    # args.sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #      args.optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    # )
    #args.sched = torch.optim.lr_scheduler.StepLR(args.optimizer,step_size=500,gamma=0.1,verbose=True)

    wandb.init(project="IMA205")
    wandb.run.name = args.name
    wandb.config.epochs = args.epochs
    wandb.config.batch_size = args.batch_size
    wandb.config.learning_rate = args.lr
    wandb.config.weight_decay = args.weight_decay
    wandb.config.train_dataset_length = len(train_dataset)
    wandb.config.val_dataset_length = len(val_dataset)
    wandb.config.optmizer = "ADAMW"
    wandb.config.momentum = args.momentum_sgd
    
    best_dice = 0

    for epoch in range(1, args.epochs+1):
        print('\nEpoch : %d'%epoch)
        train_loss = train(model, args, train_loader)
        
        if epoch % 5 == 1:
            val_loss, dice_score, saved_images = eval(model, args, val_loader)

            input, pred_image, target_image = saved_images[0], saved_images[1], saved_images[2]

        wandb.log({
            "train_loss": train_loss, 
            
            "val_loss": val_loss, 
            "dice_score": dice_score,
            
            "Segmentation" : wandb.Image(
                input,
                masks={
                    "predictions": {"mask_data": pred_image, "class_labels": class_labels}, 
                    "ground_truth": {"mask_data": target_image, "class_labels": class_labels}
                    },
                classes=class_set
                ),
            "prediction" : wandb.Image(pred_image),
            "target" : wandb.Image(target_image),
            # "learning rate": scheduler.get_lr()[-1]
            })
            
        if epoch % 20 == 0 and (dice_score > best_dice):
            torch.save(model, 'checkpoint/' + args.name + '_' + str(epoch)+'_dice_' + str(dice_score)+'.pth')
            best_dice = dice_score

        # scheduler.step(dice_score)

wandb.finish()

if __name__ == '__main__':
    main()
