import argparse
import torch
from src.dataset import SegmentationDataset
import wandb

from src.unet import UNet

from tqdm import tqdm
import numpy as np
import albumentations as A
import cv2
from src.dice_score import dice_coeff, dice_loss
import torch.nn.functional as F

class_label = {
    0: 'peau',
    1: 'maladie',
}

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
        loss = args.criterion(masks_pred.squeeze(1), true_masks.float())
        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float())
        
        args.optimizer.zero_grad()
        loss.backward()
        args.optimizer.step()

        loop.set_postfix(loss=loss.item())
        
        running_loss += loss.item()

    train_loss = running_loss/iteration
    
    print('Train Loss: %.3f'%(train_loss))
    return(train_loss)

def eval(model, args, val_loader):
    model.eval()

    running_loss = 0

    saved_images = [None,None,None]

    iteration = 0

    loop = tqdm(val_loader)

    with torch.no_grad():
        for batch_idx, (inputs, true_masks) in enumerate(loop):
            iteration += 1
            
            inputs, true_masks = inputs.to(args.device), true_masks.to(args.device)
            masks_pred=model(inputs)


            loss = args.criterion(masks_pred,true_masks)
            running_loss += loss.item()

            masks_pred = masks_pred.argmax(1)
            true_masks = true_masks.argmax(1)
            masks_pred.to(torch.int)
            true_masks.to(torch.int)

            dice_score = dice_coeff(masks_pred, true_masks, reduce_batch_first=False)

            if(batch_idx == 0):
                torch.cuda.synchronize()
                
                saved_images[0] = inputs[1].cpu().numpy().transpose(1, 2, 0)
                label = true_masks[1].cpu().numpy()
                output = masks_pred[1].cpu().numpy()
                
                image = np.zeros((IMG_SIZE[0], IMG_SIZE[1]))
                for i in range(2):
                    image[:, :] = label == i
                saved_images[1] = image
                cv2.imwrite('label.png', image)
                
                image = np.zeros((IMG_SIZE[0], IMG_SIZE[1]))
                for i in range(2):
                    image[:, :] = output == i
                saved_images[2] = image
                
    val_loss = running_loss/iteration
    dice_score = dice_score/iteration

    print('Eval Loss: %.3f'%(val_loss))
    return(val_loss, dice_score, saved_images)

def main():
    torch.cuda.empty_cache()
    # Training settings
    parser = argparse.ArgumentParser(description='Information Removal at the bottleneck in Deep Neural Networks')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    parser.add_argument('--dev', default="cuda:1")
    parser.add_argument('--momentum-sgd', type=float, default=0.9, metavar='M', help='Momentum')
    parser.add_argument('--datapath', default='LPCVCDataset')
    parser.add_argument('--name', default='RUN')
    args = parser.parse_args()

    args.device = torch.device(args.dev)
    if args.dev != "cpu":
        torch.cuda.set_device(args.device)

    model = UNet(in_chans=3, nclass=2).to(args.device)

    # transform = A.Compose([A.Resize(width=IMG_SIZE[1], height=IMG_SIZE[0], interpolation=cv2.INTER_NEAREST)])

    aug_data = A.Compose([
        A.Resize(width=IMG_SIZE[1], height=IMG_SIZE[0], interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=[-60, 60], p=0.8, interpolation=cv2.INTER_NEAREST),
        A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.2], contrast_limit=0.2, p=0.3),
    ], p=1.0)

    
    train_dataset = SegmentationDataset(
        datapath='/home/infres/jsun-22/Documents/IMA205/Medical_classification/Dataset/Train_seg',
        augmentation = aug_data,
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
    #args.criterion = utils.losses.DiceLoss()
    
    args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(args.optimizer, 'max', patience=8)
    
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
        val_loss,dice_score, saved_images = eval(model, args, val_loader)

        input_image, target_image, pred_image = saved_images[0], saved_images[1], saved_images[2]
        wandb.log({
            "train_loss": train_loss, 
            "val_loss": val_loss, 
            "val_dice": dice_score, 
            
            "input_image" : wandb.Image(input_image,
                            masks={"predictions": {"mask_data": pred_image, "class_labels": class_label},
                                    "ground_truth": {"mask_data": target_image, "class_labels": class_label}}
                            ), 
            "target_image" : wandb.Image(target_image), 
            "pred_image" : wandb.Image(pred_image),
            "learning rate": args.optimizer.param_groups[0]['lr']
            })

        if epoch > 20 and (dice_score > best_dice):
             torch.save(model, 'checkpoint/' + args.name + '.pth')
             best_dice = dice_score

        scheduler.step(dice_score)

    wandb.finish()

if __name__ == '__main__':
    main()
