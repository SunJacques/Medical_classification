import argparse
import torch
from src.dataset import SegmentationDataset
import wandb
from src.unet import UNet
from tqdm import tqdm
from torchvision.transforms import v2
from src.dice_score import dice_coeff, dice_loss
import torch.nn.functional as F

class_label = {
    0: 'peau',
    1: 'maladie',
}

IMG_SIZE = (512, 512)

def train(model, args, train_loader,epoch):
    model.train()
    model.to(args.device)

    running_loss = 0
    iteration = 0
    iters = len(train_loader)
    loop = tqdm(train_loader)
    
    for batch_idx, (inputs, true_masks) in enumerate(loop):
        iteration+=1
        
        inputs, true_masks = inputs.to(args.device), true_masks.to(args.device)
        masks_pred = model(inputs)
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
    dice_score = 0

    loop = tqdm(val_loader)

    with torch.no_grad():
        for batch_idx, (inputs, true_masks) in enumerate(loop):
            iteration += 1
            inputs, true_masks = inputs.to(args.device), true_masks.to(args.device)
            masks_pred=model(inputs)

            loss = args.criterion(masks_pred,true_masks)
            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float())
            
            running_loss += loss.item()

            masks_pred = masks_pred.argmax(1)
            true_masks = true_masks.argmax(1)
            masks_pred.to(torch.int)
            true_masks.to(torch.int)

            dice_score += dice_coeff(masks_pred, true_masks, reduce_batch_first=False)

            if(batch_idx == 0):
                torch.cuda.synchronize()
                
                saved_images[0] = inputs[1].cpu().numpy().transpose(1, 2, 0)
                saved_images[1] = true_masks[1].cpu().numpy()
                saved_images[2] = masks_pred[1].cpu().numpy()
                
    val_loss = running_loss/iteration
    dice_score = dice_score/iteration

    print('Eval Loss: %.3f'%(val_loss))
    return(val_loss, dice_score, saved_images)

def main():
    torch.cuda.empty_cache()
    # Training settings
    parser = argparse.ArgumentParser(description='Information Removal at the bottleneck in Deep Neural Networks')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dev', default="cuda:1")
    parser.add_argument('--momentum-sgd', type=float, default=None, metavar='M', help='Momentum')
    parser.add_argument('--BN', type=bool, default=False)
    parser.add_argument('--name', default='RUN')
    args = parser.parse_args()

    args.device = torch.device(args.dev)
    if args.dev != "cpu":
        torch.cuda.set_device(args.device)

    model = UNet(in_chans=3, nclass=2, batch_norm = True).to(args.device)

    # Data Augmentation
    aug_data1 = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=[-60,60]),
    ])
    aug_data2 = v2.ColorJitter(brightness=[0.7, 1.3], contrast=[0.7, 1.3], saturation=[0.7, 1.3])
    
    train_dataset = SegmentationDataset(
        datapath='/home/infres/jsun-22/Documents/IMA205/Medical_classification/Dataset/Train_seg',
        augmentation = [aug_data1, aug_data2],
        train=True
    )
    val_dataset = SegmentationDataset(
        datapath='/home/infres/jsun-22/Documents/IMA205/Medical_classification/Dataset/Test_seg',
        train=False
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


    args.criterion = torch.nn.BCEWithLogitsLoss()
    args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    wandb.init(project="IMA205", dir="tmp/")
    wandb.run.name = args.name
    wandb.config.epochs = args.epochs
    wandb.config.batch_size = args.batch_size
    wandb.config.learning_rate = args.lr
    wandb.config.weight_decay = args.weight_decay
    wandb.config.train_dataset_length = len(train_dataset)
    wandb.config.val_dataset_length = len(val_dataset)
    wandb.config.optmizer = "ADAMW"
    wandb.config.momentum = args.momentum_sgd

    for epoch in range(1, args.epochs+1):
        print('\nEpoch : %d'%epoch)
        train_loss = train(model, args, train_loader, epoch)
        val_loss,dice_score, saved_images = eval(model, args, val_loader)

        input_image, target_image, pred_image = saved_images[0], saved_images[1], saved_images[2]
        wandb.log({
            "train_loss": train_loss, 
            "val_loss": val_loss, 
            "dice_score": dice_score, 
            
            "input_image" : wandb.Image(input_image,
                            masks={"predictions": {"mask_data": pred_image, "class_labels": class_label},
                                    "ground_truth": {"mask_data": target_image, "class_labels": class_label}}
                            ), 
            "target_image" : wandb.Image(target_image), 
            "pred_image" : wandb.Image(pred_image),
            "learning rate": args.optimizer.param_groups[0]['lr']
            })

        if (epoch > 100 and epoch % 5 == 0) or (epoch in [30,50,100]):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': args.optimizer.state_dict(),
            }, 'checkpoints2/' + args.name + "_epoch" + str(epoch) +'.pth')
            
        if args.sched in [0,1]:
            args.scheduler.step()

    wandb.finish()

if __name__ == '__main__':
    main()
