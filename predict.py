import argparse
import torch
from src.dataset import SegmentationDataset
from src.unet import UNet
from tqdm import tqdm
import numpy as np
import cv2

def mask_to_image(mask: np.ndarray):

    out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate([0, 255]):
        out[mask == i] = v

    return out

def predict(model, device, val_loader):
    model.eval()
    loop = tqdm(val_loader)

    with torch.no_grad():
        for inputs, filename in loop:
            
            inputs = inputs.to(device)
            mask_pred = model(inputs)

            mask_pred = mask_pred.argmax(1)
            mask_pred.to(torch.int)
            
            result = mask_to_image(mask_pred[0].cpu().numpy())
            cv2.imwrite("Dataset/Train_segmentation2/" + filename[0] + "_seg.png", result)

def main():
    torch.cuda.empty_cache()
    # Training settings
    parser = argparse.ArgumentParser(description='Information Removal at the bottleneck in Deep Neural Networks')
    parser.add_argument('--dev', default="cuda:0")
    parser.add_argument('--output', default='/home/infres/jsun-22/Documents/IMA205/Medical_classification/Dataset/Train_segmentation"')
    args = parser.parse_args()

    args.device = torch.device(args.dev)
    # if args.device != "cpu":
    #     torch.cuda.set_device(args.device)

    model = UNet(in_chans=3, nclass=2, batch_norm=True).to(args.device)
    model.load_state_dict(torch.load('Simple+BN_epoch150.pth')['model_state_dict'])
    
    input_dataset = SegmentationDataset(
        datapath='/home/infres/jsun-22/Documents/IMA205/Medical_classification/Dataset/Train_segmentation',
        predict=True,
    )
    
    input_loader = torch.utils.data.DataLoader(input_dataset, batch_size=1, shuffle=False, num_workers=1)

    predict(model=model, device=args.device, val_loader=input_loader)
    

if __name__ == '__main__':
    main()
