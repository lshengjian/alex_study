import torch,os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
    
def get_stat(data_dir = 'd:/dataset/ants-bees/train',trans=None):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]) if trans is None else trans
    dataset = ImageFolder(root=data_dir, transform=transform)

    # 计算均值和方差
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=0)
    means=torch.zeros(3)
    stds=torch.zeros(3)
    cnt=0
    for imgs,_ in tqdm(loader):
        means += torch.mean(imgs, dim=(0, 2, 3))
        stds += torch.std(imgs, dim=(0, 2, 3))
        cnt+=1
    return means/cnt,stds/cnt

def imshow(images:torch.Tensor, titles=None,mean=[0.5199, 0.4703, 0.3442],std=[0.2700, 0.2535, 0.2807]):
    plt.figure(figsize=(8, 8))
    # 遍历3x3网格中的每个图像
    for i in range(9):
        img=images[i].permute((1,2,0)).numpy()
        img=img*np.array(mean)+np.array(std)
        img=img.clip(0,1)
        plt.subplot(3, 3, i+1) # 3行3列的子图
        img = (img*255).astype(np.uint8)
        plt.imshow(img, interpolation='nearest')  # 调整图像通道顺序
        if titles is not None:
            plt.title(titles[i])
        plt.axis('off')  # 不显示坐标轴

    plt.show()
    #plt.pause(5)  # pause a bit so that plots are updated




if __name__ == '__main__':
    # mean,std=get_stat()
    # print(f'Mean: {mean}')
    # print(f'Std: {std}')
    data_dir = 'd:/dataset/ants-bees'
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5199, 0.4703, 0.3442],[0.2700, 0.2535, 0.2807])
    ])
    image_dataset = ImageFolder(os.path.join(data_dir, 'val'),transform)
    dataloader = DataLoader(image_dataset, batch_size=9,shuffle=True, num_workers=0)
    inputs, classes = next(iter(dataloader))
    imshow(inputs,classes.numpy())
