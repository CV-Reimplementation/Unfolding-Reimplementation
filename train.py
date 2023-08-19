import os,cv2
import torch,math,random
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import matplotlib.image as img
from dataset import my_dataset_threeIn
import torchvision.transforms as transforms
from IterModel3 import Deshadow_netS4
from UTILS import seed_everything
from tqdm import tqdm
from accelerate import Accelerator

if __name__ == '__main__':
    seed_everything()

    accelerator = Accelerator()

    SAVE_PATH = './checkpoint.pth'

    trans_train = transforms.Compose(
        [
            transforms.Resize([256,256]),
            transforms.ToTensor()
        ])
    train_in_path = "/gdata1/zhuyr/Derain_torch/shadow_SRD_test/shadow_re/"
    train_gt_path = "/gdata1/zhuyr/Derain_torch/shadow_SRD_test/shadow_free_re/"
    train_mask_path = "/gdata1/zhuyr/Derain_torch/shadow_SRD_test/shadow_mask_re/"
    filenames_train_in = sorted(os.listdir(train_in_path))
    filenames_train_gt = sorted(os.listdir(train_gt_path))
    filenames_train_mask = sorted((os.listdir(train_mask_path)))
    assert(filenames_train_mask==filenames_train_in)
    assert(filenames_train_in==filenames_train_gt)
    
    net = Deshadow_netS4(ex1=6,ex2=4)
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    criterion = torch.nn.MSELoss()
    print('#generator parameters:',sum(param.numel() for param in net.parameters()))

    train_data = my_dataset_threeIn(
        root_in=train_in_path,root_label =train_gt_path
       ,root_mask=train_mask_path,transform=trans_train)
    train_loader = DataLoader(dataset=train_data, batch_size=2,num_workers=4,shuffle=True)

    net, optimizer, scheduler, train_loader = accelerator.prepare(net, optimizer, scheduler, train_loader)

    epoch = 150

    net.train()
    for curr in range(epoch):
        for i, (data_in, label, mask) in enumerate(tqdm(train_loader)):
            _, _, _, res = net(data_in, mask)
            res = torch.clamp(res, 0., 1.)

            loss = criterion(res, label)
            loss.backward()

            optimizer.step()
            scheduler.step()
        torch.save(net.state_dict(), SAVE_PATH)
        print("Epoch: ", curr)
