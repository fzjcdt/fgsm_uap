import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from fat.fgsm_uap import FGSM_UAP
from models import MyPreActResNet18
from utils import set_seed
from utils import train_transform_tiny_imagenet, test_transform, TinyImageNet200


def fgsm_uap_test(seed=0):
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_set = TinyImageNet200('./data/tiny-imagenet', train=True, download=True,
                                transform=train_transform_tiny_imagenet)

    test_set = TinyImageNet200('./data/tiny-imagenet', train=False, download=True,
                               transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

    model = MyPreActResNet18(num_classes=200).to(device)
    opt = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(opt, milestones=[100], gamma=0.1)

    fgsm_uap = FGSM_UAP(model, device=device, log_dir='./log_tiny_imagenet/', seed=seed)
    fgsm_uap.train(opt, scheduler, train_loader, test_loader, total_epoch=110,
                   label_smoothing=0.4, weight_average=True, tau=0.9995,
                   uap_num=300, class_num=200, image_shape=(3, 64, 64))
