import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from fat.fgsm_one_uap import FGSM_One_UAP
from models import MyResNet18
from utils.dataset import train_transform, test_transform


def fgsm_one_uap_test(seed=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_set = CIFAR10('./data/cifar10', train=True, download=True, transform=train_transform)
    test_set = CIFAR10('./data/cifar10', train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

    model = MyResNet18(num_classes=10).to(device)
    opt = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(opt, milestones=[100, 105], gamma=0.1)
    fgsm_uap = FGSM_One_UAP(model, device=device, seed=seed)
    fgsm_uap.train(opt, scheduler, train_loader, test_loader, total_epoch=110)
