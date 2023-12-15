import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from autoattack import AutoAttack
from models import ResNet18
from utils.dataset import test_transform
from utils.set_seed import set_seed

test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data/cifar10', train=False, transform=test_transform),
                                          batch_size=10000, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X, y = [], []
for i, (x, y_) in enumerate(test_loader):
    X = x.to(device)
    y = y_.to(device)

base_dir = './log_cifar10/'
method_names = ['fgsm_uap']

for method_name in method_names:
    method_path = os.path.join(base_dir, method_name)

    for time_path in os.listdir(method_path):
        time_path = os.path.join(method_path, time_path)
        if 'seed' not in time_path:
            continue

        # best
        if os.path.exists(os.path.join(time_path, method_name + '_best_aa_test.log')):
            continue

        set_seed(0)
        model_path = os.path.join(time_path, 'best.pth')

        model = ResNet18().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        log_path = os.path.join(time_path, method_name + '_best_aa_test.log')
        attacker = AutoAttack(model, eps=8.0 / 255, device=device, log_path=log_path)
        attacker.run_standard_evaluation(X, y, bs=100)

        # last
        if os.path.exists(os.path.join(time_path, method_name + '_last_aa_test.log')):
            continue

        set_seed(0)
        model_path = os.path.join(time_path, 'last.pth')

        model = ResNet18().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        log_path = os.path.join(time_path, method_name + '_last_aa_test.log')
        attacker = AutoAttack(model, eps=8.0 / 255, device=device, log_path=log_path)
        attacker.run_standard_evaluation(X, y, bs=100)
