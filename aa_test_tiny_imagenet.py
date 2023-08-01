import os

import torch
from torch.utils.data import DataLoader

from autoattack import AutoAttack
from models import PreActResNet18
from utils.dataset import TinyImageNet200, test_transform
from utils.set_seed import set_seed

test_set = TinyImageNet200('./data/tiny-imagenet', train=False, download=True,
                           transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X, y = [], []
for i, (x, y_) in enumerate(test_loader):
    X = x.to(device)
    y = y_.to(device)

base_dir = './log_tiny_imagenet/'
method_names = ['fgsm_uap']

for method_name in method_names:
    method_path = os.path.join(base_dir, method_name)

    for time_path in os.listdir(method_path):
        time_path = os.path.join(method_path, time_path)

        if 'seed' not in time_path:
            continue

        if os.path.exists(os.path.join(time_path, method_name + '_best_aa_test.log')):
            continue

        set_seed(0)
        model_path = os.path.join(time_path, 'best.pth')

        model = PreActResNet18(num_classes=200).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        log_path = os.path.join(time_path, method_name + '_best_aa_test.log')
        attacker = AutoAttack(model, eps=8.0 / 255, device=device, log_path=log_path)
        attacker.run_standard_evaluation(X, y, bs=100)

        if os.path.exists(os.path.join(time_path, method_name + '_last_aa_test.log')):
            continue

        set_seed(0)
        model_path = os.path.join(time_path, 'last.pth')

        model = PreActResNet18(num_classes=200).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        log_path = os.path.join(time_path, method_name + '_last_aa_test.log')
        attacker = AutoAttack(model, eps=8.0 / 255, device=device, log_path=log_path)
        attacker.run_standard_evaluation(X, y, bs=100)
