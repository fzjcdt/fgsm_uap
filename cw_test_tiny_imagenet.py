import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import PreActResNet18
from utils.dataset import TinyImageNet200, test_transform
from utils.logger import Logger
from utils.set_seed import set_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_mean = (0.0, 0.0, 0.0)
data_std = (1.0, 1.0, 1.0)
mu = torch.tensor(data_mean).view(3, 1, 1).to(device)
std = torch.tensor(data_std).view(3, 1, 1).to(device)

upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()


def cw_Linf_attack(model, X, y, epsilon, alpha, attack_iters, restarts):
    """
    from https://github.com/jiaxiaojunQAQ/FGSM-PGI/blob/master/utils.py
    """
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for zz in range(restarts):
        delta = torch.zeros_like(X).to(device)
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)

            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = CW_loss(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_cw(test_loader, model, attack_iters=50, restarts=1):
    alpha = (2 / 255.) / std
    epsilon = (8 / 255.) / std
    acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)
        pgd_delta = cw_Linf_attack(model, X, y, epsilon, alpha, attack_iters=attack_iters, restarts=restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return acc / n


test_set = TinyImageNet200('./data/tiny-imagenet', train=False, download=True,
                           transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=4)


base_dir = './log_tiny_imagenet/'
method_names = ['fgsm_uap']

for method_name in method_names:
    method_path = os.path.join(base_dir, method_name)
    if not os.path.exists(method_path):
        continue

    for time_path in os.listdir(method_path):
        time_path = os.path.join(method_path, time_path)

        if 'seed' not in time_path or not os.path.exists(time_path):
            continue

        if time_path is None or os.path.exists(os.path.join(time_path, method_name + '_best_cw_test.log')):
            continue

        set_seed(0)
        logger = Logger(os.path.join(time_path, method_name + '_best_cw_test.log'))
        model_path = os.path.join(time_path, 'best.pth')

        model = PreActResNet18(num_classes=200).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        logger.log('c&w accuracy: {} %'.format(100 * evaluate_cw(test_loader, model)))

        # last
        if time_path is None or os.path.exists(os.path.join(time_path, method_name + '_last_cw_test.log')):
            continue

        set_seed(0)
        logger = Logger(os.path.join(time_path, method_name + '_last_cw_test.log'))
        model_path = os.path.join(time_path, 'last.pth')

        model = PreActResNet18(num_classes=200).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logger.log('c&w accuracy: {} %'.format(100 * evaluate_cw(test_loader, model)))
