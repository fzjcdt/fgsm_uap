import os

import torch
from torch.utils.data import DataLoader
from torchattacks import PGD
from torchvision import datasets

from models import ResNet18
from utils.dataset import test_transform
from utils.logger import Logger
from utils.set_seed import set_seed

test_loader = torch.utils.data.DataLoader(datasets.CIFAR100('./data/cifar100', train=False, transform=test_transform),
                                          batch_size=100, shuffle=False, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_dir = './log_cifar100/'
method_names = ['fgsm_uap']

for method_name in method_names:
    method_path = os.path.join(base_dir, method_name)

    for time_path in os.listdir(method_path):
        time_path = os.path.join(method_path, time_path)
        if 'seed' not in time_path:
            continue

        if os.path.exists(os.path.join(time_path, method_name + '_best_pgd_test.log')):
            continue

        set_seed(0)
        logger = Logger(os.path.join(time_path, method_name + '_best_pgd_test.log'))
        model_path = os.path.join(time_path, 'best.pth')

        model = ResNet18(num_classes=100).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        attacker = PGD(model, eps=8.0 / 255, alpha=2.0 / 255, steps=10)
        clean_correct, correct, total = 0, 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pre = torch.max(outputs.data, 1)
            clean_correct += (pre == labels).sum().item()

            adv_images = attacker(images, labels)
            outputs = model(adv_images)
            _, pre = torch.max(outputs.data, 1)
            correct += (pre == labels).sum().item()
            total += labels.size(0)
        logger.log('clean accuracy: {} %'.format(100 * clean_correct / total))
        logger.log('pgd-10 accuracy: {} %'.format(100 * correct / total))

        attacker = PGD(model, eps=8.0 / 255, alpha=2.0 / 255, steps=20)
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            adv_images = attacker(images, labels)
            outputs = model(adv_images)
            _, pre = torch.max(outputs.data, 1)
            correct += (pre == labels).sum().item()
            total += labels.size(0)
        logger.log('pgd-20 accuracy: {} %'.format(100 * correct / total))

        attacker = PGD(model, eps=8.0 / 255, alpha=2.0 / 255, steps=50)
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            adv_images = attacker(images, labels)
            outputs = model(adv_images)
            _, pre = torch.max(outputs.data, 1)
            correct += (pre == labels).sum().item()
            total += labels.size(0)
        logger.log('pgd-50 accuracy: {} %'.format(100 * correct / total))

        output_path = os.path.join(time_path, 'output.log')
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                lines = f.readlines()
                flag = False
                for line in lines:
                    line = line.strip()
                    if line == 'total_training_time:':
                        flag = True
                    if flag and line != '':
                        logger.log(line)

        logger.new_line()

        # last
        if os.path.exists(os.path.join(time_path, method_name + '_last_pgd_test.log')):
            continue

        set_seed(0)
        logger = Logger(os.path.join(time_path, method_name + '_last_pgd_test.log'))
        model_path = os.path.join(time_path, 'last.pth')

        model = ResNet18(num_classes=100).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        attacker = PGD(model, eps=8.0 / 255, alpha=2.0 / 255, steps=10)
        clean_correct, correct, total = 0, 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pre = torch.max(outputs.data, 1)
            clean_correct += (pre == labels).sum().item()

            adv_images = attacker(images, labels)
            outputs = model(adv_images)
            _, pre = torch.max(outputs.data, 1)
            correct += (pre == labels).sum().item()
            total += labels.size(0)
        logger.log('clean accuracy: {} %'.format(100 * clean_correct / total))
        logger.log('pgd-10 accuracy: {} %'.format(100 * correct / total))

        attacker = PGD(model, eps=8.0 / 255, alpha=2.0 / 255, steps=20)
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            adv_images = attacker(images, labels)
            outputs = model(adv_images)
            _, pre = torch.max(outputs.data, 1)
            correct += (pre == labels).sum().item()
            total += labels.size(0)
        logger.log('pgd-20 accuracy: {} %'.format(100 * correct / total))

        attacker = PGD(model, eps=8.0 / 255, alpha=2.0 / 255, steps=50)
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            adv_images = attacker(images, labels)
            outputs = model(adv_images)
            _, pre = torch.max(outputs.data, 1)
            correct += (pre == labels).sum().item()
            total += labels.size(0)
        logger.log('pgd-50 accuracy: {} %'.format(100 * correct / total))

        output_path = os.path.join(time_path, 'output.log')
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                lines = f.readlines()
                flag = False
                for line in lines:
                    line = line.strip()
                    if line == 'total_training_time:':
                        flag = True
                    if flag and line != '':
                        logger.log(line)

        logger.new_line()
