import copy
import os
import time

import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torchattacks import PGD, FGSM

from .at_base import ATBase


class FGSM_One_UAP(ATBase):

    def __init__(self, model, eps=8.0 / 255, log_dir='./log_cifar10/', name='fgsm_one_uap', device=None, seed=0):
        super(FGSM_One_UAP, self).__init__(model, eps=eps, log_dir=log_dir, name=name, device=device, seed=seed)
        self.momentum_decay = 0.3
        self.lamda = 10
        self.uap_eps = 10.0 / 255

    def train(self, opt, scheduler, train_loader, test_loader, total_epoch=110, label_smoothing=0.4,
              weight_average=True, tau=0.999, image_shape=(3, 32, 32), eval_start=90):
        if label_smoothing is not None:
            criterion = CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            criterion = CrossEntropyLoss()

        loss_fn = MSELoss()
        uaps = torch.zeros(image_shape).uniform_(-self.uap_eps, self.uap_eps).to(self.device)
        uaps = torch.clamp(self.uap_eps * torch.sign(uaps), -self.uap_eps, self.uap_eps)

        momentum = torch.zeros(image_shape).to(self.device)

        if weight_average:
            wa_model = copy.deepcopy(self.model)
            exp_avg = self.model.state_dict()
            if tau is None:
                raise ValueError('tau should not be None when weight_average is True')
            pgd_attacker = PGD(wa_model, eps=self.eps, alpha=2.0 / 255, steps=10)
            fgsm_attacker = FGSM(wa_model, eps=self.eps)
        else:
            pgd_attacker = PGD(self.model, eps=self.eps, alpha=2.0 / 255, steps=10)
            fgsm_attacker = FGSM(self.model, eps=self.eps)

        self.logger.log('scheduler: {}'.format(scheduler.__class__.__name__))
        self.logger.log('label smoothing: {}'.format(label_smoothing))
        self.logger.log('weight average: {}, tau: {}'.format(weight_average, tau))
        self.logger.new_line()
        self.logger.new_line()

        best_pgd_acc, best_test_acc, total_training_time, total_test_time = 0.0, 0.0, 0.0, 0.0
        train_acc_list, test_acc_list, pgd_acc_list, fgsm_acc_list = [], [], [], []

        for epoch in range(total_epoch):
            self.logger.log('============ Epoch {} ============'.format(epoch))
            self.model.train()
            train_loss, train_correct, train_n = 0, 0, 0
            start_time = time.time()

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                adv_images = images.clone() + uaps
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
                adv_images.requires_grad_(True)

                ori_output = self.model(adv_images)
                ori_loss = criterion(ori_output, labels)
                ori_loss.backward(retain_graph=True)

                grad_x = adv_images.grad.detach()
                adv_images = adv_images + self.eps * adv_images.grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

                output = self.model(adv_images)
                loss = criterion(output, labels) + self.lamda * loss_fn(output.float(), ori_output.float())

                # train
                opt.zero_grad()
                loss.backward()
                opt.step()

                grad_norm = torch.norm(grad_x, p=1)
                cur_grad = grad_x / grad_norm
                momentum = cur_grad.mean(dim=0) + momentum * 0.3
                uaps = torch.clamp(uaps + self.uap_eps * torch.sign(momentum), -self.uap_eps, self.uap_eps)

                momentum = momentum.detach()
                uaps = uaps.detach()

                # weight average
                if weight_average:
                    """
                    for p_wa, p_model in zip(wa_model.parameters(), self.model.parameters()):
                        p_wa.data = p_wa.data * tau + p_model.data * (1 - tau)
                    """
                    for key, value in self.model.state_dict().items():
                        exp_avg[key] = (1 - tau) * value + tau * exp_avg[key]

                train_loss += loss.item() * labels.size(0)
                train_correct += (output.max(1)[1] == labels).sum().item()
                train_n += labels.size(0)

            scheduler.step()

            if weight_average:
                self.model.eval()
                wa_model.load_state_dict(exp_avg)
                wa_model.eval()
                # update bn
                """
                for module1, module2 in zip(wa_model.modules(), self.model.modules()):
                    if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
                        module1.running_mean = module2.running_mean
                        module1.running_var = module2.running_var
                        module1.num_batches_tracked = module2.num_batches_tracked
                """
            else:
                self.model.eval()

            total_training_time += time.time() - start_time
            self.logger.log('Training time: {:.2f}'.format(time.time() - start_time))
            self.logger.log('Training loss: {:.4f}'.format(train_loss / train_n))
            self.logger.log('Training accuracy: {:.4f}'.format(train_correct / train_n))
            train_acc_list.append(train_correct / train_n)

            if epoch < eval_start:
                self.logger.new_line()
                continue

            start_time = time.time()
            test_correct, fgsm_correct, pgd_correct, test_num = 0, 0, 0, 0

            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # clean accuracy
                if weight_average:
                    output = wa_model(images)
                else:
                    output = self.model(images)
                test_correct += (output.max(1)[1] == labels).sum().item()
                test_num += labels.size(0)

                # pgd accuracy
                adv_images = pgd_attacker(images, labels)
                if weight_average:
                    output = wa_model(adv_images)
                else:
                    output = self.model(adv_images)
                pgd_correct += (output.max(1)[1] == labels).sum().item()

                # fgsm accuracy
                fgsm_images = fgsm_attacker(images, labels)
                if weight_average:
                    output = wa_model(fgsm_images)
                else:
                    output = self.model(fgsm_images)
                fgsm_correct += (output.max(1)[1] == labels).sum().item()

            total_test_time += time.time() - start_time
            self.logger.log('Test time: {:.2f}'.format(time.time() - start_time))
            self.logger.log('Test accuracy: {:.4f}'.format(test_correct / test_num))
            self.logger.log('FGSM accuracy: {:.4f}'.format(fgsm_correct / test_num))
            self.logger.log('PGD accuracy: {:.4f}'.format(pgd_correct / test_num))

            test_acc_list.append(test_correct / test_num)
            pgd_acc_list.append(pgd_correct / test_num)
            fgsm_acc_list.append(fgsm_correct / test_num)

            if pgd_correct / test_num > best_pgd_acc or (
                    pgd_correct / test_num == best_pgd_acc and test_correct / test_num > best_test_acc):
                best_pgd_acc = pgd_correct / test_num
                best_test_acc = test_correct / test_num
                if weight_average:
                    torch.save(wa_model.state_dict(), os.path.join(self.output_dir, 'best.pth'))
                else:
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best.pth'))

            self.logger.new_line()

        self.logger.log("train_acc_list: \n" + str(train_acc_list))
        self.logger.new_line()
        self.logger.log("test_acc_list: \n" + str(test_acc_list))
        self.logger.new_line()
        self.logger.log("pgd_acc_list: \n" + str(pgd_acc_list))
        self.logger.new_line()
        self.logger.log("fgsm_acc_list: \n" + str(fgsm_acc_list))
        self.logger.new_line()
        self.logger.log("total_training_time: \n" + str(total_training_time))
        self.logger.new_line()
        self.logger.log("total_test_time: \n" + str(total_test_time))
        self.logger.new_line()

        if weight_average:
            torch.save(wa_model.state_dict(), os.path.join(self.output_dir, 'last.pth'))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'ori_last.pth'))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'last.pth'))
