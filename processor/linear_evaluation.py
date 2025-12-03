import sys
import argparse
import yaml
import math
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
import time

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class LE_Processor(Processor):
    """
        Processor for Linear Evaluation.
    """
    trainable_layers = ['linear1.0.linear.weight',
                    'linear1.0.linear.bias',
                    'linear1.0.bn.weight',
                    'linear1.0.bn.bias',
                    'linear2.0.linear.weight',
                    'linear2.0.linear.bias',
                    'linear2.0.bn.weight',
                    'linear2.0.bn.bias',
                    'linear3.weight',
                    'linear3.bias']
    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)

        # Optional fine-tuning: unfreeze backbone if requested
        unfreeze_backbone = getattr(self.arg, 'unfreeze_backbone', False)
        if not unfreeze_backbone:
            # Linear evaluation (default): freeze all but FC layers
            for name, param in self.model.encoder_q.named_parameters():
                if name not in self.trainable_layers:
                    param.requires_grad = False # freeze
            self.num_grad_layers = 10
            if hasattr(self.model, 'encoder_q_motion'):
                for name, param in self.model.encoder_q_motion.named_parameters():
                    if name not in self.trainable_layers:
                        param.requires_grad = False
                self.num_grad_layers += 10
            if hasattr(self.model, 'encoder_q_bone'):
                for name, param in self.model.encoder_q_bone.named_parameters():
                    if name not in self.trainable_layers:
                        param.requires_grad = False
                self.num_grad_layers += 10
        else:
            # Fine-tuning: leave all layers trainable
            for param in self.model.parameters():
                param.requires_grad = True
            # Disable strict grad layer count check later
            self.num_grad_layers = None

        self.loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        if self.num_grad_layers is not None:
            assert len(parameters) == self.num_grad_layers
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                parameters,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                parameters,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch'] > np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def show_best(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
        accuracy = round(accuracy, 5)
        self.current_result = accuracy
        if self.best_result <= accuracy:
            self.best_result = accuracy
        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))

    def train(self, epoch):
        self.model.eval()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label in loader:
            self.global_step += 1
            # get data
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            
            if self.arg.stream not in ['joint', 'motion', 'bone', 'all']:
                raise ValueError

            # forward
            output = self.model(None, data, stream=self.arg.stream)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent NaN errors
            if hasattr(self.arg, 'grad_clip') and self.arg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_clip)
            
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.show_epoch_info()

    def test(self, epoch):
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        total_infer_time = 0
        total_sample = 0

        for data, label in loader:
            # get data
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            
            if self.arg.stream not in ['joint', 'motion', 'bone', 'all']:
                raise ValueError

            # inference
            batch_size = label.shape[0]
            with torch.no_grad():
                start_time = time.time()
                output = self.model(None, data, stream=self.arg.stream)
                infer_time = time.time() - start_time
            total_infer_time += infer_time
            total_sample += batch_size
            result_frag.append(output.data.cpu().numpy())

            # get loss
            loss = self.loss(output, label)
            loss_value.append(loss.item())
            label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)

        self.eval_info['eval_mean_loss']= np.mean(loss_value)
        avg_infer_time = total_infer_time / float(total_sample)
        self.eval_info['avg_infer_time_per_seq'] = avg_infer_time
        self.io.print_log('\tAvg inference time per sequence: {:.6f} s'.format(avg_infer_time))
        self.show_eval_info()

        # show top-k accuracy 
        for k in self.arg.show_topk:
            self.show_topk(k)
        self.show_best(1)

        self.eval_log_writer(epoch)

    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--stream', type=str, default='joint', help='the stream of input')
        parser.add_argument('--mining_epoch', type=int, default=1e6,
                            help='the starting epoch of nearest neighbor mining')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in nearest neighbor mining')
        parser.add_argument('--unfreeze_backbone', type=str2bool, default=False,
                            help='If true, fine-tune the whole backbone instead of linear eval')
        parser.add_argument('--grad_clip', type=float, default=0.0,
                            help='Gradient clipping value (0.0 means no clipping)')

        return parser
