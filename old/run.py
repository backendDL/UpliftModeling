from models.model import Model
from data_loader import Dataset
from tqdm import tqdm
from utils import *

import os
import argparse
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

LABEL_NAME = 'D+7 Return Retention'

def test(model, dataset, batch_size=512, tb_writer=None, step=None):
    assert not ((tb_writer is None) ^ (step is None)), (tb_writer, step)
    y_preds = []
    with torch.no_grad():
        model.eval()
        for X_test, y_target in dataset.test_mini_batch(batch_size * 2):
            X_test = X_test.cuda(); y_target = y_target.cuda()
            y_pred = model(X_test)
            y_preds.append(y_pred)

    y_preds = torch.cat(y_preds, dim=0).cpu()
    y_pred_bin = (y_preds > 0.5).float()
    y_target = dataset.test_y[:len(y_pred_bin)].cpu()
    test_acc = torch.eq(y_pred_bin, y_target).sum() / y_pred_bin.shape[0]
    test_f1 = f1(y_pred_bin, y_target)

    if tb_writer:
        tb_writer.add_scalar(f'{LABEL_NAME}/test_acc', test_acc, step)
        tb_writer.add_scalar(f'{LABEL_NAME}/test_f1', test_f1, step)

    print(f'\t[TEST] {LABEL_NAME} ACC {test_acc:.3f}')
    print(f'\t[TEST] {LABEL_NAME} F1  {test_f1:.3f}')

    filters = [
        '>.95', '>.90', '>.75', '>.50',
        '<.5', '<.25', '<.1', '<.05',
    ]

    for f in filters:
        assert f[0] in '<>', f

        cf = (lambda a, b: a > b) if f[0] == '>' else (lambda a, b: a < b)
        t = float(f[1:])
        bins = cf(y_preds, t)
        count = bins.sum()
        _y = y_target[bins]
        precision = _y.sum() / len(_y) # precision = TP / (TP + FP)
    
        if tb_writer:
            tb_writer.add_scalar(f'{LABEL_NAME}/{f} precision', precision, step)

        print(f'\t[TEST] {f} LABELED {count} PRECISION {precision:.3f}')

def train(model, dataset, lr=1e-1, batch_size=256, epochs=100):
    # tensorboard
    board_writer = SummaryWriter()
    
    # scientific notation in the config.yaml file can be recognized as string type
    if type(lr) is str: learning_rate = float(lr)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    loss_func = nn.BCELoss(reduction='mean')
    
    step = 1
    while step <= epochs:
        
        train_losses = []
        y_pred_bin = []
        with tqdm(total=len(dataset.train_X) // batch_size, leave=False) as progress: 
            #region train
            with torch.set_grad_enabled(True):
                model.train()
                for X_train, y_target in dataset.train_mini_batch(batch_size):
                    X_train = X_train.cuda(); y_target = y_target.cuda()

                    optimizer.zero_grad()
                    y_pred = model(X_train)
                    y_pred_bin.append((y_pred > 0.5).cpu())

                    loss = loss_func(y_pred, y_target)
                    train_losses.append(loss.item())
                    
                    loss.backward()
                    optimizer.step()

                    progress.update()
                    progress.set_description(f'epoch {four_digit(step)}')
            
            y_pred_bin = torch.cat(y_pred_bin).float()
            y_target = dataset.train_y[:y_pred_bin.shape[0]]

            train_loss = np.mean(train_losses)
            train_acc = torch.eq(y_pred_bin, y_target).sum() / y_pred_bin.shape[0]
            train_f1 = f1(y_pred_bin, y_target)
            #endregion
        
            #region validation
            val_losses = []
            y_pred_bin = []
            with torch.no_grad():
                model.eval()
                for X_val, y_target in dataset.val_mini_batch(batch_size * 2):
                    X_val = X_val.cuda(); y_target = y_target.cuda()
                    y_pred = model(X_val)
                    y_pred_bin.append((y_pred > 0.5).cpu())

                    loss = loss_func(y_pred, y_target)
                    val_losses.append(loss.item())
            
            y_pred_bin = torch.cat(y_pred_bin).float()
            y_target = dataset.val_y[:y_pred_bin.shape[0]]

            val_loss = np.mean(val_losses)
            val_acc = torch.eq(y_pred_bin, y_target).sum() / y_pred_bin.shape[0]
            val_f1 = f1(y_pred_bin, y_target)
            #endregion
        
        #region description
        print('epoch:', step)
        print(f'\t[train] loss: {train_loss},')
        print(f'\t[train] {LABEL_NAME} ACC {train_acc:.3f}')
        print(f'\t[train] {LABEL_NAME} F1  {train_f1:.3f}')
        print(f'\t[ val ] loss: {val_loss},')
        print(f'\t[ val ] {LABEL_NAME} ACC {val_acc:.3f}')
        print(f'\t[ val ] {LABEL_NAME} F1  {val_f1:.3f}')
        print('-' * 100)
        #endregion

        #region logging(tensorboard)
        board_writer.add_scalar('loss/train', train_loss, step)
        board_writer.add_scalar('loss/val', val_loss, step)

        board_writer.add_scalar(f'{LABEL_NAME}/train/acc', train_acc, step)
        board_writer.add_scalar(f'{LABEL_NAME}/train/f1', train_f1, step)
        board_writer.add_scalar(f'{LABEL_NAME}/val/acc', val_acc, step)
        board_writer.add_scalar(f'{LABEL_NAME}/val/f1', val_f1, step)
        #endregion

        # make checkpoint
        torch.save({k:v.cpu() for k, v in model.state_dict().items()}, f'ckpts/{six_digit(step)}.pt')

        # lr decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9

        step += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='./config.yaml', required=False)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    dataset = Dataset(cfg['Dataset'])
    dataset.prepare_train_data() # 오래 걸림.

    model = Model(**cfg['model']).cuda()

    os.makedirs('ckpts', exist_ok=True)

    train(model, dataset, **cfg['train'])
    # do test with test() function