import argparse
import random
from math import inf
import os
import glob
from typing import List, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklift.metrics import uplift_auc_score, qini_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

import wandb
from tqdm import tqdm

from model import UpliftWrapperForRNN, DirectUpliftLoss, RNNEmbedding, TCNEmbedding
from dataset import BackendDataset
from TCN import TCN

def set_all_seeds(seed, verbose=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_model(args):

    if args.model_type == "rnn":
        module = RNNEmbedding(
            in_features=6, 
            hidden_size=args.rnn_hidden_size, 
            out_features=args.rnn_out_features, 
            num_layers=args.rnn_num_layers, 
            dropout=args.rnn_dropout, 
            bidirectional=False
        )
        out_features = args.rnn_out_features
    elif args.model_type == "tcn":
        module = TCNEmbedding(
            input_size=6,
            output_size=args.tcn_out_features,
            num_channels=args.tcn_num_channels,
            kernel_size=2,
            dropout=args.tcn_dropout,
            num_layers=args.tcn_num_layers,
            num_ensembles=args.tcn_num_ensembles,
        )
        out_features = args.tcn_out_features
    model = UpliftWrapperForRNN(module, out_features)
    return model

def get_dataset(args):
    dataset = BackendDataset(
        json_path=os.path.join(args.data_path, "dataset.json"),
        data_path=args.data_path,
    )
    return dataset

def split_by_indices(dataset, train_ids: List[int]):

    eval_ids = list(set(range(len(dataset))) - set(train_ids))

    train_set = Subset(dataset, train_ids)
    eval_set = Subset(dataset, eval_ids)

    return train_set, eval_set

def split_dataset(args, dataset: Dataset):

    train_ratio = 1.0 - args.split_ratio

    if args.split_method == "random":
        if (train_ratio > 1.0) or (train_ratio < 0.0):
            raise ValueError(f"args.split_ratio must be between 0 and 1, but got {args.split_ratio}.")
        elif train_ratio == 0.0:
            return dataset, None
        elif train_ratio == 1.0:
            return None, dataset
        else:
            size = int(len(dataset) * train_ratio)
            train_ids = torch.randperm(len(dataset))[:size].numpy()
            return split_by_indices(dataset, train_ids)

    elif args.split_method == "game_id":
        train_ids = dataset.df.loc[dataset.df["gamer_id"] != args.game_id]
        return split_by_indices(dataset, train_ids)

    else:
        return dataset, None


def train(args, model, train_dl, loss_fn, optimizer, device):

    uplifts = []
    probs = []
    preds = []
    losses = []
    answers = []
    treatments = []

    for batch in tqdm(train_dl):
        optimizer.zero_grad()

        X = batch[0].to(device)
        t = batch[1].to(device)
        y = batch[2].to(device)

        out = model(X, t)
        loss = loss_fn(out, t, y)

        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        probs.extend(out["pred"].detach().cpu().tolist())
        uplifts.extend(out["uplift"].detach().cpu().tolist())

        y_pred = np.where(out["pred"].detach().cpu().numpy() > args.cutoff, 1, 0)
        preds.extend(y_pred.tolist())
        answers.extend(y.detach().cpu().tolist())
        treatments.extend(t.detach().cpu().tolist())

    answers = np.array(answers)
    preds = np.array(preds)

    acc = accuracy_score(answers, preds)
    f1 = f1_score(answers, preds)
    auc = roc_auc_score(answers, probs)
    auuc = uplift_auc_score(answers, uplifts, treatments)
    qini = qini_auc_score(answers, uplifts, treatments)

    return {
        "loss": np.mean(losses),
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "auuc": auuc,
        "qini": qini,
        "probs": probs,
        "uplifts": uplifts,
    }


def evaluate(args, model, eval_dl, loss_fn, device):

    uplifts = []
    probs = []
    preds = []
    losses = []
    answers = []
    treatments = []


    with torch.no_grad():
        for batch in tqdm(eval_dl):

            X = batch[0].to(device)
            t = batch[1].to(device)
            y = batch[2].to(device)

            out = model(X, t)
            loss = loss_fn(out, t, y)
            losses.append(loss.item())

            probs.extend(out["pred"].detach().cpu().tolist())
            uplifts.extend(out["uplift"].detach().cpu().tolist())

            y_pred = np.where(out["pred"].detach().cpu().numpy() > args.cutoff, 1, 0)
            preds.extend(y_pred.tolist())
            answers.extend(y.detach().cpu().tolist())
            treatments.extend(t.detach().cpu().tolist())

    answers = np.array(answers)
    preds = np.array(preds)

    acc = accuracy_score(answers, preds)
    f1 = f1_score(answers, preds)
    auc = roc_auc_score(answers, probs)
    auuc = uplift_auc_score(answers, uplifts, treatments)
    qini = qini_auc_score(answers, uplifts, treatments)

    return {
        "loss": np.mean(losses),
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "auuc": auuc,
        "qini": qini,
        "probs": probs,
        "uplifts": uplifts,
    }

def get_weighted_sampler(targets: Iterable[int]):
    target = torch.from_numpy(targets.unique())
    target = target[torch.randperm(len(target))]

    y_cnt = targets.value_counts().tolist()
    y_cnt = [c+1000 for c in y_cnt] # reduce the oversampling
    weights_by_class = 1. / torch.tensor(y_cnt, dtype=torch.float)
    weights_by_class = weights_by_class / sum(weights_by_class) # normalize

    all_weights = weights_by_class[targets]
    
    sampler = WeightedRandomSampler(
        weights=all_weights,
        num_samples=len(all_weights),
        replacement=True,
    )
    return sampler


def main(args):

    if args.seed:
        set_all_seeds(args.seed)

    model = get_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    dataset = get_dataset(args)
    train_set, eval_set = split_dataset(args, dataset)

    if args.use_weighted_sampler:
        y_train = dataset.df["y"][train_set.indices].reset_index(drop=True)
        sampler = get_weighted_sampler(y_train)
        train_dl = DataLoader(train_set, args.per_device_train_batch_size, sampler=sampler) if train_set is not None else None
    else:
        train_dl = DataLoader(train_set, args.per_device_train_batch_size, shuffle=True) if train_set is not None else None

    
    eval_dl = DataLoader(eval_set, args.per_device_eval_batch_size, shuffle=False) if eval_set is not None else None
    if args.do_eval and eval_dl is None:
        # if no eval data, then use train data for evaluation.
        eval_dl = train_dl
    print(f"Dataset properly loaded with length {len(dataset)}")

    loss_fn = DirectUpliftLoss(propensity_score=args.propensity_score, alpha=args.alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    saved_metrics = {}

    model.to(device)

    for epoch in range(args.num_epochs):
        print(f"Starting {epoch} epoch")

        if args.do_train and train_dl is not None:
            model.train()
            metrics = train(args, model, train_dl, loss_fn, optimizer, device)

            metrics["epoch"] = epoch
            metrics["probs"] = wandb.Histogram(np_histogram=np.histogram(np.array(metrics["probs"])))
            metrics["uplifts"] = wandb.Histogram(np_histogram=np.histogram(np.array(metrics["uplifts"])))
            wandb.log({"train/"+k: v for k, v in metrics.items()})
        

        if args.do_eval and eval_dl is not None:
            model.eval()
            metrics = evaluate(args, model, eval_dl, loss_fn, device)

            current_metric = metrics[args.best_metric]
            current_metric = current_metric if args.higher_the_better else -current_metric

            if len(saved_metrics.keys()) < 5 or current_metric > max(saved_metrics.keys()):
                if not os.path.isdir(args.save_path):
                    os.makedirs(args.save_path)
                saved_metrics[current_metric] = epoch
                print(f"Saving the best model at epoch {epoch}")
                torch.save(model.state_dict(), os.path.join(args.save_path, f"best_model_epoch{epoch}.pt"))
                if len(saved_metrics) > args.max_saved_models:
                    min_metric = min(saved_metrics.keys())
                    min_epoch = saved_metrics[min_metric]
                    print(f"Exceeds the max saved models. Remove the worst one at epoch {min_epoch}")
                    os.remove(os.path.join(args.save_path, f"best_model_epoch{min_epoch}.pt"))
                    del saved_metrics[min_metric]

            metrics["epoch"] = epoch
            metrics["probs"] = wandb.Histogram(np_histogram=np.histogram(np.array(metrics["probs"])))
            metrics["uplifts"] = wandb.Histogram(np_histogram=np.histogram(np.array(metrics["uplifts"])))
            wandb.log({"eval/"+k: v for k, v in metrics.items()})

    
    if args.do_predict:
        if saved_metrics:
            # if do_eval == True so that evaluation metrics exist
            best_metric = max(saved_metrics.keys())
            best_epoch = saved_metrics[best_metric]
            model_file = os.path.join(args.save_path, f"best_model_epoch{best_epoch}.pt")
            model.load_state_dict(torch.load(model_file))
        else:
            # otherwise, assuming the model state dict with the 
            model_files = glob.glob(os.path.join(args.save_path, f"best_model*"))
            model_file  = model_files[-1]
            model.load_state_dict(torch.load(model_file))

        model.eval()
        metrics = evaluate(args, model, eval_dl, loss_fn, device)

        file_names = dataset.df.file_name[eval_set.indices]
        prediction = pd.DataFrame({"file_name": file_names, "prob": metrics["probs"], "uplift": metrics["uplifts"]})
        prediction.to_csv("prediction.json")

        metrics["epoch"] = epoch
        metrics["probs"] = wandb.Histogram(np_histogram=np.histogram(np.array(metrics["probs"])))
        metrics["uplifts"] = wandb.Histogram(np_histogram=np.histogram(np.array(metrics["uplifts"])))
        wandb.log({"pred/"+k: v for k, v in metrics.items()})

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the model')

    parser.add_argument('--seed', type=int)

    parser.add_argument('--save_path', type=str, default="./saved", help="trained weight save path (deafult: ./saved)")
    parser.add_argument('--data_path', type=str, default="./dataset", help="dataset path (deafult: ./dataset)")
    parser.add_argument('--split_method', type=str, default="random", choices=["random", "game_id", "none"])
    parser.add_argument('--eval_game_id', type=str)
    parser.add_argument('--split_ratio', type=float, default=0.2)
    parser.add_argument('--use_weighted_sampler', action='store_true')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_predict', action='store_true')

    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning rate (default: 1e-3)")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="weight decay for AdamW (default: 0.01)")
    parser.add_argument('--propensity_score', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--per_device_train_batch_size', type=int, default=64)
    parser.add_argument('--per_device_eval_batch_size',  type=int, default=128)

    parser.add_argument('--model_type', type=str, choices=["rnn", "tcn"], default="rnn")

    parser.add_argument('--rnn_hidden_size',  type=int, default=16)
    parser.add_argument('--rnn_out_features',  type=int, default=16)
    parser.add_argument('--rnn_num_layers',  type=int, default=1)
    parser.add_argument('--rnn_dropout',  type=float, default=0.2)

    parser.add_argument('--tcn_num_channels', type=int, nargs='+', default=[16])
    parser.add_argument('--tcn_out_features', type=int, default=16)
    parser.add_argument('--tcn_dropout', type=float, default=0.2)
    parser.add_argument('--tcn_num_layers', type=int, default=1)
    parser.add_argument('--tcn_num_ensembles', type=int, default=1)

    parser.add_argument('--cutoff', type=float, default=0.5)

    parser.add_argument('--max_saved_models', type=int, default=5)
    parser.add_argument('--best_metric', type=str, default="loss")
    parser.add_argument('--higher_the_better', action='store_true')
    
    parser.add_argument('--no_cuda', action='store_true')

    args = parser.parse_args()

    wandb.init(entity="uplift-modeling", project="uplift-rnn", name=f"epochs_{args.num_epochs}_lr_{args.learning_rate}", config=args)

    main(args)
