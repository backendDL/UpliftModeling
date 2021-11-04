import numpy as np

#region evaluation utils
def f1(pred, target):
    # expect the type of inputs are torch.Tensor
    assert pred.shape == target.shape
    TP = (pred * target).sum().float()
    TN = ((1 - pred) * (1 - target)).sum().float()
    FP = (pred * (1 - target)).sum().float()
    FN = ((1 - pred) * target).sum().float()

    precision = TP / (TP + FP + 1e-5)
    recall = TP / (TP + FN + 1e-5)

    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    return f1
#endregion

#region string utils
def n_digit(n, x):
    x = str(x)
    return '0' * (n - len(x)) + x

def six_digit(x):
    return n_digit(6, x)

def four_digit(x):
    return n_digit(4, x)

def two_digit(x):
    return n_digit(2, x)
#endregion