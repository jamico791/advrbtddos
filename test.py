import torch
import options
import argparse
import numpy as np
from models import (get_torchvision_models, get_criterion)
from utils import get_dataloader


# Handle parser
parser = argparse.ArgumentParser(description='Test a trained ML model')

options.base_training_cfgs(parser)

args = vars(parser.parse_args())


def loss_batch(model, loss_fn, xb, yb, opt=None, metric=None):
    # Calculate loss
    preds = model(xb)
    print(preds)
    loss = loss_fn(preds, yb)
    print(loss)

    # Optimization step
    if opt is not None:
        # Calculate gradients
        loss.backward()
        # Adjust weights and biases
        opt.step()
        # Reset gradients to zero
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result

def evaluate(model, loss_fn, valid_dl, metric=None):
    with torch.no_grad():
        # Pass each batch through the model
        results = [loss_batch(model, loss_fn, xb, yb, metric=metric) for xb, yb in valid_dl]
        # Separate loss, counts, and metrics
        losses, nums, metrics = zip(*results)
        # Total size of the dataset
        total = np.sum(nums)
        # Calculate average loss
        print('Losses:', losses)
        print('Nums', nums)
        print('Metrics:', metrics)
        avg_loss = np.sum(np.multiply(losses, nums) / total)
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums) / total)

        return avg_loss, total, avg_metric

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds) 

def predict_image(image, model):
    xb = image.unsqueeze(0)
    yb = model(image)
    _,preds = torch.max(yb, dim=1)
    return preds[0].item()


if __name__ == '__main__':
    # Do stuff
    model = get_torchvision_models(**args)

    checkpoint = torch.load(args['resume_from'], map_location='cpu')

    model.load_state_dict(checkpoint['model'])

    test_dataloader = get_dataloader(dataset_type='test', **args)

    criterion = get_criterion(**args)

    values = evaluate(model, criterion, test_dataloader, metric=accuracy)
    print(values)
