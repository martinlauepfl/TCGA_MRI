# transfer learning

import os
import math
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
import neptune

epochs = 1
batchsize = 8
learningrate = 0.001

model = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,  # model output channels (number of classes in your dataset)
)

# load trained model
state_dict = torch.load('./checkpoint/my_model_test_acc_1.00_test_dice_0.93_epoch_9.pth')
model.load_state_dict(state_dict)

run = neptune.init_run(
    project="zijingliu123/TCGA",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4YjA0NDIxMy02MTMwLTQwN2QtYTcxNy0wMDA0MWEwOGYwYTUifQ==",
)  # your credentials

run["parameters"] = {
    "batch_size": batchsize,
    "learning_rate": learningrate,
    "num_epochs": epochs
}

model.to('cuda')

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)


def Dice(inp, target, eps=1):
    input_flatten = inp.flatten()
    target_flatten = target.flatten()
    overlap = np.sum(input_flatten * target_flatten)
    return np.clip(((2. * overlap) / (np.sum(target_flatten) + np.sum(input_flatten) + eps)), 1e-4, 0.9999)

from tqdm import tqdm

def fit(epoch, model, trainloader, testloader):
    correct = 0
    total = 0
    running_loss = 0
    epoch_iou = []

    train_dice = []
    test_dice = []
    train_dice_sd = []
    test_dice_sd = []

    # 72-100 lines, You can comment out when testing the model
    # model.train()
    # for x, y in tqdm(testloader):
    #     x, y = x.to('cuda'), y.to('cuda')
    #     y_pred = model(x)
    #     loss = loss_fn(y_pred, y)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     with torch.no_grad():
    #         y_pred = torch.argmax(y_pred, dim=1)
    #         correct += (y_pred == y).sum().item()
    #         total += y.size(0)
    #         running_loss += loss.item()
    #
    #         intersection = torch.logical_and(y, y_pred)
    #         union = torch.logical_or(y, y_pred)
    #         batch_iou = torch.sum(intersection) / torch.sum(union)
    #         epoch_iou.append(batch_iou.item())
    #
    #         y_pred_dice = y_pred
    #         y_pred_dice = y_pred_dice.cpu().numpy()
    #         ys = y.cpu().numpy()
    #         dice = Dice(y_pred_dice, ys)
    #         train_dice.append(dice)
    #         dice_sd = np.std(train_dice)
    #         train_dice_sd.append(dice_sd)
    #
    # epoch_loss = running_loss / len(trainloader.dataset)
    # epoch_acc = correct / (total * 128 * 128)

    test_correct = 0
    test_total = 0
    test_running_loss = 0
    epoch_test_iou = []

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(testloader):
            x, y = x.to('cuda'), y.to('cuda')
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()

            intersection = torch.logical_and(y, y_pred)
            union = torch.logical_or(y, y_pred)
            batch_iou = torch.sum(intersection) / torch.sum(union)
            epoch_test_iou.append(batch_iou.item())

            y_pred_dice = y_pred
            y_pred_dice = y_pred_dice.cpu().numpy()
            ys = y.cpu().numpy()
            dice = Dice(y_pred_dice, ys)
            test_dice.append(dice)
            dice_sd = np.std(test_dice)
            test_dice_sd.append(dice_sd)

    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / (test_total * 128 * 128)

    # Log metrics to Neptune
    # run["train/loss"].log(round(epoch_loss, 3))
    # run["train/accuracy"].log(round(epoch_acc, 3))
    # run["train/iou"].log(round(np.mean(epoch_iou), 3))
    # run["train/dice"].log(round(np.mean(train_dice), 3))
    # run["train/dice_SD"].log(round(np.mean(train_dice_sd), 3))

    # Log metrics to Neptune
    run["test/loss"].log(round(epoch_test_loss, 3))
    run["test/accuracy"].log(round(epoch_test_acc, 3))
    run["test/iou"].log(round(np.mean(epoch_test_iou), 3))
    run["test/dice"].log(round(np.mean(test_dice), 3))
    run["test/dice_SD"].log(round(np.mean(test_dice_sd), 3))

    # Log epoch number
    run["epoch"].log(epoch)

    # model_name = 'my_model_acc_{:.2f}_iou_{:.2f}_epoch_{}.pth'.format(acc, iou, epoch)
    model_name = 'my_model_test_acc_{:.2f}_test_dice_{:.2f}_epoch_{}.pth'.format(round(epoch_test_acc, 3),round(np.mean(test_dice), 3), epoch)
    model_path = './checkpoint/' + model_name
    torch.save(model.state_dict(), model_path)

    print('epoch: ', epoch,
          # 'loss??? ', round(epoch_loss, 3),
          # 'accuracy:', round(epoch_acc, 3),
          # 'IOU:', round(np.mean(epoch_iou), 3),
          # 'Dice:', round(np.mean(train_dice), 3),
          # 'Dice_SD:', round(np.mean(train_dice_sd), 3),
          'test_loss??? ', round(epoch_test_loss, 3),
          'test_accuracy:', round(epoch_test_acc, 3),
          'test_iou:', round(np.mean(epoch_test_iou), 3),
          'testDice:', round(np.mean(test_dice), 3),
          'testDice_SD:', round(np.mean(test_dice_sd), 3)

          )

    # return epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc
    return epoch_test_loss, epoch_test_acc
