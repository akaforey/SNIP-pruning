import matplotlib.pyplot as plt
from torchvision import datasets
import numpy as np
import torch
from datasets import RamLoader
from pruning import Pruner
from models import LeNet5
# from models import vgg16
from network import NN

DEVICE = 'cuda'
# DEVICE = 'cpu'

# model = vgg16()
model = LeNet5()
nn = NN(model, DEVICE)

prune = Pruner(model, RamLoader['train'], DEVICE, silent=False)
prune.snip(0.8)

loop_operation = [prune.indicate]

nn.optim = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay=0.0005)
nn.scheduler = torch.optim.lr_scheduler.StepLR(nn.optim, step_size=40, gamma=0.1)

hist = nn.fit(RamLoader, torch.nn.CrossEntropyLoss(), s_report=5,
              v_report=5, max_epochs=140, silent=False, loop_operation=loop_operation)

