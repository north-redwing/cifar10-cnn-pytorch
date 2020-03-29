import os
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt


def get_loader(batch_size, num_workers, path_to_dataset):
    """Get a train loader and evaluation loader for CIFAR10.

    Args:
        batch_size (int): how many data to use for training at once
        num_workers (int): how many subprocesses to use for date loading.
        path_to_dataset (str): where to save dataset

    Returns:
        train_loader (iters): train loader.
        eval_load (iters): evaluation loader.
        classes (tuple): classes the CIFAR10 dataset has.

    """
    os.makedirs(path_to_dataset, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = torchvision.datasets.CIFAR10(
        root=path_to_dataset,
        train=True,
        download=True,
        transform=transform
    )
    eval_data = torchvision.datasets.CIFAR10(
        root=path_to_dataset,
        train=False,
        download=True,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    classes = (
        'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
        'truck')
    return train_loader, eval_loader, classes


def images2probs(model, images):
    output = model(images)
    _, predicted = torch.max(output, 1)
    preds = np.squeeze(predicted.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in
                   zip(preds, output)]


def plot_classes_preds(model, images, labels, classes):
    preds, probs = images2probs(model, images)
    fig = plt.figure(figsize=(12, 48))
    for idx in range(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        color = 'green' if preds[idx] == labels[idx].item() else 'red'
        title_str = '%s, %.1f \n(label: %s)' % (
            classes[preds[idx]], probs[idx] * 100.0, classes[labels[idx]])
        matplotlib_imshow(images[idx])
        ax.set_title(title_str, color=color)
    return fig


def get_args():
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--batch_size', default=4, type=int, help='')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--n_epoch', default=2, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--no_hyperdash', default=False, action='store_true')
    parser.add_argument('--checkpoint_dir_name', default=None, type=str)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.device = torch.device('cpu')
    if not args.no_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda:0')
        if torch.cuda.device_count() > 1:
            gpu_ids = [id for id in range(len(torch.cuda.device_count()))]
            args.device = torch.device(f'cuda:{gpu_ids[0]}')
    print('####################')
    print('device ===> ', args.device)
    print('####################')
    return args


def matplotlib_imshow(img, one_channnel=False):
    if one_channnel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.cpu().numpy()
    if one_channnel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
