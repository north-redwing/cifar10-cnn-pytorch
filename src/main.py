import os
import datetime
import csv
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from pathlib import Path
from time import time
from net import CNN
from utils import get_loader, plot_classes_preds, get_args, matplotlib_imshow


def train(train_loader, model, criterion, optimizer, device, writer, epoch,
          classes):
    correct = 0
    total = 0
    running_loss = 0.0
    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = correct / total
    writer.add_figure('prediction vs. actuals', plot_classes_preds(model,
                                                                   inputs,
                                                                   labels,
                                                                   classes),
                      global_step=epoch * len(train_loader))
    loss = running_loss / len(train_loader)
    return loss, acc


def eval(eval_loader, model, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in eval_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    loss = running_loss / len(eval_loader)
    return loss, acc


def main():
    start_time = time()
    args = get_args()
    if args.checkpoint_dir_name:
        dir_name = args.checkpoint_dir_name
    else:
        dir_name = datetime.datetime.now().strftime('%y%m%d%H%M%S')
    path_to_dir = Path(__file__).resolve().parents[1]
    path_to_dir = os.path.join(path_to_dir, *['log', dir_name])
    os.makedirs(path_to_dir, exist_ok=True)
    # tensorboard
    path_to_tensorboard = os.path.join(path_to_dir, 'tensorboard')
    os.makedirs(path_to_tensorboard, exist_ok=True)
    writer = SummaryWriter(path_to_tensorboard)
    # model saving
    os.makedirs(os.path.join(path_to_dir, 'model'), exist_ok=True)
    path_to_model = os.path.join(path_to_dir, *['model', 'model.tar'])
    # csv
    os.makedirs(os.path.join(path_to_dir, 'csv'), exist_ok=True)
    path_to_results_csv= os.path.join(path_to_dir, *['csv', 'results.csv'])
    path_to_args_csv= os.path.join(path_to_dir, *['csv', 'args.csv'])
    if not args.checkpoint_dir_name:
        with open(path_to_args_csv, 'a') as f:
            args_dict = vars(args)
            param_writer = csv.DictWriter(f, list(args_dict.keys()))
            param_writer.writeheader()
            param_writer.writerow(args_dict)
    

    # logging using hyperdash
    if not args.no_hyperdash:
        from hyperdash import Experiment
        exp = Experiment('Classification task on CIFAR10 dataset with CNN')
        for key in vars(args).keys():
            exec("args.%s = exp.param('%s', args.%s)" % (key, key, key))
    else:
        exp = None

    path_to_dataset = os.path.join(Path(__file__).resolve().parents[2], 'datasets')
    os.makedirs(path_to_dataset, exist_ok=True)
    train_loader, eval_loader, classes = get_loader(batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    path_to_dataset=path_to_dataset)

    # show some of the training images, for fun.
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid)
    writer.add_image('four_CIFAR10_images', img_grid)

    # define a network, loss function and optimizer
    model = CNN()
    writer.add_graph(model, images)
    model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    start_epoch = 0
    # resume training
    if args.checkpoint_dir_name:
        print('\nLoading the model...')
        checkpoint = torch.load(path_to_model)
        model.state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    summary(model, input_size=(3, 32, 32))
    model.to(args.device)

    # train the network
    print('\n--------------------')
    print('Start training and evaluating the CNN')
    for epoch in range(start_epoch, args.n_epoch):
        start_time_per_epoch = time()
        train_loss, train_acc = train(train_loader, model, criterion,
                                      optimizer, args.device, writer, epoch,
                                      classes)
        eval_loss, eval_acc = eval(eval_loader, model, criterion, args.device)
        elapsed_time_per_epoch = time() - start_time_per_epoch
        result_dict = {'epoch': epoch, 'train_loss': train_loss, 'eval_loss': eval_loss , 'train_acc': train_acc , 'eval_acc': eval_acc , 'elapsed time': elapsed_time_per_epoch}
        with open(path_to_results_csv, 'a') as f:
            result_writer = csv.DictWriter(f, list(result_dict.keys()))
            if epoch == 0: result_writer.writeheader()
            result_writer.writerow(result_dict)
        # checkpoint
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   path_to_model)
        if exp:
            exp.metric('train loss', train_loss)
            exp.metric('eval loss', eval_loss)
            exp.metric('train acc', train_acc)
            exp.metric('eval acc', eval_acc)
        else:
            print(result_dict)
        
        writer.add_scalar('loss/train_loss', train_loss,
                          epoch * len(train_loader))
        writer.add_scalar('loss/eval_loss', eval_loss,
                          epoch * len(eval_loader))
        writer.add_scalar('acc/train_acc', train_acc,
                          epoch * len(train_loader))
        writer.add_scalar('acc/eval_acc', eval_acc,
                          epoch * len(eval_loader))
    
    elapsed_time = time() - start_time
    print('\nFinished Training, elapsed time ===> %f' % elapsed_time)
    if exp:
        exp.end()
    writer.close()


if __name__ == '__main__':
    main()
