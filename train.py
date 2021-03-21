import os, torch, random, pickle, time
from argparse import ArgumentParser
import numpy as np
import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel.scatter_gather import gather
import load_data as ld
import dataset
import transforms
from parallel import DataParallelModel, DataParallelCriterion
from utils import SalEval, AverageMeterSmooth, Logger, plot_training_process


parser = ArgumentParser()
parser.add_argument('--data_dir', default='./Data', type=str, help='data directory')
parser.add_argument('--width', default=336, type=int, help='width of RGB image')
parser.add_argument('--height', default=336, type=int, help='height of RGB image')
parser.add_argument('--max_epochs', default=50, type=int, help='max number of epochs')
parser.add_argument('--num_workers', default=10, type=int, help='No. of parallel threads')
parser.add_argument('--batch_size', default=20, type=int, help='batch size')
parser.add_argument('--lr', default=5e-4, type=float, help='initial learning rate')
parser.add_argument('--warmup', default=0, type=int, help='lr warming up epoches')
parser.add_argument('--scheduler', default='poly', type=str, choices=['step', 'poly', 'cos'],
                    help='Lr scheduler (valid: step, poly, cos)')
parser.add_argument('--gamma', default=0.1, type=float, help='gamma for multi-step lr decay')
parser.add_argument('--milestones', default='[30, 60, 90]', type=str, help='milestones for multi-step lr decay')
parser.add_argument('--print_freq', default=50, type=int, help='frequency of printing training info')
parser.add_argument('--savedir', default='./Results', type=str, help='Directory to save the results')
parser.add_argument('--resume', default=None, type=str, help='use this checkpoint to continue training')
parser.add_argument('--cached_data_file', default='duts_train.p', type=str, help='Cached file name')
parser.add_argument('--pretrained', default=None, type=str, help='path for the ImageNet pretrained backbone model')
parser.add_argument('--seed', default=666, type=int, help='Random Seed')
parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to run on the GPU')
parser.add_argument('--model', default='Models.SAMNet', type=str, help='which model to test')

args = parser.parse_args()

exec('from {} import FastSal as net'.format(args.model))

cudnn.benchmark = False
cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)


def adjust_lr(optimizer, epoch):
    if epoch < args.warmup:
        lr = args.lr * (epoch + 1) / args.warmup
    else:
        if args.scheduler == 'cos':
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / args.max_epochs))
        elif args.scheduler == 'poly':
            lr = args.lr * (1 - epoch * 1.0 / args.max_epochs) ** 0.9
        elif args.scheduler == 'step':
            lr = args.lr
            for milestone in eval(args.milestones):
                if epoch >= milestone:
                    lr *= args.gamma
        else:
            raise ValueError('Unknown lr mode {}'.format(args.scheduler))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, inputs, target):
        if isinstance(target, tuple):
            target = target[0]
        target = target.float()
        loss = F.binary_cross_entropy(inputs[:, 0, :, :], target)
        for i in range(1, inputs.shape[1]):
            loss += 0.4 * F.binary_cross_entropy(inputs[:, i, :, :], target)

        return loss


@torch.no_grad()
def val(val_loader, epoch):
    # switch to evaluation mode
    model.eval()
    salEvalVal = SalEval()

    total_batches = len(val_loader)
    for iter, (input, target) in enumerate(val_loader):
        if args.gpu:
            input = input.cuda()
            target = target.cuda()
        input = torch.autograd.Variable(input)
        target = torch.autograd.Variable(target)

        start_time = time.time()
        # run the mdoel
        output = model(input)

        torch.cuda.synchronize()
        val_times.update(time.time() - start_time)

        loss = criterion(output, target)
        val_losses.update(loss.item())

        # compute the confusion matrix
        if args.gpu and torch.cuda.device_count() > 1:
            output = gather(output, 0, dim=0)
        salEvalVal.addBatch(output[:, 0, :, :], target)

        if iter % args.print_freq == 0:
            logger.info('Epoch [%d/%d] Iter [%d/%d] Time: %.3f loss: %.3f (avg: %.3f)' %
                        (epoch, args.max_epochs, iter, total_batches, val_times.avg,
                        val_losses.val, val_losses.avg))

    F_beta, MAE = salEvalVal.getMetric()
    record['val']['F_beta'].append(F_beta)
    record['val']['MAE'].append(MAE)

    return F_beta, MAE


def train(train_loader, epoch, cur_iter=0, verbose=True):
    # switch to train mode
    model.train()
    if verbose:
        salEvalTrain = SalEval()

    total_batches = len(train_loader)
    scale = cur_iter // total_batches
    end = time.time()
    for iter, (input, target) in enumerate(train_loader):
        if args.gpu == True:
            input = input.cuda()
            target = target.cuda()
        input = torch.autograd.Variable(input)
        target = torch.autograd.Variable(target)

        start_time = time.time()

        # run the mdoel
        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses[scale].update(loss.item())
        train_batch_times[scale].update(time.time() - start_time)
        train_data_times[scale].update(start_time - end)
        record[scale].append(train_losses[scale].avg)

        if verbose:
            # compute the confusion matrix
            if args.gpu and torch.cuda.device_count() > 1:
                output = gather(output, 0, dim=0)
            salEvalTrain.addBatch(output[:, 0, :, :], target)

        if iter % args.print_freq == 0:
            logger.info('Epoch [%d/%d] Iter [%d/%d] Batch time: %.3f Data time: %.3f ' \
                        'loss: %.3f (avg: %.3f) lr: %.1e' % \
                        (epoch, args.max_epochs, iter + cur_iter, total_batches + cur_iter, \
                         train_batch_times[scale].avg, train_data_times[scale].avg, \
                         train_losses[scale].val, train_losses[scale].avg, lr))
        end = time.time()

    if verbose:
        F_beta, MAE = salEvalTrain.getMetric()
        record['train']['F_beta'].append(F_beta)
        record['train']['MAE'].append(MAE)

        return F_beta, MAE


# create the directory if not exist
if not os.path.exists(args.savedir):
    os.mkdir(args.savedir)

log_name = 'log_' + datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S') + '.txt'
logger = Logger(os.path.join(args.savedir, log_name))
logger.info('Called with args:')
for (key, value) in vars(args).items():
    logger.info('{0:16} | {1}'.format(key, value))

# check if processed data file exists or not
if not os.path.isfile(args.cached_data_file):
    data_loader = ld.LoadData(args.data_dir, 'DUTS-TR', args.cached_data_file)
    data = data_loader.process()
    if data is None:
        logger.info('Error while pickling data. Please check.')
        exit(-1)
else:
    data = pickle.load(open(args.cached_data_file, 'rb'))

# ImageNet statistics
mean = np.array([0.485 * 255., 0.456 * 255., 0.406 * 255.], dtype=np.float32)
std = np.array([0.229 * 255., 0.224 * 255., 0.225 * 255.], dtype=np.float32)

# load the model
model = net(pretrained=args.pretrained)
if args.gpu and torch.cuda.device_count() > 1:
    model = DataParallelModel(model)
if args.gpu:
    model = model.cuda()

logger.info('Model Architecture:\n' + str(model))
total_paramters = sum([np.prod(p.size()) for p in model.parameters()])
logger.info('Total network parameters: ' + str(total_paramters))

logger.info('Data statistics:')
logger.info('mean: [%.5f, %.5f, %.5f], std: [%.5f, %.5f, %.5f]' % (*data['mean'], *data['std']))

criterion = CrossEntropyLoss()
if args.gpu and torch.cuda.device_count() > 1 :
    criterion = DataParallelCriterion(criterion)
if args.gpu:
    criterion = criterion.cuda()

train_losses = [AverageMeterSmooth() for _ in range(5)]
train_batch_times = [AverageMeterSmooth() for _ in range(5)]
train_data_times = [AverageMeterSmooth() for _ in range(5)]
val_losses = AverageMeterSmooth()
val_times = AverageMeterSmooth()

record = {
        0: [], 1: [], 2: [], 3: [], 4: [], 'lr': [],
        'val': {'F_beta': [], 'MAE': []},
        'train': {'F_beta': [], 'MAE': []}
        }
bests = {'F_beta_tr': 0., 'F_beta_val': 0., 'MAE_tr': 1., 'MAE_val': 1.}

# compose the data with transforms
trainTransform_main = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Normalize(mean=data['mean'], std=data['std']),
        transforms.Scale(args.width, args.height),
        transforms.RandomCropResize(int(7./224.*args.width)),
        transforms.RandomFlip(),
        transforms.ToTensor()
        ])
trainTransform_scale1 = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Normalize(mean=data['mean'], std=data['std']),
        transforms.Scale(int(args.width*2), int(args.height*2)),
        transforms.RandomCropResize(int(28./224.*args.width)),
        transforms.RandomFlip(),
        transforms.ToTensor()
        ])
trainTransform_scale2 = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Normalize(mean=data['mean'], std=data['std']),
        transforms.Scale(int(args.width*1.5), int(args.height*1.5)),
        transforms.RandomCropResize(int(22./224.*args.width)),
        transforms.RandomFlip(),
        transforms.ToTensor()
        ])
trainTransform_scale3 = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Normalize(mean=data['mean'], std=data['std']),
        transforms.Scale(int(args.width*1.25), int(args.height*1.25)),
        transforms.RandomCropResize(int(22./224.*args.width)),
        transforms.RandomFlip(),
        transforms.ToTensor()
        ])
trainTransform_scale4 = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Normalize(mean=data['mean'], std=data['std']),
        transforms.Scale(int(args.width*0.75), int(args.height*0.75)),
        transforms.RandomCropResize(int(7./224.*args.width)),
        transforms.RandomFlip(),
        transforms.ToTensor()
        ])
valTransform = transforms.Compose([
        transforms.Normalize(mean=data['mean'], std=data['std']),
        transforms.Scale(args.width, args.height),
        transforms.ToTensor()
        ])

# since we training from scratch, we create data loaders at different scales
# so that we can generate more augmented data and prevent the network from overfitting
train_set = dataset.Dataset(args.data_dir, 'DUTS-TR', transform=None)
val_set = dataset.Dataset(args.data_dir, 'ECSSD', transform=valTransform)
train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
        )
val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
        )
max_batches = len(train_loader) * 5

optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, eval(args.milestones), args.gamma)
logger.info('Optimizer Info:\n' + str(optimizer))

start_epoch = 0
if args.resume is not None:
    if os.path.isfile(args.resume):
        logger.info('=> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
        logger.info('=> loaded checkpoint {} (epoch {})'.format(args.resume, checkpoint['epoch']))
    else:
        logger.info('=> no checkpoint found at {}'.format(args.resume))

for epoch in range(start_epoch, args.max_epochs):
    # train for one epoch
    lr = adjust_lr(optimizer, epoch)
    record['lr'].append(lr)
    length = len(train_loader)

    train_set.transform = trainTransform_scale1
    train(train_loader, epoch, 0, verbose=False)
    train_set.transform = trainTransform_scale2
    train(train_loader, epoch, length, verbose=False)
    train_set.transform = trainTransform_scale3
    train(train_loader, epoch, length * 2, verbose=False)
    train_set.transform = trainTransform_scale4
    train(train_loader, epoch, length * 3, verbose=False)
    train_set.transform = trainTransform_main
    F_beta_tr, MAE_tr = train(train_loader, epoch, length * 4, verbose=True)

    # evaluate on validation set
    F_beta_val, MAE_val = val(val_loader, epoch)
    if F_beta_tr > bests['F_beta_tr']: bests['F_beta_tr'] = F_beta_tr
    if MAE_tr < bests['MAE_tr']: bests['MAE_tr'] = MAE_tr
    if F_beta_val > bests['F_beta_val']: bests['F_beta_val'] = F_beta_val
    if MAE_val < bests['MAE_val']: bests['MAE_val'] = MAE_val

    scheduler.step()
    torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_F_beta': bests['F_beta_val'],
            'best_MAE': bests['MAE_val']
            }, os.path.join(args.savedir, 'checkpoint.pth'))

    # save the model also
    model_file_name = os.path.join(args.savedir, 'model_epoch' + str(epoch + 1) + '.pth')
    torch.save(model.state_dict(), model_file_name)

    logger.info('Epoch %d: F_beta (tr) %.4f (Best: %.4f) MAE (tr) %.4f (Best: %.4f) ' \
                'F_beta (val) %.4f (Best: %.4f) MAE (val) %.4f (Best: %.4f)' % \
                (epoch, F_beta_tr, bests['F_beta_tr'], MAE_tr, bests['MAE_tr'], \
                F_beta_val, bests['F_beta_val'], MAE_val, bests['MAE_val']))
    plot_training_process(record, args.savedir, bests)

logger.close()
