import torch, cv2, time, os
from PIL import Image
import os.path as osp
import numpy as np
import torch.nn.functional as F
from argparse import ArgumentParser
from utils import SalEval, Logger


parser = ArgumentParser()
parser.add_argument('--data_dir', default='./Data', type=str, help='data directory')
parser.add_argument('--file_list', default='ALL', type=str, help='dataset list',
                    choices=['ECSSD.txt', 'DUT-OMRON.txt', 'DUTS-TE.txt', 'HKU-IS.txt',
                             'SOD.txt', 'THUR15K.txt', 'ALL'])
parser.add_argument('--width', default=336, type=int, help='width of RGB image')
parser.add_argument('--height', default=336, type=int, help='height of RGB image')
parser.add_argument('--savedir', default='./Outputs', type=str, help='directory to save the results')
parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                    help='Run on CPU or GPU. If TRUE, then GPU')
parser.add_argument('--pretrained', default=None, type=str, help='pretrained model')
parser.add_argument('--model', default='Models.SAMNet', type=str, help='which model to test')

args = parser.parse_args()
print('Called with args:')
for (key, value) in vars(args).items():
    print('{0:10} | {1}'.format(key, value))

exec('from {} import FastSal as net'.format(args.model))

if not osp.isdir(args.savedir):
    os.mkdir(args.savedir)

logger = Logger(osp.join(args.savedir, 'results.txt'))

model = net()
if not osp.isfile(args.pretrained):
    print('Pre-trained model file does not exist...')
    exit(-1)
state_dict = torch.load(args.pretrained)
if list(state_dict.keys())[0][:7] == 'module.':
    state_dict = {key[7:]: value for key, value in state_dict.items()}
model.load_state_dict(state_dict, strict=True)
print('Model resumed from %s' % args.pretrained)

if args.gpu:
    model = model.cuda()
# set to evaluation mode
model.eval()

# ImageNet statistics
mean = np.array([0.485 * 255., 0.456 * 255., 0.406 * 255.], dtype=np.float32)
std = np.array([0.229 * 255., 0.224 * 255., 0.225 * 255.], dtype=np.float32)

if args.file_list == 'ALL':
    args.file_list = ['ECSSD.txt', 'DUT-OMRON.txt', 'DUTS-TE.txt', 'HKU-IS.txt', 'SOD.txt', 'THUR15K.txt']
else:
    args.file_list = [args.file_list]

for file_list in args.file_list:
    image_list = list()
    label_list = list()
    with open(osp.join('Lists', file_list)) as lines:
        for line in lines:
            line_arr = line.split()
            image_list.append(line_arr[0].strip())
            label_list.append(line_arr[1].strip())

    if not osp.isdir(osp.join(args.savedir, osp.dirname(image_list[0]))):
        os.mkdir(osp.join(args.savedir, osp.dirname(image_list[0])))

    saleval = SalEval()
    for idx in range(len(image_list)):
        image = Image.open(osp.join(args.data_dir, image_list[idx])).convert('RGB')
        image = np.array(image, dtype=np.float32)
        height, width = image.shape[:2]
        image = (image - mean) / std
        image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).unsqueeze(0)

        label = Image.open(osp.join(args.data_dir, label_list[idx])).convert('L')
        label = np.array(label, dtype=np.uint8)
        label = torch.ByteTensor(label / 255).unsqueeze(0)

        if args.gpu:
            image, label = image.cuda(), label.cuda()
        start_time = time.time()
        with torch.no_grad():
            pred = model(image)[:, 0, :, :].unsqueeze(1)
            torch.cuda.synchronize()
        diff_time = time.time() - start_time
        print('\rSaliency prediction for {} dataset [{}/{}] takes {:.3f}s per image'\
                .format(osp.dirname(image_list[idx]), idx + 1, len(image_list), diff_time),
                end='' if idx + 1 != len(image_list) else '\n')

        assert pred.shape[-2:] == image.shape[-2:], '%s vs. %s' % (str(pred.shape), str(image.shape))
        pred = F.interpolate(pred, size=[height, width], mode='bilinear', align_corners=False)
        pred = pred.squeeze(1)
        saleval.addBatch(pred, label)

        pred = (pred[0] * 255).cpu().numpy().astype(np.uint8)
        cv2.imwrite(osp.join(args.savedir, label_list[idx]), pred)

    F_beta, MAE = saleval.getMetric()
    logger.info('%10s: F_beta (Val) = %.4f, MAE (Val) = %.4f' % (osp.dirname(image_list[0]), F_beta, MAE))

logger.close()
