import os.path as osp
from PIL import Image
from torch.utils.data import Dataset


class Dataset(Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, data_dir, dataset, transform=None):
        '''
        :param data_dir: directory where the dataset is kept
        :param transform: Type of transformation. SEe Transforms.py for supported transformations
        '''
        self.data_dir = data_dir
        self.transform = transform
        self.img_list = list()
        self.msk_list = list()
        with open(osp.join('Lists', dataset + '.txt'), 'r') as lines:
            for line in lines:
                line_arr = line.split()
                self.img_list.append(osp.join(self.data_dir, line_arr[0].strip()))
                self.msk_list.append(osp.join(self.data_dir, line_arr[1].strip()))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        '''
        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        image = Image.open(self.img_list[idx]).convert('RGB')
        label = Image.open(self.msk_list[idx]).convert('L')
        if self.transform is not None:
            [image, label] = self.transform(image, label)

        return image, label
