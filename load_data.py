import numpy as np
from PIL import Image
import pickle
import os.path as osp


class LoadData(object):
    '''
    Class to laod the data
    '''
    def __init__(self, data_dir, dataset, cached_data_file):
        '''
        :param data_dir: directory where the dataset is kept
        :param cached_data_file: location where cached file has to be stored
        '''
        self.data_dir = data_dir
        self.dataset = dataset
        self.cached_data_file = cached_data_file
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)

    def process(self):
        '''
        main.py calls this function
        We expect train.txt and val.txt files to be inside the data directory.
        :return:
        '''
        no_files = 0
        with open(osp.join('Lists', self.dataset + '.txt'), 'r') as lines:
            for line in lines:
                # we expect the text file to contain the data in following format
                # <RGB Image>, <Label Image>
                line_arr = line.split()
                img_file = osp.join(self.data_dir, line_arr[0].strip())
                label_file = osp.join(self.data_dir, line_arr[1].strip())

                rgb_img = Image.open(img_file).convert('RGB')
                rgb_img = np.array(rgb_img, dtype=np.float32)
                self.mean[0] += np.mean(rgb_img[:, :, 0])
                self.mean[1] += np.mean(rgb_img[:, :, 1])
                self.mean[2] += np.mean(rgb_img[:, :, 2])

                self.std[0] += np.std(rgb_img[:, :, 0])
                self.std[1] += np.std(rgb_img[:, :, 1])
                self.std[2] += np.std(rgb_img[:, :, 2])

                no_files += 1

        # divide the mean and std values by the sample space size
        self.mean /= no_files
        self.std /= no_files

        data_dict = dict()
        data_dict['mean'] = self.mean
        data_dict['std'] = self.std
        pickle.dump(data_dict, open(self.cached_data_file, 'wb'))

        return data_dict
