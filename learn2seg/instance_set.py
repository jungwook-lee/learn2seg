import os
import cv2


class InstanceDataset(object):
    splits = ['train', 'val', 'test']
    split_path = dict()
    im_size = None

    def __init__(self, config):
        conf = config['dataset_config']
        path = conf['data_path']

        # Check if the root path exists first
        if not os.path.exists(path):
            raise RuntimeError("Dataset is missing at: " + path)

        for split in self.splits:
            self.split_path[split] = os.path.join(path, split)

        self.im_size = conf['image_size']

    def get_data_size(self, split):
        """ Assumes that image/label files have identical number of samples """
        path = os.path.join(self.split_path[split], 'image')
        return len(os.listdir(path))

    def get_image(self, split, index, label=False):
        """ Given index of the data, return the corresponding image """
        if label:
            type_str = 'label'
        else:
            type_str = 'image'

        im_file_name = str(index) + '.png'
        im_path = os.path.join(self.split_path[split], type_str, im_file_name)

        return cv2.imread(im_path, 1)

    def get_label(self, split, index):
        return


if __name__ == "__main__":
    # TODO: move the testing code to test suite
    ex_conf = dict()

    data_conf = dict()
    data_conf['data_path'] = '/home/jung/dataset/seg_teeth_v2'
    data_conf['im_size'] = (10, 10)
    ex_conf['dataset_config'] = data_conf

    dataset = InstanceDataset(ex_conf)
    print(dataset.get_data_size('train'))
    print(dataset.get_image('train', 1).shape)
