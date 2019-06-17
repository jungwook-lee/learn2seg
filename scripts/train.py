""" Unet training code from https://github.com/zhixuhao/unet """
import argparse

from learn2seg.feeder import *
from learn2seg.datasets.instance import InstanceDataset

import learn2seg.tools as tools
import learn2seg.trainer as trainer
import learn2seg.evaluator as evaluator
import learn2seg.filter as filter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training step for learn2seg')
    parser.add_argument('configs', metavar='c', type=str, nargs='+',
                        help='configuration file to run training')

    args = parser.parse_args()
    config_dict = tools.get_configs(args.configs[0])
    train_config = config_dict['train_config']
    new_dataset = InstanceDataset(config_dict)

    out_path = train_config['out_path']
    if os.path.exists(out_path):
        raise ValueError("Out path already exists!")
    else:
        os.mkdir(out_path)

    # Implement iterative training
    train_iterations = train_config['train_iterations']
    for train_it in range(train_iterations):
        his = trainer.train_model(new_dataset, train_config, train_it)
        trainer.plot_model(his, train_config, train_it)

        # Evaluate the performance
        evaluator.eval(new_dataset, train_config, train_it)

        # Filter the output with less than 50 % iou
        pre_dir, cur_dir = filter.get_label_dirs(config_dict, train_it)
        print(pre_dir, cur_dir)

        print('------- End of Iteration --------')
        filter.iou_filter(pre_dir=pre_dir, cur_dir=cur_dir, iou_thresh=0.5)
