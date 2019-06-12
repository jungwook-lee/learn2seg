""" Unet training code from https://github.com/zhixuhao/unet """
import argparse

from learn2seg.feeder import *
from learn2seg.datasets.instance import InstanceDataset

import learn2seg.tools as tools
import learn2seg.trainer as trainer
import learn2seg.evaluator as evaluator


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

    his = trainer.train_model(new_dataset, train_config)
    trainer.plot_model(his, train_config)

    # Evaluate the performance
    evaluator.eval(new_dataset, train_config)
