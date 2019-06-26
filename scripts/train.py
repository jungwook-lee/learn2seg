""" Unet training code from https://github.com/zhixuhao/unet """
import argparse
from shutil import copyfile

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
    model_config = config_dict['model_config']
    train_config = config_dict['train_config']
    eval_config = config_dict['eval_config']
    new_dataset = InstanceDataset(config_dict)

    out_path = config_dict['out_path']
    if os.path.exists(out_path):
        raise ValueError("Out path already exists!")
    else:
        os.mkdir(out_path)
    
    # Copy the config file before beginning the training
    config_out_file = os.path.join(out_path, 'run_config.yaml')
    copyfile(args.configs[0], config_out_file)

    # Implement iterative training
    train_iterations = train_config['train_iterations']
    for it in range(train_iterations):

        his = trainer.train_model(new_dataset,
                                  model_config,
                                  train_config,
                                  out_path,
                                  it)

        trainer.plot_model(his, out_path, it)

        # Evaluate the performance
        evaluator.eval(new_dataset, model_config, eval_config, out_path, it)

        # Filter the output with less than 50 % iou
        pre_dir, cur_dir = filter.get_label_dirs(config_dict, it)
        print(pre_dir, cur_dir)

        print('------- End of Iteration --------')
        
        # Don't filter output for the final iteration
        if it < (train_iterations - 1):
          filter.iou_filter(pre_dir=pre_dir, cur_dir=cur_dir, iou_thresh=0.5)
