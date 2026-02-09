import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys
import torch
import torch.distributed as dist

def setup_distributed_env():
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)

def parse_args():
    parser = argparse.ArgumentParser(description='SAM3 OVSS evaluation')
    parser.add_argument('--config', required=True)
    parser.add_argument('--work-dir', default='',
                        help='work directory to save logs/checkpoints.')
    parser.add_argument('--show-dir', default='',
                        help='directory to save visualization images.')
    parser.add_argument('--launcher', default='none', 
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        help="Launch mode.")
    return parser.parse_args()


def main():
    setup_distributed_env()
    from mmengine.config import Config
    from mmengine.runner import Runner
    
    import custom_datasets
    import sam3_ovss
    
    def infer_dataset_name_from_config(config_path: str) -> str:
        base = os.path.basename(config_path)
        stem = os.path.splitext(base)[0]
        if stem.startswith('cfg_'):
            stem = stem[len('cfg_'):]
        return stem

    args = parse_args()
    cfg = Config.fromfile(args.config)

    if not args.work_dir:
        dataset_name = infer_dataset_name_from_config(args.config)
        cfg.work_dir = os.path.join('./work_dirs', dataset_name)
    else:
        cfg.work_dir = args.work_dir
    
    if args.show_dir:
        if 'visualization' in cfg.default_hooks:
            cfg.default_hooks['visualization']['draw'] = True
            cfg.visualizer['alpha'] = 0.6
            cfg.visualizer['save_dir'] = args.show_dir
        else:
            print("Warning: VisualizationHook not found in config.")

    cfg.launcher = args.launcher

    runner = Runner.from_cfg(cfg)
    runner.test()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
