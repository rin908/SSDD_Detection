"""仅显式运行才评测；默认完整 test，可选近海/远海。"""
import argparse
from copy import deepcopy
from common import add_arguments, load_config, project_path, save_result


def main():
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--work-dir')
    p.add_argument('--subset', choices=['test', 'inshore', 'offshore'], default='test')
    args = p.parse_args()
    register_all_modules()
    cfg = load_config(args)
    cfg.load_from = str(project_path(args.checkpoint))
    metrics = Runner.from_cfg(deepcopy(cfg)).test()
    save_result(args, cfg, args.subset, {k: float(v) for k, v in metrics.items()}, args.checkpoint)


if __name__ == '__main__':
    main()
