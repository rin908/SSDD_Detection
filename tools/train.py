"""MMDetection 3.x 单卡训练入口；仅显式运行才启动训练。"""
import argparse
from common import ROOT, add_arguments, load_config, project_path


def main():
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.add_argument('--work-dir')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--resume', nargs='?', const='auto')
    args = p.parse_args()
    register_all_modules()
    cfg = load_config(args)
    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'
    if args.resume:
        cfg.resume = True
        cfg.load_from = None if args.resume == 'auto' else str(project_path(args.resume))
    effective_lr = cfg.optim_wrapper.optimizer.lr
    if cfg.auto_scale_lr.enable:
        effective_lr *= cfg.train_dataloader.batch_size / cfg.auto_scale_lr.base_batch_size
    print(f'单卡有效学习率（warmup 前）: {effective_lr:g}')
    from summarize import summarize
    try:
        Runner.from_cfg(cfg).train()
    finally:
        summarize()  # 即使中断，也汇总已写出的训练日志


if __name__ == '__main__':
    main()
