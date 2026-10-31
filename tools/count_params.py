"""无需数据或权重，统计 backbone + neck + RPN/ROI head 的全部参数。"""
import argparse
from copy import deepcopy
from common import add_arguments, load_config, save_result


def main():
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.add_argument('--work-dir')
    args = p.parse_args()
    register_all_modules()
    cfg = load_config(args)
    # build 不调用 init_weights，不触发 ImageNet 权重下载。
    model = MODELS.build(deepcopy(cfg.model))
    seen, rows = set(), {}
    for name in ('backbone', 'neck', 'rpn_head', 'roi_head'):
        count = 0
        for param in getattr(model, name).parameters():
            if id(param) not in seen:
                count += param.numel()
                seen.add(id(param))
        rows[name] = count / 1e6
    rows['total_M'] = sum(rows.values())
    print('| 模块 | Params (M) |\n|---|---:|')
    for name, count in rows.items():
        print(f'| {name} | {count:.4f} |')
    save_result(args, cfg, 'params', rows)


if __name__ == '__main__':
    main()
