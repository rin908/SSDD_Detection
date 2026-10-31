"""所有入口统一项目根、数据根和结果记录。"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = 'configs/faster_rcnn_r50_fpn_ssdd.py'


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(args):
    from mmengine.config import Config
    os.chdir(ROOT)
    cfg = Config.fromfile(str(project_path(args.config)))
    if getattr(args, 'cfg_options', None):
        cfg.merge_from_dict(args.cfg_options)
    root = project_path(os.environ.get('SSDD_DATA_ROOT', cfg.data_root))
    cfg.data_root = root.as_posix() + '/'
    original = getattr(args, 'original', False)
    for split in ('train', 'val', 'test'):
        dataset = cfg[f'{split}_dataloader'].dataset
        dataset.data_root = cfg.data_root
        if original:
            dataset.ann_file = f'annotations/original/{split}_norm.json'
        if split != 'train':
            cfg[f'{split}_evaluator'].ann_file = str(root / dataset.ann_file)
    subset = getattr(args, 'subset', 'test')
    if subset != 'test':
        ann = f'annotations/subsets/test_{subset}.json'
        cfg.test_dataloader.dataset.ann_file = ann
        cfg.test_evaluator.ann_file = str(root / ann)
    cfg.work_dir = str(project_path(getattr(args, 'work_dir', None) or cfg.work_dir))
    if original:
        cfg.work_dir += '_original'
    return cfg


def add_arguments(parser):
    from mmengine.config import DictAction
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--original', action='store_true', help='未去重对照，保留面积/类别修正')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_result(args, cfg, section, value, checkpoint=None):
    from summarize import summarize
    # 各实验独立目录，改变权重/配置时拒绝拼接旧结果。
    out = ROOT / 'results' / Path(cfg.work_dir).name
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'metrics.json'
    record = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    signature = hashlib.sha256(repr(dict(
        model=cfg.model.to_dict(), pipeline=cfg.test_dataloader.dataset.pipeline,
        original=args.original)).encode()).hexdigest()
    if record.get('signature', signature) != signature:
        raise ValueError('结果目录已有其他配置，请指定新的 --work-dir')
    record['signature'] = signature
    if checkpoint:
        digest = sha256(project_path(checkpoint))
        if record.get('checkpoint_sha256', digest) != digest:
            raise ValueError('结果目录已有其他权重，请指定新的 --work-dir')
        record['checkpoint_sha256'] = digest
        record['checkpoint'] = Path(checkpoint).name
    record['experiment'] = out.name
    record[section] = value
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)
    summarize()
    print(f'已写入 results/{out.name}/metrics.json 和 results/summary.md')
