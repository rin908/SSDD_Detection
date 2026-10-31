"""只读 COCO 源数据；显式运行后生成独立的规范化副本。"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'Official-SSDD-OPEN/BBox_RBox_PSeg_SSDD/coco_style'


def resolve(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def pixel_hash(path):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert('RGB')
        return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()


def select(doc, ids):
    out = copy.deepcopy(doc)
    out['images'] = [i for i in out['images'] if i['id'] in ids]
    out['annotations'] = [a for a in out['annotations'] if a['image_id'] in ids]
    return out


def prepare(source, target, audit_only=False):
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('输入、输出目录不能互相包含')
    docs, paths, hashes = {}, {}, {}
    bucket_changes = total = 0
    def bucket(area):
        return 'S' if area < 32**2 else ('M' if area < 96**2 else 'L')
    for split in ('train', 'test', 'test_inshore', 'test_offshore'):
        doc = json.loads((source / 'annotations' / f'{split}.json').read_text(encoding='utf-8'))
        if len(doc['categories']) != 1 or doc['categories'][0]['id'] != 0:
            raise ValueError(f'{split}: 预期源数据只有 category_id=0')
        ids = {i['id'] for i in doc['images']}
        if len(ids) != len(doc['images']):
            raise ValueError(f'{split}: 重复 image id')
        if len({a['id'] for a in doc['annotations']}) != len(doc['annotations']):
            raise ValueError(f'{split}: 重复 annotation id')
        hashes[split] = {}
        for im in doc['images']:
            name = im['file_name']
            if Path(name).name != name or '\\' in name:
                raise ValueError('源 file_name 必须是文件名')
            path = source / 'images' / split / name
            digest = pixel_hash(path)
            from PIL import Image
            with Image.open(path) as image:
                if image.size != (im['width'], im['height']):
                    raise ValueError(f'图片尺寸与 JSON 不一致: {name}')
            if name in paths and pixel_hash(paths[name]) != digest:
                raise ValueError(f'平铺文件名冲突: {name}')
            paths.setdefault(name, path)
            hashes[split][im['id']] = digest
        for ann in doc['annotations']:
            if ann['image_id'] not in ids or ann['category_id'] != 0:
                raise ValueError('标注引用非法')
            x, y, w, h = ann['bbox']
            if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                raise ValueError('非法 bbox')
            if split in ('train', 'test'):
                total += 1
                bucket_changes += bucket(ann['area']) != bucket(w * h)
            ann['category_id'], ann['area'] = 1, w * h
        doc['categories'][0]['id'] = 1
        docs[split] = doc
    # 子集必须是完整测试集的同一批图及同样的框，禁止子集参与训练。
    test_images = {i['id']: i for i in docs['test']['images']}
    for split in ('test_inshore', 'test_offshore'):
        for im in docs[split]['images']:
            if im != test_images.get(im['id']) or hashes[split][im['id']] != hashes['test'][im['id']]:
                raise ValueError('子集与 test 图片不一致')
            def boxes(doc):
                return sorted((tuple(a['bbox']), a['category_id'], a.get('iscrowd', 0))
                              for a in doc['annotations'] if a['image_id'] == im['id'])
            if boxes(docs[split]) != boxes(docs['test']):
                raise ValueError('子集与 test 框不一致')
    shore = set(hashes['test_inshore'])
    sea = set(hashes['test_offshore'])
    if shore & sea or shore | sea != set(test_images):
        raise ValueError('近海/远海子集未完整、互斥地覆盖测试集')
    test_by_hash = defaultdict(list)
    for im in docs['test']['images']:
        test_by_hash[hashes['test'][im['id']]].append(im['file_name'])
    duplicates = [dict(train=i['file_name'], test=test_by_hash[hashes['train'][i['id']]])
                  for i in docs['train']['images'] if hashes['train'][i['id']] in test_by_hash]
    clean = [i for i in docs['train']['images'] if hashes['train'][i['id']] not in test_by_hash]
    # 同内容图片作为一个组切分，防止 train/val 泄漏。
    groups = defaultdict(list)
    for im in sorted(clean, key=lambda i: i['file_name']):
        groups[hashes['train'][im['id']]].append(im['id'])
    ordered = list(groups.values())
    random.Random(0).shuffle(ordered)
    desired = round(len(docs['train']['images']) * 0.1)
    val_ids = set()
    for group in ordered:
        if len(val_ids) >= desired:
            break
        val_ids.update(group)
    clean_ids = {i['id'] for i in clean}
    main = dict(train=select(docs['train'], clean_ids - val_ids),
                val=select(docs['train'], val_ids), test=docs['test'])
    report = dict(seed=0, hash='SHA256(decoded RGB + dimensions)',
                  policy='remove duplicates from train before validation split; keep test fixed',
                  source_counts={k: len(v['images']) for k, v in docs.items()},
                  main_counts={k: len(v['images']) for k, v in main.items()},
                  duplicate_matches=duplicates,
                  duplicate_group_count=len({hashes['train'][i['id']] for i in docs['train']['images']
                                             if hashes['train'][i['id']] in test_by_hash}),
                  removed_train=len(duplicates),
                  area_bucket_changes=bucket_changes, annotations_checked=total,
                  area_bucket_change_fraction=bucket_changes / total,
                  note='桶变化仅诊断；32^2/96^2 边界以 COCOeval 实际包含规则为准')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if audit_only:
        return report
    # 防止误覆盖其他数据；重复运行先比对现有图片。
    for name, src in paths.items():
        dst = target / 'images' / name
        if dst.exists() and hashlib.sha256(dst.read_bytes()).digest() != hashlib.sha256(src.read_bytes()).digest():
            raise ValueError(f'输出已存在不同图片: {name}')
    (target / 'images').mkdir(parents=True, exist_ok=True)
    for name, src in paths.items():
        dst = target / 'images' / name
        if not dst.exists():
            shutil.copyfile(src, dst)
    annotations = target / 'annotations'
    for split, doc in main.items():
        dump(annotations / f'{split}_norm.json', doc)
    dump(annotations / 'val.json', main['val'])  # 用户要求的显式别名
    original_train_ids = set(hashes['train']) - val_ids
    for split, doc in dict(train=select(docs['train'], original_train_ids),
                           val=main['val'], test=docs['test']).items():
        dump(annotations / 'original' / f'{split}_norm.json', doc)
    for split in ('test_inshore', 'test_offshore'):
        dump(annotations / 'subsets' / f'{split}.json', docs[split])
    dump(target / 'prepare_report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default=SOURCE)
    p.add_argument('--output', default=os.environ.get('SSDD_DATA_ROOT', 'data/SSDD'))
    p.add_argument('--audit-only', action='store_true')
    a = p.parse_args()
    prepare(resolve(a.source).resolve(), resolve(a.output).resolve(), a.audit_only)


if __name__ == '__main__':
    main()
