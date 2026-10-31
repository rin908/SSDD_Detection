"""batch=1 FP32：磁盘读图到 NMS 后 CPU 输出，warmup 50、计时 200 次。"""
import argparse
import json
import subprocess
import time
from copy import deepcopy
from common import add_arguments, load_config, project_path, save_result


def main():
    import torch
    from mmdet.apis import init_detector, inference_detector
    from mmdet.utils import register_all_modules
    from mmcv.transforms import Compose
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--work-dir')
    args = p.parse_args()
    register_all_modules()
    cfg = load_config(args)
    if not torch.cuda.is_available():
        raise RuntimeError('此口径必须使用 CUDA GPU')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = init_detector(deepcopy(cfg), str(project_path(args.checkpoint)), device='cuda:0').float().eval()
    pipeline = Compose([deepcopy(t) for t in cfg.test_dataloader.dataset.pipeline
                        if t['type'] != 'LoadAnnotations'])
    dataset = cfg.test_dataloader.dataset
    ann = json.loads((project_path(cfg.data_root) / dataset.ann_file).read_text(encoding='utf-8'))
    images = [project_path(cfg.data_root) / 'images' / i['file_name']
              for i in sorted(ann['images'], key=lambda i: i['file_name'])]
    if not images:
        raise ValueError('测试集为空')
    def infer(index):
        # 每次传文件名，包含读图、预处理、H2D、网络、NMS、D2H 和 Python 输出。
        result = inference_detector(model, str(images[index % len(images)]), test_pipeline=pipeline)
        pred = result.pred_instances
        return dict(bboxes=pred.bboxes.cpu().tolist(), scores=pred.scores.cpu().tolist(),
                    labels=pred.labels.cpu().tolist())
    elapsed = []
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        for index in range(50):
            infer(index)
        torch.cuda.synchronize()
        for index in range(200):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = infer(index)
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
    try:
        driver = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            text=True).strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        driver = 'unknown (nvidia-smi unavailable)'
    mean = sum(elapsed) / len(elapsed)
    result = dict(fps=1 / mean, latency_ms=mean * 1000, warmup=50, iterations=200,
                  batch_size=1, precision='FP32; TF32 off', gpu=torch.cuda.get_device_name(0),
                  driver=driver, torch=torch.__version__, cuda=torch.version.cuda,
                  input_pipeline=cfg.test_dataloader.dataset.pipeline,
                  output='CPU Python lists after NMS; no visualization or disk serialization',
                  disk_cache='OS cache may be warm; no application image cache')
    print('| GPU | 驱动 | FPS | latency (ms) |\n|---|---|---:|---:|')
    print(f'| {result["gpu"]} | {driver} | {1/mean:.3f} | {mean*1000:.3f} |')
    print('输入 pipeline:', result['input_pipeline'])
    save_result(args, cfg, 'benchmark', result, args.checkpoint)


if __name__ == '__main__':
    main()
