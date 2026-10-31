"""汇总已有产物，不启动训练或推理，缺失值写 N/A。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']


def summarize():
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    lines = ['# SSDD 实验汇总', '',
             'AP 使用 COCO 0–1 标度，-1 表示没有可评目标；N/A 表示尚未测量。', '',
             '| 实验 / 测试子集 | AP | AP50 | AP75 | AP_S | AP_M | AP_L | Params (M) | FPS |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    details = []
    for path in sorted(output.glob('*/metrics.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        for subset in ('test', 'inshore', 'offshore'):
            if subset != 'test' and subset not in record:
                continue
            metrics = record.get(subset, {})
            values = [metrics.get(f'coco/bbox_{key}') for key in KEYS]
            # FPS 是完整测试输入的 benchmark，不伪装为子集独立吞吐。
            values += [record.get('params', {}).get('total_M'),
                       record.get('benchmark', {}).get('fps') if subset == 'test' else None]
            cells = ['N/A' if v is None else f'{v:.4f}' for v in values]
            lines.append('| ' + record['experiment'] + '/' + subset + ' | ' + ' | '.join(cells) + ' |')
        bench = record.get('benchmark')
        if bench:
            details += ['', f'{record["experiment"]}：latency={bench["latency_ms"]:.3f} ms；'
                      f'GPU={bench["gpu"]}；驱动={bench["driver"]}；FP32 / batch=1；'
                      f'输入={json.dumps(bench["input_pipeline"], ensure_ascii=False)}', '']
    lines += details
    lines += ['', '## 训练日志（验证集，与测试结果分开）', '',
              '| 实验 / 日志 | 记录数 | 最佳验证 AP | 最后 loss |', '|---|---:|---:|---:|']
    for path in sorted((ROOT / 'work_dirs').glob('**/vis_data/scalars.json')):
        records = []
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # 训练中断时允许末行不完整
        aps = [r['coco/bbox_mAP'] for r in records if 'coco/bbox_mAP' in r and 'epoch' in r]
        losses = [r['loss'] for r in records if 'loss' in r]
        lines.append(f'| {path.relative_to(ROOT).as_posix()} | {len(records)} | '
                     f'{max(aps) if aps else "N/A"} | {losses[-1] if losses else "N/A"} |')
    (output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    summarize()
