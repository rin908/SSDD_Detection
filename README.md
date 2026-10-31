# SSDD 船舶检测

MMDetection 3.3.0 / Faster R-CNN / ResNet-50-FPN，单卡 8GB 配置。当前阶段仅交付代码与文档，没有安装环境、训练、评测或生成实测指标。

按顺序阅读：[环境搭建](docs/01_环境搭建.md)、[数据准备](docs/02_数据准备.md)、[训练与评测](docs/03_训练与评测.md)、[常见报错](docs/04_常见报错.md)、[云服务器迁移](docs/05_云服务器迁移.md)。[超参表](docs/超参表.md)列出全部关键设置。

```text
configs/       基础模型与可选 1024 变体
tools/         数据准备、训练、评测、Params、FPS、汇总、环境校验
scripts/       PowerShell / Bash 等价训练入口
docs/          中文操作文档
data/          用户运行准备命令后生成（已忽略）
results/       指标 JSON 与 Markdown 汇总
work_dirs/     用户启动训练后生成（已忽略）
Official-SSDD-OPEN/  原始数据，只读（已忽略）
```

完成环境安装后，在项目根运行以下命令准备数据；预计 10–60 秒，输出 train=823、val=93、test=232。失败时先运行 `python tools/prepare_data.py --audit-only` 按报错检查文件，勿修改原始数据。

```bash
python tools/prepare_data.py
```

只复制 COCO train/test 图片的文件名并集，不复制 VOC、旋转框或分割目录。主结果从训练侧移除 10 组中与测试像素内容相同的 12 张图（其中 10 张文件字节也完全相同）；对照保留这 12 张训练图，其余修正及验证集相同。原始文件从未覆盖。

所有 Python 入口通过脚本位置定位项目根，启动脚本通过自身目录定位；`SSDD_DATA_ROOT` 指向**规范化数据根**，原始数据位置用 `prepare_data.py --source` 指定。项目中没有硬编码本机盘符。

接口按官方 [MMDetection v3.3.0 配置](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py)、[训练入口](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/tools/train.py)、[推理接口](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/apis/inference.py)编写。实际环境验收必须由安装后的 `tools/check_environment.py` 完成；静态检查不能替代 CUDA 验证。
