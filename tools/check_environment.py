"""允许单独运行的版本、CUDA 算子与配置冒烟检查，不训练、不下载权重。"""
from common import ROOT


def main():
    import torch
    import torchvision
    import mmcv
    import mmengine
    import mmdet
    from mmcv.ops import nms
    from mmengine.config import Config
    actual = (torch.__version__, torch.version.cuda, mmcv.__version__, mmdet.__version__)
    print(*actual)
    assert actual == ('2.1.0+cu118', '11.8', '2.1.0', '3.3.0'), actual
    assert torchvision.__version__ == '0.16.0+cu118'
    assert mmengine.__version__ == '0.10.4'
    assert torch.cuda.is_available(), 'CUDA GPU 不可用'
    boxes = torch.tensor([[0., 0., 10., 10.]], device='cuda')
    nms(boxes, torch.ones(1, device='cuda'), 0.5)
    for name in ('faster_rcnn_r50_fpn_ssdd.py', 'faster_rcnn_r50_fpn_ssdd_1024.py'):
        cfg = Config.fromfile(str(ROOT / 'configs' / name))
        assert cfg.model.roi_head.bbox_head.num_classes == 1
        print(name, 'Config OK')
    print('CUDA NMS / 环境检查通过')


if __name__ == '__main__':
    main()
