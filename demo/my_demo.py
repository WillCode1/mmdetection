import time
import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector


# 指定模型的配置文件和 checkpoint 文件路径
# config_file = '../configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
# checkpoint_file = '../checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

config_file = '../configs/yolox/yolox_tiny_8xb8-300e_coco.py'
checkpoint_file = '../checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# model = init_detector(config_file, checkpoint_file, device='cpu')

# 初始化可视化工具
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# 从 checkpoint 中加载 Dataset_meta，并将其传递给模型的 init_detector
visualizer.dataset_meta = model.dataset_meta

# 测试单张图片并展示结果
img = './demo.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
start_time = time.time()
result = inference_detector(model, img)
end_time = time.time()
print(f"执行时间: {(end_time - start_time) * 1000} ms")

# 显示结果
img = mmcv.imread(img)
img = mmcv.imconvert(img, 'bgr', 'rgb')


visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    show=True)

exit(0)

# 测试视频并展示结果
# 构建测试 pipeline
model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

# 可视化工具在第33行和35行已经初完成了初始化，如果直接在一个 jupyter nodebook 中运行这个 demo，
# 这里则不需要再创建一个可视化工具了。
# 初始化可视化工具
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# 从 checkpoint 中加载 Dataset_meta，并将其传递给模型的 init_detector
visualizer.dataset_meta = model.dataset_meta

# 显示间隔 (ms), 0 表示暂停
wait_time = 1

video = mmcv.VideoReader('video.mp4')

cv2.namedWindow('video', 0)

for frame in track_iter_progress(video):
    result = inference_detector(model, frame, test_pipeline=test_pipeline)
    visualizer.add_datasample(
        name='video',
        image=frame,
        data_sample=result,
        draw_gt=False,
        show=False)
    frame = visualizer.get_image()
    mmcv.imshow(frame, 'video', wait_time)

cv2.destroyAllWindows()
