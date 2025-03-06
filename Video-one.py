import cv2
from ultralytics import YOLO
import os

# 初始化模型
model = YOLO('yolo11x.pt')
video_path = 'D:\\yolo11\\JPEGlmages\\2 - Trim.mp4'
output_folder = 'Annotations'
output_video_path = os.path.join(output_folder, '2 - Trim.mp4')

# 获取视频的帧率、宽度和高度
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义视频编解码器并创建输出视频文件，这里改为H264编码
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 对每一帧进行预测
    results = model.predict(frame)
    for result in results:
        annotated_frame = result.plot()

    # 将标注好的帧写入输出视频
    out.write(annotated_frame)

cap.release()
out.release()