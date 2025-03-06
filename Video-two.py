import cv2
from ultralytics import YOLO
import os
import json

# 初始化模型
model = YOLO('yolo11x.pt')
video_path = 'D:\\yolo11\\JPEGlmages\\2 - Trim.mp4'
output_folder = 'Annotations'

# 创建保存图片的文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)
frame_count = 0
all_results = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 对每一帧进行预测
    results = model.predict(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            result_dict = {
                'class': int(box.cls),
                'confidence': float(box.conf),
                'xyxy': box.xyxy.tolist()
            }
            all_results.append(result_dict)

        # 在图片上绘制检测结果
        annotated_frame = result.plot()

        # 保存带有检测结果的图片
        output_path = os.path.join(output_folder, f'frame_{frame_count}.jpg')
        cv2.imwrite(output_path, annotated_frame)

    frame_count += 1

cap.release()

# 保存检测结果为json文件
with open(os.path.join(output_folder, 'detection_results.json'), 'w') as f:
    json.dump(all_results, f)