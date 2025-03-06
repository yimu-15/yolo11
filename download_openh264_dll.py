import requests
import shutil
import os

# 这里是从GitHub获取文件的正确链接，记得检查是否最新
url = "http://ciscobinary.openh264.org/openh264-2.3.1-win64.dll.bz2"
local_path = r"D:\yolo11\venv\Scripts\openh264-2.3.1-win64.dll"

# 检查目标目录是否存在，不存在则创建
if not os.path.exists(os.path.dirname(local_path)):
    os.makedirs(os.path.dirname(local_path))

try:
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(local_path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)

    print("文件下载成功并放置到指定目录。")
except requests.exceptions.RequestException as e:
    print(f"下载过程出现错误: {e}")
except Exception as e:
    print(f"其他错误发生: {e}")