import os
import cv2


def convert_to_grayscale(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 构建完整的文件路径
        input_path = os.path.join(input_folder, filename)

        # 检查文件是否是图像文件
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 读取彩色图像
            image = cv2.imread(input_path)

            # 将图像转换为灰度图像
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 构建输出文件路径
            output_path = os.path.join(output_folder, filename)

            # 保存灰度图像
            cv2.imwrite(output_path, gray_image)

            print(f'Converted {input_path} to {output_path}')


# 使用示例
input_folder = './data/crack/map/'  # 替换为输入文件夹路径
output_folder = './data/crack_grey/map/'  # 替换为输出文件夹路径
convert_to_grayscale(input_folder, output_folder)
