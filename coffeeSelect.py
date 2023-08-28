import cv2
from PIL import Image
import numpy as np

# 读取原始图片
from matplotlib import pyplot as plt

image = cv2.imread(r'E:\pythonProject\imageSegmentation\image\coffee\imag4.jpg')
# 增加图片亮度和对比度
# alpha = 1.5  # 亮度增益
# beta = 30  # 对比度增益
# image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# 将图像转换为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义颜色范围（在HSV颜色空间中）
yellow_lower = np.array([18, 50, 100])
yellow_upper = np.array([60, 255, 255])
black_lower = np.array([40, 0, 0])
black_upper = np.array([180, 255, 200])
brown_lower = np.array([0, 0, 0])
brown_upper = np.array([100, 80, 200])

# 根据颜色范围创建掩码
yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
black_mask = cv2.inRange(hsv_image, black_lower, black_upper)
brown_mask = cv2.inRange(hsv_image, brown_lower, brown_upper)



# 使用不同的颜色标注每种咖啡豆的区域
yellow_color = (0, 255, 255)  # 黄色
black_color = (255, 0, 0)  # 蓝色
brown_color = (0,  255,0)  # 绿色
image_with_labels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image_with_labels[np.where(yellow_mask == 255)] = yellow_color
# image_with_labels[np.where(black_mask == 255)] = black_color
image_with_labels[np.where(brown_mask == 255)] = black_color

# 计算每种颜色的占比
total_pixels = image.shape[0] * image.shape[1]
yellow_pixels = cv2.countNonZero(yellow_mask)
black_pixels = cv2.countNonZero(black_mask)
brown_pixels = cv2.countNonZero(brown_mask)

yellow_percentage = (yellow_pixels / total_pixels) * 100
black_percentage = (black_pixels / total_pixels) * 100
brown_percentage = (brown_pixels / total_pixels) * 100

# 显示图像和占比
fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.text(0.5, 0.1, 'Brown: {:.2f}%'.format(brown_percentage), ha='center', va='center',color='yellow',
             transform=plt.gca().transAxes)
plt.imshow( image_with_labels)
plt.show()
print('Yellow percentage:', yellow_percentage)
print('Black percentage:', black_percentage)
print('Brown percentage:', brown_percentage)
