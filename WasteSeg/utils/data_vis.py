import matplotlib.pyplot as plt
"""
这段代码定义了一个名为 plot_img_and_mask 的函数，
用于绘制输入图像和输出掩模的图像。"""


def plot_img_and_mask(img, mask):
    # img：输入图像、mask：输出掩码
    fig = plt.figure()  # 创建一个图像对象fig
    a = fig.add_subplot(1, 2, 1)  # 在fig中添加两个子图，一个用于绘制输入图像，另一个用于绘制输出掩码
    a.set_title('Input images')  # 对输入图像子图进行设置，包括设置标题和绘制图像

    plt.imshow(img)
    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')  # 对输出掩码子图进行设置，包括设置标题和绘制掩码图像

    plt.imshow(mask)
    plt.show()  # 最后显示绘制的图像和掩码
