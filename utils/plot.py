import sys
import os
import subprocess

sys.path.append('../')
from pycore.tikzeng import *

# 定义神经网络架构
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    # 输入图像的名字
    to_input("1.jpeg", to='(-5,0,0)', width=6, height=6, name="temp"),
    to_input("2.jpeg", to='(-4,0,0)', width=6, height=6, name="temp"),
    to_input("3.jpeg", to='(-3,0,0)', width=6, height=6, name="temp"),
    to_input("4.jpeg", to='(-2,0,0)', width=6, height=6, name="temp"),
    to_input("5.jpeg", to='(-1,0,0)', width=6, height=6, name="temp"),

    # 调用当中的函数绘画模型的内容
    to_Conv("conv1", s_filer=256, n_filer=3, offset="(0,0,0)", to="(0,0,0)", height=50, depth=50, width=3,
            caption='CONV1'),
    to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)", height=32, depth=32, width=3, caption="MaxPool1"),

    to_Conv("conv2", s_filer=63, n_filer=16, offset="(3,0,0)", to="(pool1-east)", height=32, depth=32, width=3,
            caption='CONV2'),
    to_connection("pool1", "conv2"),
    to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=16, depth=16, width=3, caption="MaxPool2"),

    to_Conv("conv3", s_filer=15, n_filer=64, offset="(3,0,0)", to="(pool2-east)", height=16, depth=16, width=3,
            caption='CONV3'),
    to_connection("pool2", "conv3"),
    to_Pool("pool3", offset="(0,0,0)", to="(conv3-east)", height=10, depth=10, width=3, caption="MaxPool3"),

    to_SoftMax(name='fc1', s_filer=64, offset="(4,0,0)", to="(pool3-east)", width=1.5, height=1.5, depth=100,
               opacity=0.8, caption='FC1'),
    to_connection("pool3", "fc1"),

    to_SoftMax(name='fc2', s_filer=10, offset="(2,0,0)", to="(fc1-east)", width=1.5, height=1.5, depth=50,
               opacity=0.8, caption='FC2'),
    to_connection("fc1", "fc2"),

    to_SoftMax(name='fc3', s_filer=5, offset="(2,0,0)", to="(fc2-east)", width=1.5, height=1.5, depth=5,
               opacity=0.8, caption='FC3'),
    to_connection("fc2", "fc3"),

    to_end()
]


def main():
    # 获取文件名
    namefile = str(sys.argv[0]).split('.')[0]

    # 转换成为.tex文件
    to_generate(arch, namefile + '.tex')

    # 使用 LaTeX 编译器将 .tex 文件转换为 .pdf 文件
    subprocess.call([r'D:\MiKTeX\install\miktex\bin\x64\pdflatex.exe', namefile + '.tex'])

    # 生成pdf
    pdf_file = namefile + '.pdf'
    # 生成png
    image_file = namefile + '.png'

    # 将pdf转化成为png
    subprocess.call(
        [r'D:\ghostscript\gs10.01.1\bin\gswin64c.exe', '-sDEVICE=pngalpha', '-o', image_file, '-r300', pdf_file])

    # 删除中间生成的文件
    cleanup(namefile)


# 删除中间生成的文件
def cleanup(namefile):
    extensions = ['.aux', '.log', '.tex']
    for ext in extensions:
        filename = namefile + ext
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == '__main__':
    main()
