




conda 官网： Miniconda — Anaconda documentation
Miniconda软件安装教程（Linux）_miniconda安装-CSDN博客
如何在Linux服务器上安装Anaconda（超详细）_linux安装anconda-CSDN博客


1.环境要求
建议在 Windows 系统上使用 WSL2 进行开发。

2.创建和激活 Conda 虚拟环境
在终端中依次输入以下命令:
conda create -n mamba python=3.10
conda activate mamba

3.安装 CUDA 工具包和 PyTorch
继续在终端中输入以下命令:
conda install cudatoolkit==11.8 -c nvidia
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

4.安装 CUDA 编译器和其他依赖
输入以下命令来安装 CUDA 编译器和其他依赖:
conda install -c"nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging

5.克隆项目仓库
在终端中运行以下命令来克隆项目
git clone https://github.com/hustvl/vim.git

6.进入项目目录
输入以下命令切换到项目目录:
cd vim

7.安装项目依赖
使用 pip 安装项目依赖:
pip install -r vim/vim_requirements.txt
提示: 如果安装速度慢，可以使用国内镜像源，如清华大学镜像源:
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

8.解决可能的 CUDA 环境问题
在安装 causa1_conv1d 时，可能会遇到如下报错:
No such file or directory:":/usr/local/cuda-11.8/bin/nvcc
解决方式是设置 CUDA_HOME 环境变量:
export CUDA_HOME=/usr/local/cuda
然后安装 causa1_convld:
pip install causal_conv1d>=1.1.0

然后安装 mamba-ssm:
pip install mamba-ssm
or 
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip insta11 .













