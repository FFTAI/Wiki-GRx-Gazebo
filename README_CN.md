# rl_sim

[English document](README.md)

机器人强化学习算法的仿真验证。

## 准备

拉取代码

```bash
git clone https://gitee.com/FourierIntelligence/wiki-grx-gazebo.git
```

## 依赖

在任意位置下载并部署`libtorch`

```bash
cd /path/to/your/torchlib
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cpu.zip -d ./
echo 'export Torch_DIR=/path/to/your/torchlib' >> ~/.bashrc
```

安装依赖库

```bash
sudo apt install ros-noetic-teleop-twist-keyboard ros-noetic-controller-interface  ros-noetic-gazebo-ros-control ros-noetic-joint-state-controller ros-noetic-effort-controllers ros-noetic-joint-trajectory-controller
```

安装yaml-cpp

```bash
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp && mkdir build && cd build
cmake -DYAML_BUILD_SHARED_LIBS=on .. && make
sudo make install
sudo ldconfig
```

安装lcm

```bash
git clone https://github.com/lcm-proj/lcm.git 
cd lcm && mkdir build && cd build
cmake .. && make
sudo make install
sudo ldconfig
```

## 编译

```bash
catkin build
source devel/setup.bash
```

## 运行

运行前请将训练好的pt模型文件拷贝到`rl_sim/src/rl_sim/models/YOUR_ROBOT_NAME`中，并配置`config.yaml`中的参数。

新建终端，启动gazebo仿真环境

```bash
source devel/setup.bash
roslaunch rl_sim gazebo_gr1t1.launch
```

按下键盘上的**0**键让机器人切换到默认站起姿态，按下**P**键切换到RL控制模式，任意状态按下**1**键切换到最初的趴下姿态。WS控制x，AD控制yaw，JL控制y。

按**R**重置Gazebo仿真环境。

## Issues
1. `catkin build` error info : Unable to find either executable 'empy' or Python module 'em'... try installing the package 'python-empy'
  - https://github.com/ysl208/iRoPro/issues/59
  - `catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3`

## 引用

如果您使用此代码或其部分内容，请引用以下内容：

```
@software{fan-ziqi2024rl_sar,
  author = {fan-ziqi},
  title = {{rl_sim: Simulation Verification and Physical Deployment of the Quadruped Robot's Reinforcement Learning Algorithm.}},
  url = {https://github.com/fan-ziqi/rl_sim},
  year = {2024}
}
```
