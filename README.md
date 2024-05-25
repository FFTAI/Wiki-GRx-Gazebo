# rl_sim

[中文文档](README_CN.md)

Simulation verification of robot reinforcement learning algorithms. 

## Preparation

Clone the code

```bash
git clone https://gitee.com/FourierIntelligence/wiki-grx-gazebo.git
```

## Dependency

Download and deploy `libtorch` at any location

```bash
cd /path/to/your/torchlib
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cpu.zip -d ./
echo 'export Torch_DIR=/path/to/your/torchlib' >> ~/.bashrc
```

Install dependency packages

```bash
sudo apt install ros-noetic-teleop-twist-keyboard ros-noetic-controller-interface  ros-noetic-gazebo-ros-control ros-noetic-joint-state-controller ros-noetic-effort-controllers ros-noetic-joint-trajectory-controller
```

Install yaml-cpp

```bash
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp && mkdir build && cd build
cmake -DYAML_BUILD_SHARED_LIBS=on .. && make
sudo make install
sudo ldconfig
```

Install lcm

```bash
git clone https://github.com/lcm-proj/lcm.git 
cd lcm && mkdir build && cd build
cmake .. && make
sudo make install
sudo ldconfig
```

## Compilation

```bash
catkin build
source devel/setup.bash
```

## Running

Before running, copy the trained pt model file to `rl_sim/src/rl_sim/models/YOUR_ROBOT_NAME`, and configure the parameters in `config.yaml`.

Open a new terminal, launch the gazebo simulation environment

```bash
source devel/setup.bash
roslaunch rl_sim gazebo_gr1t1.launch
```

Press **0** on the keyboard to switch the robot to the default standing position, press **P** to switch to RL control mode, and press **1** in any state to switch to the initial lying position. WS controls x-axis, AD controls yaw, and JL controls y-axis.

Press **R** to reset Gazebo environment.

## Issues
1. `catkin build` error info : Unable to find either executable 'empy' or Python module 'em'... try installing the package 'python-empy'
  - https://github.com/ysl208/iRoPro/issues/59
  - `catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3`

## Citation

Please cite the following if you use this code or parts of it:

```
@software{fan-ziqi2024rl_sar,
  author = {fan-ziqi},
  title = {{rl_sim: Simulation Verification and Physical Deployment of the Quadruped Robot's Reinforcement Learning Algorithm.}},
  url = {https://github.com/fan-ziqi/rl_sim},
  year = {2024}
}
```
