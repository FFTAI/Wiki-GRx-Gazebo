# Wiki-GRx-Gazebo

<img src="./pictures/gr1t1_webots.png" width="300" height="360" />
<img src="./pictures/gr1t2_webots.png" width="300" height="360" />

This repository provides an environment used to test the RL policy trained in NVIDIA's Isaac Gym on the GRx robot model in Gazebo.

## User Guide

1. Download and deploy `libtorch` at any location

```bash
cd /path/to/your/libtorch
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cpu.zip -d ./
echo 'export Torch_DIR=/path/to/your/libtorch' >> ~/.bashrc
```

2. Install dependency packages

```bash
sudo apt install ros-noetic-teleop-twist-keyboard ros-noetic-controller-interface ros-noetic-gazebo-ros-control ros-noetic-joint-state-controller ros-noetic-effort-controllers ros-noetic-joint-trajectory-controller
```

3. Install yaml-cpp

```bash
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp && mkdir build && cd build
cmake -DYAML_BUILD_SHARED_LIBS=on .. && make
sudo make install
sudo ldconfig
```

4. Install ROS environment (One-click installation)
    - https://fishros.org.cn/forum/topic/20/小鱼的一键安装系列?lang=zh-CN

```bash
wget http://fishros.com/install -O fishros && . fishros
```

5. Clone the code

```bash
git clone https://github.com/FFTAI/wiki-grx-gazebo.git
```

6. Build the project

```bash
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3
catkin build
source devel/setup.bash
```

7. Running: Open a new terminal, launch the gazebo simulation environment

```bash
source devel/setup.bash
roslaunch rl_sim gazebo_<ROBOT>.launch
```

\<ROBOT\> can be `gr1t1` and `gr1t2`.

8. Control:
    - Press **\<Enter\>** to toggle simulation start/stop.
    - **w** and **s** controls x-axis, **a** and **d** controls yaw, and **j** and **l** controls y-axis.
    - Press **\<Space\>** to sets all control commands to zero.
    - Press **t** switch to RL stand mode. Press **y** switch to RL walk mode.
    - If robot falls down, press **R** to reset Gazebo environment.

## Thanks

Thanks to the following repositories for providing the code for the GRx robot model in Gazebo:

- https://github.com/fan-ziqi/rl_sar

---

Thank you for your interest in the Fourier Intelligence GRx Robot Repositories.
We hope you find this resource helpful in your robotics projects!
