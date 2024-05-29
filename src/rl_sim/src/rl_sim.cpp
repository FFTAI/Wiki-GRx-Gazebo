#include "../include/rl_sim.hpp"

RL_Sim::RL_Sim()
{
    ros::NodeHandle nh;

    // read params from yaml
    nh.param<std::string>("robot_name", this->robot_name, "");
    this->ReadYaml(this->robot_name);

    // Due to the fact that the robot_state_publisher sorts the joint names alphabetically,
    // the mapping table is established according to the order defined in the YAML file
    std::vector<std::string> sorted_joint_controller_names = this->params.joint_controller_names;
    std::sort(sorted_joint_controller_names.begin(), sorted_joint_controller_names.end());
    for(size_t i = 0; i < this->params.joint_controller_names.size(); ++i)
    {
        this->sorted_to_original_index[sorted_joint_controller_names[i]] = i;
    }
    this->mapped_joint_positions = std::vector<double>(this->params.num_of_dofs, 0.0);
    this->mapped_joint_velocities = std::vector<double>(this->params.num_of_dofs, 0.0);
    this->mapped_joint_efforts = std::vector<double>(this->params.num_of_dofs, 0.0);

    // init
    torch::autograd::GradMode::set_enabled(false);
    this->joint_publishers_commands.resize(this->params.num_of_dofs);
    this->InitObservations();
    this->InitOutputs();
    this->InitControl();

    // model
    std::string model_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/models/" + this->robot_name + "/" + this->params.model_name;
    this->model = torch::jit::load(model_path);

    // publisher
    nh.param<std::string>("ros_namespace", this->ros_namespace, "");
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->joint_publishers[this->params.joint_controller_names[i]] = nh.advertise<robot_msgs::MotorCommand>(
            this->ros_namespace + this->params.joint_controller_names[i] + "/command", 10);
    }

    // subscriber
    this->cmd_vel_subscriber = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, &RL_Sim::CmdvelCallback, this);
    this->model_state_subscriber = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 10, &RL_Sim::ModelStatesCallback, this);
    this->joint_state_subscriber = nh.subscribe<sensor_msgs::JointState>(this->ros_namespace + "joint_states", 10, &RL_Sim::JointStatesCallback, this);

    // service
    this->gazebo_reset_client = nh.serviceClient<std_srvs::Empty>("/gazebo/reset_simulation");
    
    // loop
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05 ,    boost::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_control  = std::make_shared<LoopFunc>("loop_control" , 0.002,    boost::bind(&RL_Sim::RobotControl     , this));
    this->loop_rl       = std::make_shared<LoopFunc>("loop_rl"      , 0.02 ,    boost::bind(&RL_Sim::RunModel         , this));
    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();
}

RL_Sim::~RL_Sim()
{
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::GetState(RobotState<double> *state)
{
    state->imu.quaternion[3] = this->pose.orientation.w;
    state->imu.quaternion[0] = this->pose.orientation.x;
    state->imu.quaternion[1] = this->pose.orientation.y;
    state->imu.quaternion[2] = this->pose.orientation.z;

    state->imu.gyroscope[0] = this->vel.angular.x;
    state->imu.gyroscope[1] = this->vel.angular.y;
    state->imu.gyroscope[2] = this->vel.angular.z;

    // state->imu.accelerometer

    for(int i = 0; i < this->params.num_of_dofs; ++i)
    {
        state->motor_state.q[i] = this->mapped_joint_positions[i];
        state->motor_state.dq[i] = this->mapped_joint_velocities[i];
        state->motor_state.tauEst[i] = this->mapped_joint_efforts[i];
    }
}

void RL_Sim::SetCommand(const RobotCommand<double> *command)
{
    for(int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->joint_publishers_commands[i].q = command->motor_command.q[i];
        this->joint_publishers_commands[i].dq = command->motor_command.dq[i];
        this->joint_publishers_commands[i].kp = command->motor_command.kp[i];
        this->joint_publishers_commands[i].kd = command->motor_command.kd[i];
        this->joint_publishers_commands[i].tau = command->motor_command.tau[i];
    }
    
    for(int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->joint_publishers[this->params.joint_controller_names[i]].publish(this->joint_publishers_commands[i]);
    }
}

void RL_Sim::RobotControl()
{
    this->motiontime++;

    if(this->control.control_state == STATE_RESET_SIMULATION)
    {
        this->control.control_state = STATE_WAITING;
        std_srvs::Empty srv;
        this->gazebo_reset_client.call(srv);
    }

    this->GetState(&this->robot_state);
    this->StateController(&this->robot_state, &this->robot_command);
    this->SetCommand(&this->robot_command);
}

void RL_Sim::ModelStatesCallback(const gazebo_msgs::ModelStates::ConstPtr &msg)
{
    this->vel = msg->twist[2];
    this->pose = msg->pose[2];
}

void RL_Sim::CmdvelCallback(const geometry_msgs::Twist::ConstPtr &msg)
{
    this->cmd_vel = *msg;
}

void RL_Sim::MapData(const std::vector<double>& source_data, std::vector<double>& target_data)
{
    for(size_t i = 0; i < source_data.size(); ++i)
    {
        target_data[i] = source_data[this->sorted_to_original_index[this->params.joint_controller_names[i]]];
    }
}

void RL_Sim::JointStatesCallback(const sensor_msgs::JointState::ConstPtr &msg)
{
    this->MapData(msg->position, this->mapped_joint_positions);
    this->MapData(msg->velocity, this->mapped_joint_velocities);
    this->MapData(msg->effort, this->mapped_joint_efforts);
}

void RL_Sim::RunModel()
{
    if(running_state == STATE_RL_RUNNING)
    {
        // this->obs.lin_vel = torch::tensor({{this->vel.linear.x, this->vel.linear.y, this->vel.linear.z}});
        this->obs.ang_vel = torch::tensor(this->robot_state.imu.gyroscope).unsqueeze(0);
        // this->obs.commands = torch::tensor({{this->cmd_vel.linear.x, this->cmd_vel.linear.y, this->cmd_vel.angular.z}});
        this->obs.commands = torch::tensor({{this->control.x, this->control.y, this->control.yaw}});
        this->obs.base_quat = torch::tensor(this->robot_state.imu.quaternion).unsqueeze(0);
        this->obs.dof_pos = torch::tensor(this->robot_state.motor_state.q).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);
        this->obs.dof_vel = torch::tensor(this->robot_state.motor_state.dq).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);

        torch::Tensor clamped_actions = this->Forward();

        this->obs.actions = clamped_actions;

        torch::Tensor origin_output_torques = this->ComputeTorques(this->obs.actions);

        this->TorqueProtect(origin_output_torques);

        this->output_torques = torch::clamp(origin_output_torques, -(this->params.torque_limits), this->params.torque_limits);
        this->output_dof_pos = this->ComputePosition(this->obs.actions);
    }
}

torch::Tensor RL_Sim::ComputeObservation()
{
    torch::Tensor obs = torch::cat({// this->obs.lin_vel * this->params.lin_vel_scale,
                                    this->QuatRotateInverse(this->obs.base_quat, this->obs.ang_vel) * this->params.ang_vel_scale,
                                    // this->obs.ang_vel * this->params.ang_vel_scale, // TODO
                                    this->QuatRotateInverse(this->obs.base_quat, this->obs.gravity_vec),
                                    this->obs.commands * this->params.commands_scale,
                                    (this->obs.dof_pos - this->params.default_dof_pos) * this->params.dof_pos_scale,
                                    this->obs.dof_vel * this->params.dof_vel_scale,
                                    this->obs.actions
                                    },1);
    torch::Tensor clamped_obs = torch::clamp(obs, -this->params.clip_obs, this->params.clip_obs);
    return clamped_obs;
}

torch::Tensor RL_Sim::Forward()
{
    torch::autograd::GradMode::set_enabled(false);

    torch::Tensor clamped_obs = this->ComputeObservation();

    torch::Tensor actions = this->model.forward({clamped_obs}).toTensor();

    torch::Tensor clamped_actions = torch::clamp(actions, this->params.clip_actions_lower, this->params.clip_actions_upper);

    return clamped_actions;
}

void signalHandler(int signum)
{
    ros::shutdown();
    exit(0);
}

int main(int argc, char **argv)
{
    signal(SIGINT, signalHandler);

    ros::init(argc, argv, "rl_sim");

    RL_Sim rl_sim;

    ros::spin();

    return 0;
}
