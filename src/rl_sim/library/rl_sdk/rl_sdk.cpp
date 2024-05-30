#include "rl_sdk.hpp"

void RL::InitObservations()
{
    this->obs.lin_vel = torch::tensor({{0.0, 0.0, 0.0}});
    this->obs.ang_vel = torch::tensor({{0.0, 0.0, 0.0}});
    this->obs.gravity_vec = torch::tensor({{0.0, 0.0, -1.0}});
    this->obs.commands = torch::tensor({{0.0, 0.0, 0.0}});
    this->obs.base_quat = torch::tensor({{0.0, 0.0, 0.0, 1.0}});
    this->obs.dof_pos = this->params.default_dof_pos;
    this->obs.dof_vel = torch::zeros({1, this->params.num_of_dofs});
    this->obs.actions = torch::zeros({1, this->params.num_of_dofs});
}

void RL::InitOutputs()
{
    this->output_torques = torch::zeros({1, this->params.num_of_dofs});
    this->output_dof_pos = this->params.default_dof_pos;
}

void RL::InitControl()
{
    this->control.control_state = STATE_WAITING;
    this->control.x = 0.0;
    this->control.y = 0.0;
    this->control.yaw = 0.0;
}

torch::Tensor RL::ComputeTorques(torch::Tensor actions)
{
    torch::Tensor actions_scaled = actions * this->params.action_scale;
    torch::Tensor output_torques = this->params.rl_kp * (actions_scaled + this->params.default_dof_pos - this->obs.dof_pos) - this->params.rl_kd * this->obs.dof_vel;
    return output_torques;
}

torch::Tensor RL::ComputePosition(torch::Tensor actions)
{
    torch::Tensor actions_scaled = actions * this->params.action_scale;
    return actions_scaled + this->params.default_dof_pos;
}

torch::Tensor RL::QuatRotateInverse(torch::Tensor q, torch::Tensor v)
{
    c10::IntArrayRef shape = q.sizes();
    torch::Tensor q_w = q.index({torch::indexing::Slice(), -1});
    torch::Tensor q_vec = q.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    torch::Tensor a = v * (2.0 * torch::pow(q_w, 2) - 1.0).unsqueeze(-1);
    torch::Tensor b = torch::cross(q_vec, v, -1) * q_w.unsqueeze(-1) * 2.0;
    torch::Tensor c = q_vec * torch::bmm(q_vec.view({shape[0], 1, 3}), v.view({shape[0], 3, 1})).squeeze(-1) * 2.0;
    return a - b + c;
}

void RL::StateController(const RobotState<double> *state, RobotCommand<double> *command)
{
    static RobotState<double> start_state;
    static RobotState<double> now_state;
    static float getup_percent = 0.0;
    static float getdown_percent = 0.0;

    // waiting
    if(this->running_state == STATE_WAITING)
    {
        for(int i = 0; i < this->params.num_of_dofs; ++i)
        {
            command->motor_command.q[i] = state->motor_state.q[i];
        }
        if(this->control.control_state == STATE_POS_GETUP)
        {
            this->control.control_state = STATE_WAITING;
            getup_percent = 0.0;
            for(int i = 0; i < this->params.num_of_dofs; ++i)
            {
                now_state.motor_state.q[i] = state->motor_state.q[i];
                start_state.motor_state.q[i] = now_state.motor_state.q[i];
            }
            this->running_state = STATE_POS_GETUP;
            std::cout << std::endl << LOGGER::INFO << "Switching to STATE_POS_GETUP" << std::endl;
        }
    }
    // stand up (position control)
    else if(this->running_state == STATE_POS_GETUP)
    {
        if(getup_percent < 1.0)
        {
            getup_percent += 1 / 500.0;
            getup_percent = getup_percent > 1.0 ? 1.0 : getup_percent;
            for(int i = 0; i < this->params.num_of_dofs; ++i)
            {
                command->motor_command.q[i] = (1 - getup_percent) * now_state.motor_state.q[i] + getup_percent * this->params.default_dof_pos[0][i].item<double>();
                command->motor_command.dq[i] = 0;
                command->motor_command.kp[i] = this->params.fixed_kp[0][i].item<double>();
                command->motor_command.kd[i] = this->params.fixed_kd[0][i].item<double>();
                command->motor_command.tau[i] = 0;
            }
            std::cout << LOGGER::INFO << "Getting up " << std::fixed << std::setprecision(2) << getup_percent * 100.0 << "%\r";
        }
        if(this->control.control_state == STATE_RL_INIT)
        {
            this->control.control_state = STATE_WAITING;
            this->running_state = STATE_RL_INIT;
            std::cout << std::endl << LOGGER::INFO << "Switching to STATE_RL_INIT" << std::endl;
        }
        else if(this->control.control_state == STATE_POS_GETDOWN)
        {
            this->control.control_state = STATE_WAITING;
            getdown_percent = 0.0;
            for(int i = 0; i < this->params.num_of_dofs; ++i)
            {
                now_state.motor_state.q[i] = state->motor_state.q[i];
            }
            this->running_state = STATE_POS_GETDOWN;
            std::cout << std::endl << LOGGER::INFO << "Switching to STATE_POS_GETDOWN" << std::endl;
        }
    }
    // init obs and start rl loop
    else if(this->running_state == STATE_RL_INIT)
    {
        if(getup_percent == 1)
        {
            this->InitObservations();
            this->InitOutputs();
            this->InitControl();
            this->running_state = STATE_RL_RUNNING;
            std::cout << std::endl << LOGGER::INFO << "Switching to STATE_RL_RUNNING" << std::endl;
        }
    }
    // rl loop
    else if(this->running_state == STATE_RL_RUNNING)
    {
        std::cout << LOGGER::INFO << "RL Controller x:" << this->control.x << " y:" << this->control.y << " yaw:" << this->control.yaw << "          \r";
        for(int i = 0; i < this->params.num_of_dofs; ++i)
        {
            command->motor_command.q[i] =  this->output_dof_pos[0][i].item<double>();
            command->motor_command.dq[i] = 0;
            command->motor_command.kp[i] = this->params.rl_kp[0][i].item<double>();
            command->motor_command.kd[i] = this->params.rl_kd[0][i].item<double>();
            command->motor_command.tau[i] = 0;
        }
        if(this->control.control_state == STATE_POS_GETDOWN)
        {
            this->control.control_state = STATE_WAITING;
            getdown_percent = 0.0;
            for(int i = 0; i < this->params.num_of_dofs; ++i)
            {
                now_state.motor_state.q[i] = state->motor_state.q[i];
            }
            this->running_state = STATE_POS_GETDOWN;
            std::cout << std::endl << LOGGER::INFO << "Switching to STATE_POS_GETDOWN" << std::endl;
        }
        else if(this->control.control_state == STATE_POS_GETUP)
        {
            this->control.control_state = STATE_WAITING;
            getup_percent = 0.0;
            for(int i = 0; i < this->params.num_of_dofs; ++i)
            {
                now_state.motor_state.q[i] = state->motor_state.q[i];
            }
            this->running_state = STATE_POS_GETUP;
            std::cout << std::endl << LOGGER::INFO << "Switching to STATE_POS_GETUP" << std::endl;
        }
    }
    // get down (position control)
    else if(this->running_state == STATE_POS_GETDOWN)
    {
        if(getdown_percent < 1.0)
        {
            getdown_percent += 1 / 500.0;
            getdown_percent = getdown_percent > 1.0 ? 1.0 : getdown_percent;
            for(int i = 0; i < this->params.num_of_dofs; ++i)
            {
                command->motor_command.q[i] = (1 - getdown_percent) * now_state.motor_state.q[i] + getdown_percent * start_state.motor_state.q[i];
                command->motor_command.dq[i] = 0;
                command->motor_command.kp[i] = this->params.fixed_kp[0][i].item<double>();
                command->motor_command.kd[i] = this->params.fixed_kd[0][i].item<double>();
                command->motor_command.tau[i] = 0;
            }
            std::cout << LOGGER::INFO << "Getting down " << std::fixed << std::setprecision(2) << getdown_percent * 100.0 << "%\r";
        }
        if(getdown_percent == 1)
        {
            this->InitObservations();
            this->InitOutputs();
            this->InitControl();
            this->running_state = STATE_WAITING;
            std::cout << std::endl << LOGGER::INFO << "Switching to STATE_WAITING" << std::endl;
        }
    }
}

void RL::TorqueProtect(torch::Tensor origin_output_torques)
{
    std::vector<int> out_of_range_indices;
    std::vector<double> out_of_range_values;
    for(int i = 0; i < origin_output_torques.size(1); ++i)
    {
        double torque_value = origin_output_torques[0][i].item<double>();
        double limit_lower = -this->params.torque_limits[0][i].item<double>();
        double limit_upper = this->params.torque_limits[0][i].item<double>();

        if(torque_value < limit_lower || torque_value > limit_upper)
        {
            out_of_range_indices.push_back(i);
            out_of_range_values.push_back(torque_value);
        }
    }
    if(!out_of_range_indices.empty())
    {
        for(int i = 0; i < out_of_range_indices.size(); ++i)
        {
            int index = out_of_range_indices[i];
            double value = out_of_range_values[i];
            double limit_lower = -this->params.torque_limits[0][index].item<double>();
            double limit_upper = this->params.torque_limits[0][index].item<double>();

            std::cout << LOGGER::WARNING << "Torque(" << index + 1 << ")=" << value << " out of range(" << limit_lower << ", " << limit_upper << ")" << std::endl;
        }
        // Just a reminder, no protection
        // this->control.control_state = STATE_POS_GETDOWN;
        // std::cout << LOGGER::INFO << "Switching to STATE_POS_GETDOWN"<< std::endl;
    }
}

#include <termios.h>
#include <sys/ioctl.h>
static bool kbhit()
{
    termios term;
    tcgetattr(0, &term);
    
    termios term2 = term;
    term2.c_lflag &= ~ICANON;
    tcsetattr(0, TCSANOW, &term2);
    
    int byteswaiting;
    ioctl(0, FIONREAD, &byteswaiting);
    
    tcsetattr(0, TCSANOW, &term);
    
    return byteswaiting > 0;
}

void RL::KeyboardInterface()
{
    if(kbhit())
    {
        int c = fgetc(stdin);
        switch(c)
        {
            case '0': this->control.control_state = STATE_POS_GETUP; break;
            case 'p': this->control.control_state = STATE_RL_INIT; break;
            case '1': this->control.control_state = STATE_POS_GETDOWN; break;
            case 'q': break;
            case 'w': this->control.x += 0.1; break;
            case 's': this->control.x -= 0.1; break;
            case 'a': this->control.yaw += 0.1; break;
            case 'd': this->control.yaw -= 0.1; break;
            case 'i': break;
            case 'k': break;
            case 'j': this->control.y += 0.1; break;
            case 'l': this->control.y -= 0.1; break;
            case ' ': this->control.x = 0; this->control.y = 0; this->control.yaw = 0; break;
            case 'r': this->control.control_state = STATE_RESET_SIMULATION; break;
            default: break;
        }
    }
}

template<typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node& node)
{
    std::vector<T> values;
    for(const auto& val : node)
    {
        values.push_back(val.as<T>());
    }
    return values;
}

void RL::ReadYaml(std::string robot_name)
{
	YAML::Node config;
	try
	{
		config = YAML::LoadFile(CONFIG_PATH)[robot_name];
	} catch(YAML::BadFile &e)
	{

		std::cout << LOGGER::ERROR << "The file '" << CONFIG_PATH << "' does not exist" << std::endl;
		return;
	}

    this->params.model_name = config["model_name"].as<std::string>();
    this->params.dt = config["dt"].as<double>();
    this->params.decimation = config["decimation"].as<int>();
    this->params.num_observations = config["num_observations"].as<int>();
    this->params.clip_obs = config["clip_obs"].as<double>();
    this->params.action_scale = config["action_scale"].as<double>();
    this->params.clip_actions_upper = torch::tensor(ReadVectorFromYaml<double>(config["clip_actions_upper"])).view({1, -1});
    this->params.clip_actions_lower = torch::tensor(ReadVectorFromYaml<double>(config["clip_actions_lower"])).view({1, -1});
    this->params.num_of_dofs = config["num_of_dofs"].as<int>();
    this->params.lin_vel_scale = config["lin_vel_scale"].as<double>();
    this->params.ang_vel_scale = config["ang_vel_scale"].as<double>();
    this->params.dof_pos_scale = config["dof_pos_scale"].as<double>();
    this->params.dof_vel_scale = config["dof_vel_scale"].as<double>();
    // this->params.commands_scale = torch::tensor(ReadVectorFromYaml<double>(config["commands_scale"])).view({1, -1});
    this->params.commands_scale = torch::tensor({this->params.lin_vel_scale, this->params.lin_vel_scale, this->params.ang_vel_scale});
    this->params.rl_kp = torch::tensor(ReadVectorFromYaml<double>(config["rl_kp"])).view({1, -1});
    this->params.rl_kd = torch::tensor(ReadVectorFromYaml<double>(config["rl_kd"])).view({1, -1});
    this->params.fixed_kp = torch::tensor(ReadVectorFromYaml<double>(config["fixed_kp"])).view({1, -1});
    this->params.fixed_kd = torch::tensor(ReadVectorFromYaml<double>(config["fixed_kd"])).view({1, -1});
    this->params.torque_limits = torch::tensor(ReadVectorFromYaml<double>(config["torque_limits"])).view({1, -1});
    this->params.default_dof_pos = torch::tensor(ReadVectorFromYaml<double>(config["default_dof_pos"])).view({1, -1});
    this->params.joint_controller_names = ReadVectorFromYaml<std::string>(config["joint_controller_names"]);
}
