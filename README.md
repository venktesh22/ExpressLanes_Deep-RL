This project contains the Python 3 code for a deep reinforcement learning (Deep-RL) model for dynamic pricing of express lanes with multiple access locations. It contains a new reinforcement learning (RL) environment for macroscopic simulation of traffic (which we call `gym-meme`) similar to the current RL `gym` environments, and customizes the open-source implementation of VPG and PPO algorithms provided by OpenAI SpinningUp [OpenAI SpinningUp](https://github.com/openai/spinningup) to work with the new environment.

For more details about the algorithms and the RL problem, refer the paper: https://arxiv.org/abs/1909.04760

# Instructions for download
Clone the project or download the zip. The folder structure is similar to the structure provided by OpenAI SpinningUp [OpenAI SpinningUp](https://github.com/openai/spinningup), excluding the files which are not relevant for reproducing the results. The .zip file contains following folders:

+ `data`: this folder stores the output produced after running the Deep-RL algorithms
+ `gym-meme`: It contains following subfolders:

  + `gym_meme/envs/`: contains the file `meme-env.py` where the gym environment functions are defined. Other files are needed for the definition of the environment
  + `gym_meme/envs/components/`: contains python files for relevant component of the network like cell, links, nodes, etc. It also contains a `reader.py` file for file reading
  + `gym_meme/envs/inputs/`: this folder contains the inputs for running a macroscopic simulation and parameters for Deep-RL. The input files are explained later.
  
+ `spinup`: this folder contains the customized python files for the VPG and PPO algorithms under subfolders with their respective names

# Installing dependencies
First, we install the custom `gym` environment provided in the .zip file. Navigate to the folder of the .zip file. And use the following command to install the `gym-meme` environment.

`pip install -e .`

Next, we recommend installing all packages present in the `requirements.txt` file:

`pip install -r requirements.txt`

# Inputs

The inputs are located inside `/gym-meme/gym_meme/envs/inputs/` folder. Each subfolder contains input files for a different network. The input files for each network are as follows:

+ `Link.txt`: Contains information for each link including its head and tail nodes, fundamental diagram parameters, and the initial number of vehicles on the link at the beginning of the simulation (if an estimate is available). The class of a link is a string having one of the following values: (a) INLINK: a link where vehicles enter the network, (b) ONRAMP: a link connecting general purpose lanes to managed lanes, (c) OFFRAMP: a link connecting managed lanes to general purpose lanes, (d) ML: a link on the managed lane, (e) GPL: a link on the general purpose lane, and (f) OUTLINK: a link where vehicles exit the network

+ `ODDemandProfile.txt`: A file containing the demand information. Includes the start node, the end node, the start time, the end time, and the demand in vehicles per hour units departing from the start node to the end node during a time interval between the start and the end time

+ `VOT.txt`: Contains the discrete value of time (VOT) distribution. Includes two columns, the value of time and the proportion of vehicles with that VOT in the population.

+ `Parameters.txt`: Contains parameters for running the simulation including:
  + networkName: name of the network
  + Simulation_time(seconds): total duration of simulation
  + simulation_update_Step(seconds): time step duration for updating traffic flow parameters using macroscopic cell transmission model 
  + toll_update_step(seconds): frequency of update of tolls
  + demand_factor: scaling factor for demand
  + min_speed_limit_ML(mph): desired minimum speed limit on the managed lane. It is used to calculate the `%-violation` statistic
  + lane_choice_routes(binary/DR): whether a traveler compares utility over binary routes or decision routes for making a lane choice decision at any diverge
  + lane_choice_stochastic: a boolean variable that is true if the lane choices are to be modeled as stochastic
  + stddev_demand: standard deviation in demand when sampled from a Gaussian distribution

# Running a sample code
To run the vpg and ppo algorithm, we use the algorithm files `vpg.py` and `ppo.py`. The relevant arguments to these python files include:

* `--env`: name of the gym environment. For our experiments it is `meme-v0`.
* `--hid`: number of nodes in each hidden layer in the policy neural network. Default 64.
* `--l`: number of hidden layers. Default 2.
* `--gamma`: Value of $\gamma$ used for generalized advantage estimation. Default 0.99
* `--seed`: Unique value of the random seed.
* `--steps`: Number of action steps to be taken in one iteration
* `--epochs`: Number of iterations
* `--exp_name`: Name of the network that should be tested (SESE, DESE, LBJ, or FullMopac)
* `--objective`: The optimization objective. Options: RevMax, TSTTMin, Other. The 'Other' objective handles the case of multiobjective optimization
* `--jahThresh`: the value of JAH threshold above which the rewards are penalized

For example, the following command runs the VPG algorithm for the LBJ network. It runs 100 iterations where in each iteration, 10 trajectories are sampled (for a two-hours simulation with tolls updating every 5 minutes there are 24 actions per trajectory, thus steps = 24*10).

`python spinup/algos/vpg/vpg.py --env meme-v0 --steps 240 --epochs 100 --exp_name LBJ --seed 200 --objective RevMax`

# Output
The output of the run is stored in the `data` folder inside the network folder. It contains the following files:

+ `progress.txt` file which shows the variation of different statistics with different iterations
+ `tollProfile.txt`: file showing the variation of tolls for the best-found toll profile given the optimization objective
+ `GPDensities.txt`(`HOTDensities.txt`): file containing the time-space diagram values for the best-found toll profile on the general purpose lane (managed lane)
+ `config.json`: stores the configuration used for Deep-RL training
+ `simple_save/`: saves the policy parameters at the end of simulation. More info on [OpenAI SpinningUp](https://github.com/openai/spinningup)

# Questions
For any questions or issues with the code, please contact venktesh at utexas dot edu.

