import sys
import os
import os.path as osp
folderDirectory = osp.abspath(osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__)))))
sys.path.insert(0,folderDirectory)

import numpy as np
import tensorflow as tf
import pandas as pd

import gym
import gym_meme

import time
import time
from spinup.algos.sac import core
from spinup.algos.sac.core import get_vars
from spinup.utils.logx import EpochLogger

import matplotlib
matplotlib.use('PS') # generate postscript output by default
import matplotlib.pyplot as plt

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""
def sac(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=24, num_test_episodes=2, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q1(x, pi(x)).
            ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q2(x, pi(x)).
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. 
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    #custom values for MEME environment
#    steps_per_epoch=1200 #for SESE networks
#    start_steps=3600
    #=====custom code. Say n iterations till which random policy is simulated
    nItr = 2;
    start_steps = steps_per_epoch * nItr;
    update_after = update_every*2;
    #================
    
#    steps_per_epoch=240 #for DESE and LBJ networks
#    start_steps=7200

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi, q1, q2 = actor_critic(x_ph, a_ph, **ac_kwargs)
    
    with tf.variable_scope('main', reuse=True):
        # compose q with pi, for pi-learning
        _, _, _, q1_pi, q2_pi = actor_critic(x_ph, pi, **ac_kwargs)

        # get actions and log probs of actions for next states, for Q-learning
        _, pi_next, logp_pi_next, _, _ = actor_critic(x2_ph, a_ph, **ac_kwargs)
    
    # Target value network
    with tf.variable_scope('target'):
        # target q values, using actions from *current* policy
        _, _, _, q1_targ, q2_targ  = actor_critic(x2_ph, pi_next, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in 
                       ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n')%var_counts)

    # Min Double-Q:
    min_q_pi = tf.minimum(q1_pi, q2_pi)
    min_q_targ = tf.minimum(q1_targ, q2_targ)

#    # Targets for Q and V regression
#    q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*v_targ)
#    v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)
    
    # Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
    q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*(min_q_targ - alpha * logp_pi_next))

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
    value_loss = q1_loss + q2_loss

    # Policy train op 
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, 
                train_pi_op, train_value_op, target_update]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, 
                                outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2})

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

    def test_agent():
#        global sess, mu, pi, q1, q2, q1_pi, q2_pi
        for j in range(num_test_episodes):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                temp_a = get_action(o,True)
#                if j==1:
#                    print("Observation=",o)
#                    print("--before mapping action=",temp_a)
                numpyFromA = np.array(temp_a)
                numpyFromA = ((numpyFromA+1.0)*(env.state.tollMax-env.state.tollMin)/2.0)+ env.state.tollMin
                temp_a = np.ndarray.tolist(numpyFromA)
                
                o, r, d, _ = test_env.step(temp_a)
                # Take deterministic actions at test time 
#                o, r, d, _ = test_env.step(get_action(o, True)) #orig
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs
    
    maxRev=float("-inf") #negative infinity in the beginning
    #maxRevActionSeq=[]
    maxRevTSTT=0
    maxRevRevenue=0
    maxRevThroughput=0
    maxRevJAH=0
    maxRevRemVeh=0
    maxRevJAH2=0
    maxRevRMSE_MLvio=0
    maxRevPerTimeVio=0
    maxRevHOTDensity=pd.DataFrame()
    maxRevGPDensity=pd.DataFrame()
    maxtdJAHMax=0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        #======custom code======#
        #scale the noise hyperbolically to allow exploration in earlier iteration and less exploration latter
        epochId = 1+(int)(t/steps_per_epoch)
        par = 10; #scaling parameter for scaling the noise using formula below
        act_noise = (par)/(par+epochId-1);
        
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()
            #a = [a] #custom value for MEME environments

        #before stepping the environment, scale the tolls (for ML pricing problem)
        #we need to scale the sampled values of action from (-1,1) to our choices of toll coz they were sampled from tanh activation mu
        numpyFromA = np.array(a)
        numpyFromA = ((numpyFromA+1.0)*(env.state.tollMax-env.state.tollMin)/2.0)+ env.state.tollMin
        a = np.ndarray.tolist(numpyFromA)
        
        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            #get other stats and store them too
            otherStats = env.getAllOtherStats()
#            if np.any(np.isnan(np.array(otherStats))):
#                sys.exit("Nan found in statistics! Error")
            logger.store(EpTSTT=otherStats[0], EpRevenue=otherStats[1], 
                         EpThroughput=otherStats[2], EpJAH=otherStats[3],
                         EpRemVeh=otherStats[4], EpJAH2= otherStats[5],
                         EpMLViolRMSE=otherStats[6], EpPerTimeVio=otherStats[7],
                         EptdJAHMax=otherStats[8])
            #determine max rev profile
            if ep_ret> maxRev:
                maxRev=ep_ret
                maxRevActionSeq = env.state.tollProfile
                maxRevTSTT=otherStats[0]; maxRevRevenue=otherStats[1]; 
                maxRevThroughput=otherStats[2]
                maxRevJAH=otherStats[3]
                maxRevRemVeh=otherStats[4]
                maxRevJAH2= otherStats[5]
                maxRevRMSE_MLvio = otherStats[6]
                maxRevPerTimeVio= otherStats[7]
                maxRevHOTDensity = env.getHOTDensityData()
                maxRevGPDensity = env.getGPDensityData()
                maxtdJAHMax = otherStats[8]
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done'],
                            }
                outs = sess.run(step_ops, feed_dict)
                logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                             Q1Vals=outs[3], Q2Vals=outs[4], LogPi=outs[5])

        # OLD--End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch
        
#        # End of epoch wrap-up
#        if (t+1) % steps_per_epoch == 0:
#            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            
            #extra info for MEME environment
            logger.log_tabular('EpTSTT', average_only=True)
            logger.log_tabular('EpRevenue', average_only=True)
            logger.log_tabular('EpThroughput', average_only=True)
            logger.log_tabular('EpJAH', average_only=True)
            logger.log_tabular('EpRemVeh', average_only=True)
            logger.log_tabular('EpJAH2', average_only=True)
            logger.log_tabular('EpMLViolRMSE', average_only=True)
            logger.log_tabular('EpPerTimeVio', average_only=True)
            logger.log_tabular('EptdJAHMax', average_only=True)
            
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True) 
            logger.log_tabular('Q2Vals', with_min_and_max=True) 
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
    print("Max cumulative reward obtained= %f "% maxRev)
    print("Corresponding revenue($)= %f, TSTT(hrs)= %f, Throughput(veh)=%f, JAHstat= %f, remaining vehicles= %f, JAHstat2=%f, RMSEML_vio=%f, percentTimeViolated(%%)=%f" % 
          (maxRevRevenue,maxRevTSTT,maxRevThroughput,maxRevJAH,maxRevRemVeh,maxRevJAH2, maxRevRMSE_MLvio, maxRevPerTimeVio))
    outputVector = [maxRev, maxRevRevenue,maxRevTSTT,maxRevThroughput,maxRevJAH,maxRevRemVeh,maxRevJAH2, maxRevRMSE_MLvio, maxRevPerTimeVio]
    #print("\n===Max rev action sequence is\n",maxRevActionSeq)
    exportTollProfile(maxRevActionSeq, logger_kwargs, outputVector)
    exportDensityData(maxRevHOTDensity, maxRevGPDensity, logger_kwargs)

def exportTollProfile(tollProfile, logger_kwargs, outputVec):
    colNames=[]
    for i in range(len(tollProfile[0])):
        colNames.append("Agent"+str(i+1)+"_toll")
    data = pd.DataFrame(np.array(tollProfile), columns=colNames)
    data['Timestep']=data.index.values*60
    
    exportDir = logger_kwargs['output_dir']
    fileName = exportDir+"/tollProfile.txt"
    
    export_csv = data.to_csv (fileName, sep="\t", index = None, header=True)
    
    f=open(fileName,"a") #reopen in append mode
    f.write("\n\n")
    f.write("Max cumulative reward obtained= %f " % (outputVec[0]))
    f.write("Corresponding revenue($)= %f, TSTT(hrs)= %f, Throughput(veh)=%f, JAHstat= %f, remaining vehicles= %f, JAHstat2=%f, RMSEML_vio=%f, percentTimeViolated(%%)=%f" % 
          (outputVec[1],outputVec[2],outputVec[3],outputVec[4],outputVec[5],outputVec[6], outputVec[7], outputVec[8]))
    f.close()

def exportDensityData(maxRevHOTDensity, maxRevGPDensity, logger_kwargs):
    exportDir = logger_kwargs['output_dir']
    
    fileName = exportDir+"/HOTDensities.txt"
    export_csv = maxRevHOTDensity.to_csv (fileName, sep="\t", header=True, index_label="CellID")
    
    fileName = exportDir+"/GPDensities.txt"
    export_csv = maxRevGPDensity.to_csv (fileName, sep="\t", header=True, index_label="CellID")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--updateEvery', '-ue', type=int, default=0)
    
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--objective', type=str, default='RevMax') #for dynamic pricing
    parser.add_argument('--jahThresh', type=int, default=100000) #default is a high number
    args = parser.parse_args()

#    from spinup.utils.run_utils import setup_logger_kwargs
#    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    from spinup.utils.run_utils import setup_logger_kwargs
    import time
    currTime = round(time.time())
#    logger_kwargs = setup_logger_kwargs(args.exp_name, str(args.seed)+"_e"+str(args.epochs)+"_st"+str(args.steps)+"_"+"ue"+str(args.updateEvery)+"_"+args.objective+"_"+str(currTime)+"_SAC")
    
    logger_kwargs = setup_logger_kwargs(args.exp_name, str(args.seed)+"_e"+str(args.epochs)+"_st"+str(args.steps)+"_"+"ue"+str(args.updateEvery)+"_JAHPenalty"+str(args.jahThresh)+"_"+str(currTime)+"_SAC")

    sac(lambda : gym.make(args.env, netname=args.exp_name, objective=args.objective, seed=args.seed, jahThresh= args.jahThresh), 
        actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs, steps_per_epoch = args.steps, update_every=args.updateEvery,
        logger_kwargs=logger_kwargs)