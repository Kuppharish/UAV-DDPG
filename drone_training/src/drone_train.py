import tensorflow as tf
import numpy as np
#import gym
#from gym import wrappers
#import time
from replay_buffer import ReplayBuffer
from actor import ActorNetwork
from critic import CriticNetwork
from ou_noise import OUNoise
import time
import os
from subprocess import Popen, PIPE


# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Soft target update param
TAU = 0.001
# Gym environment
#ENV_NAME = 'AttFC_GyroErr-MotorVel_M4_E-v1'
RANDOM_SEED = 1234
EXPLORE = 70
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def trainer(env, outdir, epochs=100, MINIBATCH_SIZE=64, GAMMA = 0.99,epsilon=0.01, min_epsilon=0.01, BUFFER_SIZE=10000, train_indicator=False, render=False):
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:

        
        # configuring environment
        #env = gym.make(ENV_NAME)
        # configuring the random processes
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)
        # info of the environment to pass to the agent
        state_dim = env.observation_space
        action_dim = env.action_space
        action_bound = np.float64(1) # I choose this number since the mountain continuos does not have a boundary
        # Creating agent

        # FOR the RNN
        #tf.contrib.rnn.core_rnn_cell.BasicLSTMCell from https://github.com/tensorflow/tensorflow/issues/8771
        #cell = tf.contrib.rnn.BasicLSTMCell(num_units=300,state_is_tuple=True, reuse = None)
        #cell_target = tf.contrib.rnn.BasicLSTMCell(num_units=300,state_is_tuple=True, reuse = None)
        ruido = OUNoise(action_dim, mu = 0.4) # this is the Ornstein-Uhlenbeck Noise
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU, outdir)
        critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(), outdir)


        #sess.run(tf.global_variables_initializer())
        
        # Initialize target network weights
        actor.update_target_network()
        critic.update_target_network()
        # Initialize replay memory
        replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
        replay_buffer.load()

        #goal = 0
        max_state = -1.   
        try:
            critic.recover_critic()
            actor.recover_actor()
            print('********************************')
            print('models restored succesfully')
            print('********************************')
        except Exception as e :
            print('********************************')
            print(e)
            print('********************************')
        #critic.recover_critic()
        #actor.recover_actor()        
        
        for i in range(epochs):
            state = env.reset()
            #state = np.hstack(state)
            ep_reward = 0
            ep_ave_max_q = 0
            done = False
            step = 0
            max_state_episode = -1
            epsilon-=epsilon/EXPLORE
            if epsilon < min_epsilon:
                epsilon=min_epsilon
            while (not done):

                if render:
                    env.render()
                    
                #print('step', step)
                # 1. get action with actor, and add noise

                np.set_printoptions(precision=4)
                # remove comment if you want to see a step by step update
                #print(step,'a',action_original, action,'s', state[0], 'max state', max_state_episode)

                # 2. take action, see next state and reward :
                action_original = actor.predict(np.reshape(state, (1, actor.s_dim)))  # + (10. / (10. + i))* np.random.randn(1)
                action = action_original #+ max(epsilon, 0) * ruido.noise()
                '''
                for j in range(action.shape[1]):
                    if abs(action[0,j]) > 1:
                        act=action[0,j]
                        action[0,j]=act/abs(act)
                    else:
                        continue
                '''
                action = np.reshape(action, (actor.a_dim,))
                next_state, reward, done, info = env.step(action)
                if train_indicator:
                    # 3. Save in replay buffer:
                    replay_buffer.add(np.reshape(state, (actor.s_dim,)), np.reshape(action, (actor.a_dim,)), reward,
                                      done, np.reshape(next_state, (actor.s_dim,)))

                    # Keep adding experience to the memory until
                    # there are at least minibatch size samples
                    if replay_buffer.size() > MINIBATCH_SIZE:

                        # 4. sample random minibatch of transitions: 
                        s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

                        # Calculate targets
                        
                        # 5. Train critic Network (states,actions, R + gamma* V(s', a')): 
                        # 5.1 Get critic prediction = V(s', a')
                        # the a' is obtained using the actor prediction! or in other words : a' = actor(s')
                        target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch),20)

                        # 5.2 get y_t where: 
                        y_i = []
                        for k in range(MINIBATCH_SIZE):
                            if t_batch[k]:
                                y_i.append(r_batch[k])
                            else:
                                y_i.append(r_batch[k] + GAMMA * target_q[k])

                        
                        # 5.3 Train Critic! 
                        predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)), 20)
                        

                        ep_ave_max_q += np.amax(predicted_q_value)
                        
                        # 6 Compute Critic gradient (depends on states and actions)
                        # 6.1 therefore I first need to calculate the actions the current actor would take.
                        a_outs = actor.predict(s_batch)
                        # 6.2 I calculate the gradients 
                        grads = critic.action_gradients(s_batch, a_outs, 20)
                        c = np.array(grads)
                        #print(c.shape)
                        #print('...')
                        #print('...',c[0].shape)
                        #print('...')
                        actor.train(s_batch, grads[0])

                        # Update target networks
                        actor.update_target_network()
                        critic.update_target_network()
                state=next_state
                if next_state[0] > max_state_episode:
                    max_state_episode = next_state[0]

                ep_reward = ep_reward + reward
                step +=1
            

            if max_state_episode > max_state:
                max_state = max_state_episode
            
            print('th',i+1,'Step', step,'Reward:',ep_reward,'Pos', next_state[0], next_state[1],'epsilon', epsilon)
            print('*************************')
            print('now we save the model')
            critic.save_critic()
            actor.save_actor()
            print('model saved succesfuly')
            print('*************************')
            replay_buffer.save()
            #proc = Popen(['rosclean','purge'],stdout=PIPE, stdin=PIPE, stderr=PIPE,universal_newlines=True)
            #out,err = proc.communicate(input="{}\n".format("y"))
            #print('maxmimum state reach', max_state)
            #print('the reward at the end of the episode,', reward)
            #print('Efficiency', 100.*((goal)/(i+1.)))
            
        '''
        print('*************************')
        print('now we save the model')
        critic.save_critic()
        actor.save_actor()
        print('model saved succesfuly')
        print('*************************')
        replay_buffer.save()
        #env.close()
        '''
        sess.close()
    return 0
                

'''
if __name__ == '__main__':
    trainer(epochs=500, epsilon = 1., render = False)
'''
