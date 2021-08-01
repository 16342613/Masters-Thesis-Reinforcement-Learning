# # OpenGym CartPole-v0 with A3C on GPU
# # -----------------------------------
# #
# # A3C implementation with GPU optimizer threads.
# #
# # Made as part of blog series Let's make an A3C, available at
# # https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
# #
# # author: Jaromir Janisch, 2017
#
# import gym
# import numpy as np
# import random
# import tensorflow as tf
# import threading
# import time
# from keras import backend as K
# from keras.layers import *
# from keras.models import *
#
# tf.compat.v1.disable_eager_execution()
#
# # -- constants
# ENV = 'CartPole-v0'
#
# RUN_TIME = 30
# THREADS = 8
# OPTIMIZERS = 2
# THREAD_DELAY = 0.001
#
# GAMMA = 0.99
#
# N_STEP_RETURN = 8
# GAMMA_N = GAMMA ** N_STEP_RETURN
#
# EPS_START = 0.4
# EPS_STOP = .15
# EPS_STEPS = 75000
#
# MIN_BATCH = 32
# LEARNING_RATE = 5e-3
#
# LOSS_V = .5  # v loss coefficient
# LOSS_ENTROPY = .01  # entropy coefficient
#
#
# # ---------
# class Brain:
#     train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
#     lock_queue = threading.Lock()
#
#     def __init__(self):
#         self.session = tf.compat.v1.Session()
#         K.set_session(self.session)
#         K.manual_variable_initialization(True)
#
#         self.model = self._build_model()
#         self.graph = self._build_graph(self.model)
#
#         self.session.run(tf.global_variables_initializer())
#         self.default_graph = tf.get_default_graph()
#
#         self.default_graph.finalize()  # avoid modifications
#
#     def _build_model(self):
#
#         l_input = Input(batch_shape=(None, NUM_STATE))
#         l_dense = Dense(16, activation='relu')(l_input)
#
#         out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
#         out_value = Dense(1, activation='linear')(l_dense)
#
#         model = Model(inputs=[l_input], outputs=[out_actions, out_value])
#         model.make_predict_function()  # have to initialize before threading
#
#         return model
#
#     def _build_graph(self, model):
#         s_t = tf.compat.v1.placeholder(tf.float32, shape=(None, NUM_STATE))
#         a_t = tf.compat.v1.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
#         r_t = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward
#
#         p, v = model(s_t)
#
#         log_prob = tf.compat.v1.log(tf.reduce_sum(p * a_t, axis=1, keepdims=True) + 1e-10)
#         advantage = r_t - v
#
#         loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
#         loss_value = LOSS_V * tf.square(advantage)  # minimize value error
#         entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.compat.v1.log(p + 1e-10), axis=1,
#                                                keepdims=True)  # maximize entropy (regularization)
#
#         loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
#
#         optimizer = tf.optimizers.Adam()  # tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
#         minimize = optimizer.minimize(loss_total)
#
#         return s_t, a_t, r_t, minimize
#
#     def optimize(self):
#         if len(self.train_queue[0]) < MIN_BATCH:
#             time.sleep(0)  # yield
#             return
#
#         with self.lock_queue:
#             if len(self.train_queue[0]) < MIN_BATCH:  # more thread could have passed without lock
#                 return  # we can't yield inside lock
#
#             s, a, r, s_, s_mask = self.train_queue
#             self.train_queue = [[], [], [], [], []]
#
#         s = np.vstack(s)
#         a = np.vstack(a)
#         r = np.vstack(r)
#         s_ = np.vstack(s_)
#         s_mask = np.vstack(s_mask)
#
#         if len(s) > 5 * MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))
#
#         v = self.predict_v(s_)
#         r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state
#
#         s_t, a_t, r_t, minimize = self.graph
#         self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})
#
#     def train_push(self, s, a, r, s_):
#         with self.lock_queue:
#             self.train_queue[0].append(s)
#             self.train_queue[1].append(a)
#             self.train_queue[2].append(r)
#
#             if s_ is None:
#                 self.train_queue[3].append(NONE_STATE)
#                 self.train_queue[4].append(0.)
#             else:
#                 self.train_queue[3].append(s_)
#                 self.train_queue[4].append(1.)
#
#     def predict(self, s):
#         with self.default_graph.as_default():
#             p, v = self.model.predict(s)
#             return p, v
#
#     def predict_p(self, s):
#         with self.default_graph.as_default():
#             p, v = self.model.predict(s)
#             return p
#
#     def predict_v(self, s):
#         with self.default_graph.as_default():
#             p, v = self.model.predict(s)
#             return v
#
#
# # ---------
# frames = 0
#
#
# class Agent:
#     def __init__(self, eps_start, eps_end, eps_steps):
#         self.eps_start = eps_start
#         self.eps_end = eps_end
#         self.eps_steps = eps_steps
#
#         self.memory = []  # used for n_step return
#         self.R = 0.
#
#     def getEpsilon(self):
#         if frames >= self.eps_steps:
#             return self.eps_end
#         else:
#             return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate
#
#     def act(self, s):
#         eps = self.getEpsilon()
#         global frames;
#         frames = frames + 1
#
#         if random.random() < eps:
#             return random.randint(0, NUM_ACTIONS - 1)
#
#         else:
#             s = np.array([s])
#             p = brain.predict_p(s)[0]
#
#             # a = np.argmax(p)
#             a = np.random.choice(NUM_ACTIONS, p=p)
#
#             return a
#
#     def train(self, s, a, r, s_):
#         def get_sample(memory, n):
#             s, a, _, _ = memory[0]
#             _, _, _, s_ = memory[n - 1]
#
#             return s, a, self.R, s_
#
#         a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
#         a_cats[a] = 1
#
#         self.memory.append((s, a_cats, r, s_))
#
#         self.R = (self.R + r * GAMMA_N) / GAMMA
#
#         if s_ is None:
#             while len(self.memory) > 0:
#                 n = len(self.memory)
#                 s, a, r, s_ = get_sample(self.memory, n)
#                 brain.train_push(s, a, r, s_)
#
#                 self.R = (self.R - self.memory[0][2]) / GAMMA
#                 self.memory.pop(0)
#
#             self.R = 0
#
#         if len(self.memory) >= N_STEP_RETURN:
#             s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
#             brain.train_push(s, a, r, s_)
#
#             self.R = self.R - self.memory[0][2]
#             self.memory.pop(0)
#
#
# # possible edge case - if an episode ends in <N steps, the computation is incorrect
#
# # ---------
# class Environment(threading.Thread):
#     stop_signal = False
#
#     def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
#         threading.Thread.__init__(self)
#
#         self.render = render
#         self.env = gym.make(ENV)
#         self.agent = Agent(eps_start, eps_end, eps_steps)
#
#     def runEpisode(self):
#         s = self.env.reset()
#
#         R = 0
#         while True:
#             time.sleep(THREAD_DELAY)  # yield
#
#             if self.render: self.env.render()
#
#             a = self.agent.act(s)
#             s_, r, done, info = self.env.step(a)
#
#             if done:  # terminal state
#                 s_ = None
#
#             self.agent.train(s, a, r, s_)
#
#             s = s_
#             R += r
#
#             if done or self.stop_signal:
#                 break
#
#         print("Total R:", R)
#
#     def run(self):
#         while not self.stop_signal:
#             self.runEpisode()
#
#     def stop(self):
#         self.stop_signal = True
#
#
# # ---------
# class Optimizer(threading.Thread):
#     stop_signal = False
#
#     def __init__(self):
#         threading.Thread.__init__(self)
#
#     def run(self):
#         while not self.stop_signal:
#             brain.optimize()
#
#     def stop(self):
#         self.stop_signal = True
#
#
# # -- main
# env_test = Environment(render=True, eps_start=0., eps_end=0.)
# NUM_STATE = env_test.env.observation_space.shape[0]
# NUM_ACTIONS = env_test.env.action_space.n
# NONE_STATE = np.zeros(NUM_STATE)
#
# brain = Brain()  # brain is global in A3C
#
# envs = [Environment() for i in range(THREADS)]
# opts = [Optimizer() for i in range(OPTIMIZERS)]
#
# for o in opts:
#     o.start()
#
# for e in envs:
#     e.start()
#
# time.sleep(RUN_TIME)
#
# for e in envs:
#     e.stop()
# for e in envs:
#     e.join()
#
# for o in opts:
#     o.stop()
# for o in opts:
#     o.join()
#
# print("Training finished")
# env_test.run()


# Tutorial by www.pylessons.com
# Tutorial written for - Tensorflow 1.15, Keras 2.2.4

# import os
# import random
# import gym
# import pylab
# import numpy as np
# from keras.models import Model, load_model
# from keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
# from tensorflow.keras.optimizers import Adam, RMSprop
# from keras import backend as K
# import cv2
# # import needed for threading
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# import threading
# from threading import Thread, Lock
# import time
#
# # configure Keras and TensorFlow sessions and graph
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# set_session(sess)
# K.set_session(sess)
# graph = tf.get_default_graph()
#
#
# def OurModel(input_shape, action_space, lr):
#     X_input = Input(input_shape)
#
#     # X = Conv2D(32, 8, strides=(4, 4),padding="valid", activation="elu", data_format="channels_first", input_shape=input_shape)(X_input)
#     # X = Conv2D(64, 4, strides=(2, 2),padding="valid", activation="elu", data_format="channels_first")(X)
#     # X = Conv2D(64, 3, strides=(1, 1),padding="valid", activation="elu", data_format="channels_first")(X)
#     X = Flatten(input_shape=input_shape)(X_input)
#
#     X = Dense(512, activation="elu", kernel_initializer='he_uniform')(X)
#     # X = Dense(256, activation="elu", kernel_initializer='he_uniform')(X)
#     # X = Dense(64, activation="elu", kernel_initializer='he_uniform')(X)
#
#     action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X)
#     value = Dense(1, kernel_initializer='he_uniform')(X)
#
#     Actor = Model(inputs=X_input, outputs=action)
#     Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr))
#
#     Critic = Model(inputs=X_input, outputs=value)
#     Critic.compile(loss='mse', optimizer=RMSprop(lr=lr))
#
#     return Actor, Critic
#
#
# class A3CAgent:
#     # Actor-Critic Main Optimization Algorithm
#     def __init__(self, env_name):
#         # Initialization
#         # Environment and PPO parameters
#         self.env_name = env_name
#         self.env = gym.make(env_name)
#         self.action_size = self.env.action_space.n
#         self.EPISODES, self.episode, self.max_average = 10000, 0, -21.0  # specific for pong
#         self.lock = Lock()
#         self.lr = 0.000025
#
#         self.ROWS = 80
#         self.COLS = 80
#         self.REM_STEP = 4
#
#         # Instantiate plot memory
#         self.scores, self.episodes, self.average = [], [], []
#
#         self.Save_Path = 'Models'
#         self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
#
#         if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
#         self.path = '{}_A3C_{}'.format(self.env_name, self.lr)
#         self.Model_name = os.path.join(self.Save_Path, self.path)
#
#         # Create Actor-Critic network model
#         self.Actor, self.Critic = OurModel(input_shape=self.state_size, action_space=self.action_size, lr=self.lr)
#
#         # make predict function to work while multithreading
#         self.Actor._make_predict_function()
#         self.Critic._make_predict_function()
#
#         global graph
#         graph = tf.get_default_graph()
#
#     def act(self, state):
#         # Use the network to predict the next action to take, using the model
#         prediction = self.Actor.predict(state)[0]
#         action = np.random.choice(self.action_size, p=prediction)
#         return action
#
#     def discount_rewards(self, reward):
#         # Compute the gamma-discounted rewards over an episode
#         gamma = 0.99  # discount rate
#         running_add = 0
#         discounted_r = np.zeros_like(reward)
#         for i in reversed(range(0, len(reward))):
#             if reward[i] != 0:  # reset the sum, since this was a game boundary (pong specific!)
#                 running_add = 0
#             running_add = running_add * gamma + reward[i]
#             discounted_r[i] = running_add
#
#         discounted_r -= np.mean(discounted_r)  # normalizing the result
#         discounted_r /= np.std(discounted_r)  # divide by standard deviation
#         return discounted_r
#
#     def replay(self, states, actions, rewards):
#         # reshape memory to appropriate shape for training
#         states = np.vstack(states)
#         actions = np.vstack(actions)
#
#         # Compute discounted rewards
#         discounted_r = self.discount_rewards(rewards)
#
#         # Get Critic network predictions
#         value = self.Critic.predict(states)[:, 0]
#         # Compute advantages
#         advantages = discounted_r - value
#         # training Actor and Critic networks
#         self.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
#         self.Critic.fit(states, discounted_r, epochs=1, verbose=0)
#
#     def load(self, Actor_name, Critic_name):
#         self.Actor = load_model(Actor_name, compile=False)
#         # self.Critic = load_model(Critic_name, compile=False)
#
#     def save(self):
#         self.Actor.save(self.Model_name + '_Actor.h5')
#         # self.Critic.save(self.Model_name + '_Critic.h5')
#
#     pylab.figure(figsize=(18, 9))
#
#     def PlotModel(self, score, episode):
#         self.scores.append(score)
#         self.episodes.append(episode)
#         self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
#         if str(episode)[-2:] == "00":  # much faster than episode % 100
#             pylab.plot(self.episodes, self.scores, 'b')
#             pylab.plot(self.episodes, self.average, 'r')
#             pylab.ylabel('Score', fontsize=18)
#             pylab.xlabel('Steps', fontsize=18)
#             try:
#                 pylab.savefig(self.path + ".png")
#             except OSError:
#                 pass
#
#         return self.average[-1]
#
#     def imshow(self, image, rem_step=0):
#         cv2.imshow(self.Model_name + str(rem_step), image[rem_step, ...])
#         if cv2.waitKey(25) & 0xFF == ord("q"):
#             cv2.destroyAllWindows()
#             return
#
#     def GetImage(self, frame, image_memory):
#         if image_memory.shape == (1, *self.state_size):
#             image_memory = np.squeeze(image_memory)
#
#         # croping frame to 80x80 size
#         frame_cropped = frame[35:195:2, ::2, :]
#         if frame_cropped.shape[0] != self.COLS or frame_cropped.shape[1] != self.ROWS:
#             # OpenCV resize function
#             frame_cropped = cv2.resize(frame, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
#
#         # converting to RGB (numpy way)
#         frame_rgb = 0.299 * frame_cropped[:, :, 0] + 0.587 * frame_cropped[:, :, 1] + 0.114 * frame_cropped[:, :, 2]
#
#         # convert everything to black and white (agent will train faster)
#         frame_rgb[frame_rgb < 100] = 0
#         frame_rgb[frame_rgb >= 100] = 255
#         # converting to RGB (OpenCV way)
#         # frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2GRAY)
#
#         # dividing by 255 we expresses value to 0-1 representation
#         new_frame = np.array(frame_rgb).astype(np.float32) / 255.0
#
#         # push our data by 1 frame, similar as deq() function work
#         image_memory = np.roll(image_memory, 1, axis=0)
#
#         # inserting new frame to free space
#         image_memory[0, :, :] = new_frame
#
#         # show image frame
#         # self.imshow(image_memory,0)
#         # self.imshow(image_memory,1)
#         # self.imshow(image_memory,2)
#         # self.imshow(image_memory,3)
#
#         return np.expand_dims(image_memory, axis=0)
#
#     def reset(self, env):
#         image_memory = np.zeros(self.state_size)
#         frame = env.reset()
#         for i in range(self.REM_STEP):
#             state = self.GetImage(frame, image_memory)
#         return state
#
#     def step(self, action, env, image_memory):
#         next_state, reward, done, info = env.step(action)
#         next_state = self.GetImage(next_state, image_memory)
#         return next_state, reward, done, info
#
#     def run(self):
#         for e in range(self.EPISODES):
#             state = self.reset(self.env)
#             done, score, SAVING = False, 0, ''
#             # Instantiate or reset games memory
#             states, actions, rewards = [], [], []
#             while not done:
#                 # self.env.render()
#                 # Actor picks an action
#                 action = self.act(state)
#                 # Retrieve new state, reward, and whether the state is terminal
#                 next_state, reward, done, _ = self.step(action, self.env, state)
#                 # Memorize (state, action, reward) for training
#                 states.append(state)
#                 action_onehot = np.zeros([self.action_size])
#                 action_onehot[action] = 1
#                 actions.append(action_onehot)
#                 rewards.append(reward)
#                 # Update current state
#                 state = next_state
#                 score += reward
#                 if done:
#                     average = self.PlotModel(score, e)
#                     # saving best models
#                     if average >= self.max_average:
#                         self.max_average = average
#                         self.save()
#                         SAVING = "SAVING"
#                     else:
#                         SAVING = ""
#                     print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average,
#                                                                                  SAVING))
#
#                     self.replay(states, actions, rewards)
#         # close environemnt when finish training
#         self.env.close()
#
#     def train(self, n_threads):
#         self.env.close()
#         # Instantiate one environment per thread
#         envs = [gym.make(self.env_name) for i in range(n_threads)]
#
#         # Create threads
#         threads = [threading.Thread(
#             target=self.train_threading,
#             daemon=True,
#             args=(self,
#                   envs[i],
#                   i)) for i in range(n_threads)]
#
#         for t in threads:
#             time.sleep(2)
#             t.start()
#
#     def train_threading(self, agent, env, thread):
#         global graph
#         with graph.as_default():
#             while self.episode < self.EPISODES:
#                 # Reset episode
#                 score, done, SAVING = 0, False, ''
#                 state = self.reset(env)
#                 # Instantiate or reset games memory
#                 states, actions, rewards = [], [], []
#                 while not done:
#                     action = agent.act(state)
#                     next_state, reward, done, _ = self.step(action, env, state)
#
#                     states.append(state)
#                     action_onehot = np.zeros([self.action_size])
#                     action_onehot[action] = 1
#                     actions.append(action_onehot)
#                     rewards.append(reward)
#
#                     score += reward
#                     state = next_state
#
#                 self.lock.acquire()
#                 self.replay(states, actions, rewards)
#                 self.lock.release()
#
#                 # Update episode count
#                 with self.lock:
#                     average = self.PlotModel(score, self.episode)
#                     # saving best models
#                     if average >= self.max_average:
#                         self.max_average = average
#                         self.save()
#                         SAVING = "SAVING"
#                     else:
#                         SAVING = ""
#                     print(
#                         "episode: {}/{}, thread: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES,
#                                                                                            thread, score, average,
#                                                                                            SAVING))
#                     if (self.episode < self.EPISODES):
#                         self.episode += 1
#             env.close()
#
#     def test(self, Actor_name, Critic_name):
#         self.load(Actor_name, Critic_name)
#         for e in range(100):
#             state = self.reset(self.env)
#             done = False
#             score = 0
#             while not done:
#                 action = np.argmax(self.Actor.predict(state))
#                 state, reward, done, _ = self.step(action, self.env, state)
#                 score += reward
#                 if done:
#                     print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
#                     break
#         self.env.close()
#
#
# if __name__ == "__main__":
#     env_name = 'Pong-v0'
#     agent = A3CAgent(env_name)
#     # agent.run() # use as A2C
#     agent.train(n_threads=5)  # use as A3C
#     # agent.test('Pong-v0_A3C_2.5e-05_Actor.h5', 'Pong-v0_A3C_2.5e-05_Critic.h5')
import multiprocessing
import os
import threading
from queue import Queue
import gym
import tensorflow as tf
import numpy as np
import keras
from keras import layers
from matplotlib import pyplot as plt


class RandomAgent:
    """Random Agent that will play the specified game

    Arguments:
      env_name: Name of the environment to be played
      max_eps: Maximum number of episodes to run agent for.
  """

    def __init__(self, env_name, max_eps):
        self.env = gym.make(env_name)
        self.max_episodes = max_eps
        # self.global_moving_average_reward = 0
        self.res_queue = Queue()

    def run(self):
        reward_avg = 0
        for episode in range(self.max_episodes):
            done = False
            self.env.reset()
            reward_sum = 0.0
            steps = 0
            while not done:
                # Sample randomly from the action space and step
                _, reward, done, _ = self.env.step(self.env.action_space.sample())
                steps += 1
                reward_sum += reward
            # Record statistics
            # self.global_moving_average_reward = record(episode,
            #                                               reward_sum,
            #                                               0,
            #                                               self.global_moving_average_reward,
            #                                               self.res_queue, 0, steps)

            reward_avg += reward_sum
        final_avg = reward_avg / float(self.max_episodes)
        print("Average score across {} episodes: {}".format(self.max_episodes, final_avg))
        return final_avg


class ActorCriticModel(keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCriticModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dense1 = layers.Dense(100, activation='relu')
        self.policy_logits = layers.Dense(action_size)
        self.dense2 = layers.Dense(100, activation='relu')
        self.values = layers.Dense(1)

    def call(self, inputs):
        # Forward pass
        x = self.dense1(inputs)
        logits = self.policy_logits(x)
        v1 = self.dense2(inputs)
        values = self.values(v1)
        return logits, values


class MasterAgent():
    def __init__(self):
        self.game_name = 'CartPole-v0'
        save_dir = "Generated Data/Saved Models"
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        env = gym.make(self.game_name)
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.opt = tf.compat.v1.train.AdamOptimizer(0.0001, use_locking=True)
        print(self.state_size, self.action_size)

        self.global_model = ActorCriticModel(self.state_size, self.action_size)  # global network
        self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

    def train(self):
        # if args.algorithm == 'random':
        #     random_agent = RandomAgent(self.game_name, 1)
        #     random_agent.run()
        #     return

        res_queue = Queue()

        workers = [Worker(self.state_size,
                          self.action_size,
                          self.global_model,
                          self.opt, res_queue,
                          i, game_name=self.game_name,
                          save_dir=self.save_dir) for i in range(multiprocessing.cpu_count())]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()

        moving_average_rewards = []  # record episode reward to plot
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break
        [w.join() for w in workers]

        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig(os.path.join(self.save_dir,
                                 '{} Moving Average.png'.format(self.game_name)))
        plt.show()


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []


class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self,
                 state_size,
                 action_size,
                 global_model,
                 opt,
                 result_queue,
                 idx,
                 game_name='CartPole-v0',
                 save_dir='/tmp'):
        super(Worker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.local_model = ActorCriticModel(self.state_size, self.action_size)
        self.worker_idx = idx
        self.game_name = game_name
        self.env = gym.make(self.game_name).unwrapped
        self.save_dir = save_dir
        self.ep_loss = 0.0


    def run(self):
        total_step = 1
        mem = Memory()
        while Worker.global_episode < 1000:
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0

            time_count = 0
            done = False
            while not done:
                logits, _ = self.local_model(
                    tf.convert_to_tensor(current_state[None, :],
                                         dtype=tf.float32))
                probs = tf.nn.softmax(logits)

                action = np.random.choice(self.action_size, p=probs.numpy()[0])
                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1
                ep_reward += reward
                mem.store(current_state, action, reward)

                if time_count == 30 or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done,
                                                       new_state,
                                                       mem,
                                                       0.99)
                    self.ep_loss += total_loss
                    # Calculate local gradients
                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    # Push local gradients to global model
                    self.opt.apply_gradients(zip(grads,
                                                 self.global_model.trainable_weights))
                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())

                    mem.clear()
                    time_count = 0

                    if done:  # done and print information
                        # Worker.global_moving_average_reward = \
                        #     record(Worker.global_episode, ep_reward, self.worker_idx,
                        #               Worker.global_moving_average_reward, self.result_queue,
                        #               self.ep_loss, ep_steps)
                        # We must use a lock to save our model and to print to prevent data races.
                        if ep_reward > Worker.best_score:
                            with Worker.save_lock:
                                print("Saving best model to {}, "
                                      "episode score: {}".format(self.save_dir, ep_reward))
                                self.global_model.save_weights(
                                    os.path.join(self.save_dir,
                                                 'model_{}.h5'.format(self.game_name))
                                )
                                Worker.best_score = ep_reward
                        Worker.global_episode += 1
                ep_steps += 1

                time_count += 1
                current_state = new_state
                total_step += 1
        self.result_queue.put(None)

    def compute_loss(self,
                     done,
                     new_state,
                     memory,
                     gamma=0.99):
        if done:
            reward_sum = 0.  # terminal
        else:
            reward_sum = self.local_model(
                tf.convert_to_tensor(new_state[None, :],
                                     dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values = self.local_model(
            tf.convert_to_tensor(np.vstack(memory.states),
                                 dtype=tf.float32))
        # Get our advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                                         dtype=tf.float32) - values
        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss
        actions_one_hot = tf.one_hot(memory.actions, self.action_size, dtype=tf.float32)

        policy = tf.nn.softmax(logits)
        entropy = tf.reduce_sum(policy * tf.compat.v1.log(policy + 1e-20), axis=1)

        policy_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=actions_one_hot,
                                                                 logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss

    def play(self):
        env = gym.make(self.game_name).unwrapped
        state = env.reset()
        model = self.global_model
        model_path = os.path.join(self.save_dir, 'model_{}.h5'.format(self.game_name))
        print('Loading model from: {}'.format(model_path))
        model.load_weights(model_path)
        done = False
        step_counter = 0
        reward_sum = 0

        try:
            while not done:
                env.render(mode='rgb_array')
                policy, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
                policy = tf.nn.softmax(policy)
                action = np.argmax(policy)
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()



MasterAgent = MasterAgent()
MasterAgent.train()




