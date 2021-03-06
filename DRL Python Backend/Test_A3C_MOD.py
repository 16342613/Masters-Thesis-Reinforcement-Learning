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
from scipy.signal import savgol_filter
import random
import math


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
        self.dense1 = layers.Dense(64, activation='relu')
        self.policy_logits = layers.Dense(action_size)
        self.dense2 = layers.Dense(64, activation='relu')
        self.values = layers.Dense(1)

    def call(self, inputs):
        # Forward pass
        # x = self.dense1(inputs)
        # logits = self.policy_logits(x)
        # v1 = self.dense2(inputs)
        # values = self.values(v1)

        x = self.dense1(inputs)
        x = self.dense2(x)
        logits = self.policy_logits(x)
        values = self.values(x)
        return logits, values


class MasterAgent:
    results = []

    def __init__(self, workerCount):
        self.workerCount = workerCount
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

    def train(self, workerCount=None, bufferCount=None):
        if workerCount is None:
            workerCount = self.workerCount

        res_queue = Queue()
        if bufferCount is not None:
            workers = [Worker(self.state_size,
                              self.action_size,
                              self.global_model,
                              self.opt, res_queue,
                              i,
                              game_name=self.game_name,
                              bufferSize=bufferCount,
                              save_dir=self.save_dir) for i in
                       range(workerCount)]  # range(multiprocessing.cpu_count())]
        else:
            workers = [Worker(self.state_size,
                              self.action_size,
                              self.global_model,
                              self.opt, res_queue,
                              i,
                              game_name=self.game_name,
                              bufferSize=30,
                              save_dir=self.save_dir) for i in
                       range(workerCount)]  # range(multiprocessing.cpu_count())]

        for i, worker in enumerate(workers):
            MasterAgent.results.append([])
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

        # self.plot_results()


    def multi_config_train(self, overrideThreadCount):

        # for i in range(self.workerCount):
        for i in overrideThreadCount:
            self.global_model = ActorCriticModel(self.state_size, self.action_size)
            self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

            self.train(bufferCount=i)  # workerCount=i+1)
            plotData, originalData = self._generate_plot_data()
            # plt.plot([i for i in range(len(originalData))], originalData, label=str(i + 1) + " Original")

            if i == 0:
                plt.plot([i for i in range(len(plotData))], plotData, label=str(i) + " Buffer")
            else:
                plt.plot([i for i in range(len(plotData))], plotData, label=str(i) + " Buffer")

            MasterAgent.results = []  # Clear the results buffer

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("A3C Episode vs Reward for Various Thread Counts")
        plt.legend()
        plt.show()


    def _generate_plot_data(self):
        averagedResults = []

        for i in range(len(MasterAgent.results[0])):
            episodeRewards = []
            for j in range(len(MasterAgent.results)):
                episodeRewards.append(MasterAgent.results[j][i])

            averagedResults.append(sum(episodeRewards) / len(MasterAgent.results))

        windowLength = round(199 / len(MasterAgent.results))
        if windowLength % 2 == 0:
            windowLength += 1

        ySmooth = savgol_filter(averagedResults, windowLength, 6)
        return ySmooth, averagedResults


    def plot_results(self):
        averagedResults = []

        for i in range(len(MasterAgent.results[0])):
            episodeRewards = []
            for j in range(self.workerCount):
                episodeRewards.append(MasterAgent.results[j][i])

            averagedResults.append(sum(episodeRewards) / self.workerCount)

        windowLength = round(99 / self.workerCount)
        if windowLength % 2 == 0:
            windowLength += 1

        ySmooth = savgol_filter(averagedResults, windowLength, 3)
        # plt.plot([i for i in range(len(MasterAgent.results[0]))], MasterAgent.results[0])
        plt.plot([i for i in range(len(ySmooth))], ySmooth, color='black')
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
                 game_name,
                 bufferSize,
                 save_dir='/tmp'):
        super(Worker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.local_model = ActorCriticModel(self.state_size, self.action_size)
        self.worker_idx = idx
        self.bufferSize = bufferSize
        self.game_name = game_name
        self.env = gym.make(self.game_name).unwrapped
        self.save_dir = save_dir
        self.ep_loss = 0.0

    def run(self):
        total_step = 1
        mem = Memory()
        rewards = []
        episode_count = 0
        while episode_count < 1500:  # Worker.global_episode < 1500:
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0

            time_count = 0
            done = False

            while not done:
                # if self.worker_idx == 0:
                    # self.env.render()
                logits, _ = self.local_model(
                    tf.convert_to_tensor(current_state[None, :],
                                         dtype=tf.float32))
                probs = tf.nn.softmax(logits)
 
                action = np.random.choice(self.action_size, p=probs.numpy()[0])
                new_state, reward, done, _ = self.env.step(action)

                # ONLY FOR MOUNTAIN CAR
                # reward = new_state[0]

                # print(new_state)
                # print(reward)
                #if done:
                    # reward = -1
                    #reward = 100
                ep_reward += reward
                mem.store(current_state, action, reward)

                if ep_steps > 750:
                    done = True

                if time_count == self.bufferSize or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done,
                                                       new_state,
                                                       mem,
                                                       0.99)

                    # print(total_loss)
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
                        # print(ep_reward)
                        # print(Worker.global_episode)
                        # print(self.ep_loss)
                        rewards.append(ep_reward)

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
                        episode_count += 1
                ep_steps += 1

                time_count += 1
                current_state = new_state
                total_step += 1

            # print("Episode: " + str(Worker.global_episode) + " ; Reward: " + str(ep_reward) + " ; Steps: " + str(ep_steps) + " ; Env: " + str(self.worker_idx))
            print("Episode: " + str(episode_count) + " ; Reward: " + str(round(ep_reward, 2)) + " ; Steps: " + str(
                ep_steps) + " ; Env: " + str(self.worker_idx))
            MasterAgent.results[self.worker_idx].append(ep_reward)
        self.result_queue.put(None)

        # ySmooth = savgol_filter(rewards, 99, 6)
        # plt.plot([i for i in range(len(rewards))], rewards)
        # plt.plot([i for i in range(len(rewards))], ySmooth, color='black')
        # plt.show()

    def compute_loss(self,
                     done,
                     new_state,
                     memory,
                     gamma=0.99):
        if done:
            reward_sum = 0.  # terminal
        else:
            nnOut = self.local_model(tf.convert_to_tensor(new_state[None, :], dtype=tf.float32))
            reward_sum = nnOut[-1].numpy()[0]

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
        dr = np.array(discounted_rewards)[:, None]
        advantage = tf.convert_to_tensor(dr,
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


class Game:
    def __init__(self):
        self.targetPosition = [0, 0]
        self.agentPosition = [1, 1]

    def reset(self):
        self.agentPosition = [random.uniform(-10, 10), random.uniform(-10, 10)]
        return self._get_current_state()

    def step(self, action):
        # Move right
        if action == 0:
            self.agentPosition = [self.agentPosition[0] + 0.1, self.agentPosition[1]]
        # Move left
        elif action == 1:
            self.agentPosition = [self.agentPosition[0] - 0.1, self.agentPosition[1]]
        # Move up
        elif action == 2:
            self.agentPosition = [self.agentPosition[0], self.agentPosition[1] + 0.1]
        # Move down
        elif action == 3:
            self.agentPosition = [self.agentPosition[0], self.agentPosition[1] - 0.1]

        reward = math.sqrt(((self.agentPosition[0] - self.targetPosition[0]) ** 2) + (
                (self.agentPosition[1] - self.targetPosition[1]) ** 2))

        return self._get_current_state(), reward, self._check_terminal_state(), "PLACEHOLDER"

    def _get_current_state(self):
        return np.array([self.agentPosition, self.targetPosition])

    def _check_terminal_state(self):
        if math.sqrt(((self.agentPosition[0] - self.targetPosition[0]) ** 2) + (
                (self.agentPosition[1] - self.targetPosition[1]) ** 2)) < 0.3:
            return True
        else:
            return False


masterAgent = MasterAgent(4)
# masterAgent.train()
masterAgent.multi_config_train([5, 10, 20, 30, 40, 50])
