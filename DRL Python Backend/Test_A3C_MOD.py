import multiprocessing
import os
import threading
from queue import Queue
import gym
import time
import tensorflow as tf
import numpy as np
import keras
from keras import layers
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import random

from A3C_NN import A3C_NN
from A3C_Buffer import A3C_Buffer
from A3C_Master import A3C_Master
from StateTransition import StateTransition


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
        self.opt = tf.compat.v1.train.AdamOptimizer(0.00025, use_locking=True)
        print(self.state_size, self.action_size)

        self.global_model = ActorCriticModel(self.state_size, self.action_size)  # global network
        self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))
        print(self.state_size)
        print(self.action_size)

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
                          save_dir=self.save_dir) for i in range(12)]  # range(multiprocessing.cpu_count())]

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

        # plt.plot(moving_average_rewards)
        # plt.ylabel('Moving average ep reward')
        # plt.xlabel('Step')
        # plt.savefig(os.path.join(self.save_dir,
        #                          '{} Moving Average.png'.format(self.game_name)))
        # plt.show()


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
        self.memory = A3C_Buffer()

        self.rewards = []

    def append_to_buffer(self, stringInput, tempPass=False):
        if tempPass is False:
            transitionData = StateTransition.string_to_transition(stringInput)
            self.memory.populate_buffer(transitionData)

        if (len(self.memory.buffer) >= 15) or (self.memory.buffer[-1].terminalState == 1):
            self.train()

        return str(len(self.memory.buffer))

    def run(self):
        total_step = 1
        episodeCount = 0
        epsilon = 1
        while Worker.global_episode < 5000:
            current_state = self.env.reset()
            self.memory.clear_buffer()
            ep_reward = 0.
            ep_steps = 0
            self.ep_loss = 0

            time_count = 0
            done = False

            start = time.time()
            while not done:
                logits, _ = self.local_model(
                    tf.convert_to_tensor(current_state[None, :],
                                         dtype=tf.float32))
                probs = tf.nn.softmax(logits)
                if random.uniform(0, 1) < epsilon:
                    action = random.randint(0, 1)
                else:
                    action = np.random.choice(self.action_size, p=probs.numpy()[0])
                new_state, reward, done, _ = self.env.step(action)

                doneInt = 0
                if done:
                    reward = -1
                    doneInt = 1

                ep_reward += reward
                self.memory.populate_buffer(StateTransition(current_state, action, reward, new_state, 10, doneInt))
                self.append_to_buffer("", tempPass=True)

                # if done or time_count == 15:
                if done:
                    # self.train()

                    if done:
                        end = time.time()
                        print(str(Worker.global_episode) + " : " + str(round(end - start, 2)))

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
                        episodeCount += 1
                        epsilon = epsilon * 0.9925

                        self.rewards.append(ep_reward)
                        self.memory.clear_buffer()

                ep_steps += 1

                time_count += 1
                current_state = new_state
                total_step += 1

        self.result_queue.put(None)

        ySmooth = savgol_filter(self.rewards, 99, 6)
        plt.plot([i for i in range(len(self.rewards))], self.rewards)
        plt.plot([i for i in range(len(self.rewards))], ySmooth, color='black')
        plt.show()

    def train(self):
        with tf.GradientTape() as tape:
            total_loss = self.compute_loss(done=self.memory.buffer[-1].terminalState,
                                           new_state=self.memory.buffer[-1].newState,
                                           memory=self.memory,
                                           gamma=0.99)

        # print(total_loss)
        self.ep_loss += total_loss
        # Calculate local gradients
        grads = tape.gradient(total_loss, self.local_model.trainable_weights)
        # Push local gradients to global model
        self.opt.apply_gradients(zip(grads,
                                     self.global_model.trainable_weights))
        # Update local model with new weights
        self.local_model.set_weights(self.global_model.get_weights())
        self.memory.clear_buffer()

    def compute_loss(self,
                     done,
                     new_state,
                     memory,
                     gamma=0.99):
        if done == 1:
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
            tf.convert_to_tensor(np.vstack(memory.initialStates),
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


# MasterAgent = MasterAgent()
# MasterAgent.train()

a3cMaster = A3C_Master(4, 2)
for i in range(12):
    a3cMaster.assign_worker("0")
