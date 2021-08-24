import datetime

from A3C_NN import A3C_NN
from A3C_Buffer import A3C_Buffer
from StateTransition import StateTransition
from HelperPy import HelperPy
from Global import Global

import numpy as np
import tensorflow as tf
import threading
import time
import matplotlib.pyplot as plt
import os

import Test_A3C
from scipy.signal import savgol_filter
import random
import gym


# class A3C_Worker(threading.Thread):
#     save_lock = threading.Lock()
#
#     def __init__(self, inputSize, actionCount, workerID, discountFactor, optimiser, globalModel,
#                  weightUpdateInterval=30):
#         super(A3C_Worker, self).__init__()
#         self.inputSize = inputSize
#         self.actionCount = actionCount
#         self.workerID = workerID
#         self.discountFactor = discountFactor
#         self.optimiser = optimiser
#
#         self.helper = HelperPy()
#
#         self.localModel = Test_A3C.ActorCriticModel(self.inputSize, self.actionCount)
#         self.memory = A3C_Buffer()
#
#         self.weightUpdateInterval = weightUpdateInterval
#         self.acceptAppends = False
#         self.episodeLoss = [0]
#         self.elapsedEpisodes = 0
#
#         self.globalModel = globalModel
#
#         self.localModel.build((None, self.inputSize))
#         for i in range(len(Global.globalModel.trainable_variables)):
#             self.localModel.trainable_variables[i].assign(Global.globalModel.trainable_variables[i])
#
#     def append_to_buffer(self, stringInput, tempPass=False):
#         if tempPass is False:
#             transitionData = StateTransition.string_to_transition(stringInput)
#             self.memory.populate_buffer(transitionData)
#
#         if (len(self.memory.buffer) >= self.weightUpdateInterval) or (self.memory.buffer[-1].terminalState == 1):
#             self.train()
#
#         return str(len(self.memory.buffer))
#
#     def update_reward(self, stringInput):
#         inputs = self.helper.parse_string_input(stringInput)[0]
#         self.memory.update_reward(int(inputs[0]), inputs[1])
#
#         return "0"
#
#     def global_predict(self, stringInput, parseString=True):
#         parsedInput = self.helper.parse_string_input(stringInput)
#         outputs = self.localModel.get_prediction(parsedInput, parseString)
#         return outputs
#
#     def run(self):
#         print("Started worker thread")
#         while True:
#             # print(len(self.memory.buffer))
#             time.sleep(0.001)  # Do you need to sleep for stability? --> YES
#             train = False
#             if len(self.memory.buffer) >= self.weightUpdateInterval:
#                 train = True
#             elif len(self.memory.buffer) > 0:
#                 if self.memory.buffer[-1].terminalState == 1:
#                     train = True
#
#             if train is True:
#                 self.acceptAppends = False
#                 self.train()
#             else:
#                 self.acceptAppends = True
#
#     def train(self):
#         """
#         Call this method to train the NN after a number of steps have finished
#         :return:
#         """
#         lastTransition = self.memory.buffer[-1]
#         with tf.GradientTape() as tape:
#             # Get the gradients of the local model
#             # loss = self._compute_loss(lastTransition, self.memory, self.discountFactor)
#             loss = self.compute_loss_correct(lastTransition.terminalState, lastTransition.newState, self.memory, 0.99)
#
#         self.episodeLoss[-1] += loss
#
#         localGradients = tape.gradient(loss, self.localModel.trainable_weights)
#         # Apply the local gradients to the global model
#         self.optimiser.apply_gradients(zip(localGradients, self.globalModel.trainable_weights))
#         # Update the local model
#         self.localModel.set_weights(self.globalModel.get_weights())
#         self.memory.clear_buffer()
#
#     def _compute_loss(self, lastTransition, memory, discountFactor):
#         # If this is the terminal state
#         if lastTransition.terminalState == 1:
#             rewardSum = 0.
#         else:
#             # networkOutput = self.localModel.get_prediction(tf.convert_to_tensor(np.array([lastTransition.newState])))
#             networkOutput = self.localModel(tf.convert_to_tensor([lastTransition.newState], dtype=tf.float32))
#             rewardSum = networkOutput[1].numpy()[0][0]
#
#         discountedRewards = []
#         # rewards = [transition.reward for transition in memory.buffer][::-1]
#         for reward in memory.rewards[::-1]:
#             rewardSum = reward + (discountFactor * rewardSum)
#             discountedRewards.append(rewardSum)
#
#         discountedRewards.reverse()
#
#         # Compute the nn output over the whole batch/episode
#         networkOutput = self.localModel(tf.convert_to_tensor(np.vstack(memory.initialStates), dtype=tf.float32))
#
#         # Calculate the value loss
#         advantage = tf.convert_to_tensor(discountedRewards, dtype=tf.float32) - networkOutput[1]
#         valueLoss = advantage ** 2
#
#         # Calculate the policy loss
#         oheAction = tf.one_hot(memory.actions, self.actionCount, dtype=tf.float32)
#
#         # Adding entropy to the loss function discourages premature convergence
#         policy = tf.nn.softmax(networkOutput[0])
#         entropy = tf.reduce_sum(policy * tf.math.log(policy + 1e-20), axis=1)
#
#         policyLoss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=oheAction, logits=networkOutput[0])
#         policyLoss = policyLoss * tf.stop_gradient(advantage)
#         policyLoss = policyLoss - 0.01 * entropy
#
#         totalLoss = tf.reduce_mean((0.5 * valueLoss + policyLoss))
#         return totalLoss
#
#     def compute_loss_COPY(self,
#                           done,
#                           new_state,
#                           memory,
#                           gamma=0.99):
#         if done:
#             reward_sum = 0.  # terminal
#         else:
#             nnOut = self.localModel(tf.convert_to_tensor(new_state[None, :], dtype=tf.float32))
#             reward_sum = nnOut[-1].numpy()[0]
#
#         # Get discounted rewards
#         discounted_rewards = []
#         for reward in memory.rewards[::-1]:  # reverse buffer r
#             reward_sum = reward + gamma * reward_sum
#             discounted_rewards.append(reward_sum)
#         discounted_rewards.reverse()
#
#         logits, values = self.localModel(
#             tf.convert_to_tensor(np.vstack(memory.initialStates),
#                                  dtype=tf.float32))
#         # Get our advantages
#         dr = np.array(discounted_rewards)[:, None]
#         advantage = tf.convert_to_tensor(dr,
#                                          dtype=tf.float32) - values
#         # Value loss
#         value_loss = advantage ** 2
#
#         # Calculate our policy loss
#         actions_one_hot = tf.one_hot(memory.actions, self.actionCount, dtype=tf.float32)
#
#         policy = tf.nn.softmax(logits)
#         entropy = tf.reduce_sum(policy * tf.compat.v1.log(policy + 1e-20), axis=1)
#
#         policy_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=actions_one_hot,
#                                                                            logits=logits)
#         policy_loss *= tf.stop_gradient(advantage)
#         policy_loss -= 0.01 * entropy
#         total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
#         return total_loss
#
#     def compute_loss_correct(self,
#                              done,
#                              new_state,
#                              memory,
#                              gamma):
#         if done == 1:
#             reward_sum = 0.  # terminal
#         else:
#             nnOut = self.localModel(tf.convert_to_tensor(new_state[None, :], dtype=tf.float32))
#             reward_sum = nnOut[-1].numpy()[0][0]
#
#         # Get discounted rewards
#         discounted_rewards = []
#         for reward in memory.rewards[::-1]:  # reverse buffer r
#             reward_sum = reward + gamma * reward_sum
#             discounted_rewards.append(reward_sum)
#         discounted_rewards.reverse()
#
#         logits, values = self.localModel(
#             tf.convert_to_tensor(np.vstack(memory.initialStates),
#                                  dtype=tf.float32))
#         # Get our advantages
#         dr = np.array(discounted_rewards)[:, None]
#         advantage = tf.convert_to_tensor(dr,
#                                          dtype=tf.float32) - values
#         # Value loss
#         value_loss = advantage ** 2
#
#         # Calculate our policy loss
#         actions_one_hot = tf.one_hot(memory.actions, self.actionCount, dtype=tf.float32)
#
#         policy = tf.nn.softmax(logits)
#         entropy = tf.reduce_sum(policy * tf.compat.v1.log(policy + 1e-20), axis=1)
#
#         policy_loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=actions_one_hot,
#                                                                            logits=logits)
#         policy_loss *= tf.stop_gradient(advantage)
#         policy_loss -= 0.01 * entropy
#         total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
#         return total_loss

class A3C_Worker(threading.Thread):
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
        super(A3C_Worker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.local_model = A3C_NN(self.state_size, self.action_size)
        self.worker_idx = idx
        self.game_name = game_name
        self.env = gym.make(self.game_name).unwrapped
        self.save_dir = save_dir
        self.ep_loss = 0.0
        self.memory = A3C_Buffer()

        self.rewards = []
        self.helper = HelperPy()

    def append_to_buffer(self, stringInput, tempPass=False):
        if tempPass is False:
            transitionData = StateTransition.string_to_transition(stringInput)
            self.memory.populate_buffer(transitionData)
        if (len(self.memory.buffer) >= 30) or (self.memory.buffer[-1].terminalState == 1):
            self.train()

        return str(len(self.memory.buffer))

    def global_predict(self, stringInput):
        parsedInput = self.helper.parse_string_input(stringInput)
        outputs = self.local_model.get_prediction(parsedInput, parsedInput)
        return outputs

    def update_reward(self):
        pass

    def run(self):
        total_step = 1
        episodeCount = 0
        epsilon = 1
        while Global.globalEpisode < 5000:
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
                        print(str(Global.globalEpisode) + " : " + str(round(end - start, 2)))

                        if ep_reward > Global.bestReward:
                            print("Saving best model to {}, "
                                  "episode score: {}".format(self.save_dir, ep_reward))
                            Global.bestReward = ep_reward
                            # with Worker.save_lock:
                            #     print("Saving best model to {}, "
                            #           "episode score: {}".format(self.save_dir, ep_reward))
                            #     self.global_model.save_weights(
                            #         os.path.join(self.save_dir,
                            #                      'model_{}.h5'.format(self.game_name))
                            #     )
                            #     Worker.best_score = ep_reward
                        Global.globalEpisode += 1
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
