import os
import random
from collections import deque

import graphical, game
import numpy as np
import pylab
import tensorflow as tf
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Flatten, Input, MaxPool1D
from keras.layers.convolutional import Conv2D, Conv1D
from keras import backend as K
from keras.utils.np_utils import to_categorical
from tensorflow.keras import optimizers, losses
from tensorflow.keras import Model


model_path = os.path.join(os.getcwd(), 'save_model')

if not os.path.isdir(model_path):
    os.mkdir(model_path)

filename = model_path + "/best_result.txt"


def prep_board(board):
    s = board.replace('\n', '')
    num_array = []
    for character in s:
        number = ord(character) - 96
        num_array.append(number)
    output = np.asarray(split(num_array))
    output.shape = (1, 10, 8)
    return output


def split(word):
    return [char for char in word]


class Agent:
    def __init__(self):
        self.last_move= []
        self.current_game = 1
        self.no_games = 1000
        self.turns = []
        self.average = 0
        self.actions = []

        self.x_return = []
        self.y_return = []
        self.binary_return = []

        self.states = []
        self.scores = []
        self.final_scores = []
        self.total_score = 0
        self.max_score = 0
        self.lr = 0.01
        self.gamma = 0.99
        self.opt = optimizers.Adam(self.lr)
        self.state_size = (10, 8)
        self.io = IO()
        self.actor, self.critic = self.build_model()

        self.load_model(self.io.read(filename=filename))
        self.tau = .125
        self.memory = deque(maxlen=2000)

    def build_model(self):
        x_input = Input(shape=self.state_size)
        conv = Conv1D(filters=128, kernel_size=3, strides=1, activation='relu')(x_input)
        conv1 = MaxPool1D(pool_size=2)(conv)
        conv4 = Flatten()(conv1)
        conv5 = Dense(128, activation='relu')(conv4)
        fc = Dense(32, activation='relu')(conv5)

        x_cord = Dense(8, activation='softmax', name='x')(fc)
        y_cord = Dense(10, activation='softmax', name='y')(fc)
        binary_val = Dense(2, activation='softmax', name='bin')(fc)

        x_value = Dense(1, activation='linear', name='x_critic')(fc)
        y_value = Dense(1, activation='linear', name='y_critic')(fc)
        binary_value = Dense(1, activation='linear', name='bin_critic')(fc)

        actor = Model(inputs=x_input, outputs=[x_cord, y_cord, binary_val])
        critic = Model(inputs=x_input, outputs=[x_value, y_value, binary_value])

        critic.compile(loss={'x_critic': 'mse',
                             'y_critic': 'mse',
                             'bin_critic': 'mse'}, optimizer=self.opt)

        actor.compile(loss={'x': 'categorical_crossentropy',
                            'y': 'categorical_crossentropy',
                            'bin': 'categorical_crossentropy'}, optimizer=self.opt)

        actor.summary()
        critic.summary()

        return actor, critic

    """Weighted Gaussian log likelihood loss function for RL"""


    def discount_scores(self, scores):
        # Compute the gamma-discounted scores over an episode
        gamma = self.gamma  # discount rate
        running_add = 0
        discounted_r = np.zeros_like(scores)
        discounted_r = discounted_r.astype(float)
        for i in reversed(range(0, len(scores))):
            running_add = running_add * gamma + scores[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r)  # normalizing the result
        discounted_r /= np.std(discounted_r)  # divide by standard deviation
        return discounted_r

    def replay(self):
        # reshape memory to appropriate shape for training
        states = np.vstack(self.states)
        #print(states)
        x_return = np.vstack(self.x_return)
        y_return = np.vstack(self.y_return)
        binary_return = np.vstack(self.binary_return)

        # Compute discounted scores
        discounted_r = self.discount_scores(self.scores)

        # Get Critic network predictions
        values_x = self.critic.predict(states)[0][0]
        values_y = self.critic.predict(states)[0][1]
        values_binary = self.critic.predict(states)[0][2]
        # Compute advantages
        advantages_x = discounted_r - values_x
        advantages_y = discounted_r - values_y
        advantages_binary = discounted_r - values_binary
        # training Actor and Critic networks
        self.actor.fit(states,  {'x': x_return, 'y': y_return, 'bin': binary_return}, sample_weight={'x': advantages_x, 'y': advantages_y, 'bin': advantages_binary}, epochs=50, verbose=0)
        self.critic.fit(states, {'x_critic': discounted_r, 'y_critic': discounted_r, 'bin_critic': discounted_r}, epochs=50, verbose=0)
        # reset training memory
        self.states, self.actions, self.scores, self.total_score, self.x_return, self.y_return, self.binary_return = [], [], [], 0, [], [], []

    def load_model(self, name):
        name = name.decode("utf-8")
        print(name)
        self.max_score = int(name)
        if self.max_score != -125:
            self.actor.load_weights(os.path.join(model_path, name + "_actor.h5"))
            self.critic.load_weights(os.path.join(model_path, name + "_critic.h5"))

    def save_model(self, name):
        self.actor.save_weights(os.path.join(model_path, name + "_actor.h5"))
        self.critic.save_weights(os.path.join(model_path, name + "_critic.h5"))

    pylab.figure(figsize=(18, 9))

    def ai_callback(self, board, score, moves_left):
        state = prep_board(board)
        # Actor picks an action
        action = self.actor.predict(state)
        x_val = np.argmax(action[0][0], axis=0)
        y_val = np.argmax(action[1][0], axis=0)
        dir_binary = np.argmax(action[2][0], axis=0)
        result = x_val, y_val, dir_binary
        #print("result:", result, "last", self.last_move)

        self.x_return.append(np.asarray(action[0]))
        self.y_return.append(np.asarray(action[1]))
        self.binary_return.append(np.asarray(action[2]))
        self.actions.append([np.asarray(action[0]), np.asarray(action[1]), np.asarray(action[2])])

        print(result)
        return result

    def transition_callback(self, board, move, score_delta, next_board, moves_left):
        # store turn actions to memory
        turn = 25 - moves_left
        state = prep_board(board)
        self.states.append(state)
        if move == self.last_move:
            score_delta -= 5
        if move[0] == 9 & move[2] == 1:
            score_delta -= 10
        if move[0] == 0 & move[2] == 0:
            score_delta -= 10
        if move[1] == 0 & move[2] == 0:
            score_delta -= 10
        if move[1] == 7 & move[2] == 1:
            score_delta -= 10
        self.total_score += score_delta
        self.scores.append(score_delta)
        #print("turn: {}/25, score: {}, total score: {}".format(turn, score_delta, self.total_score))
        self.last_move = move
        pass  # This can be used to monitor outcomes of moves

    def end_of_game_callback(self, boards, scores, moves, final_score):
        # saving best models
        self.replay()
        self.final_scores.append(final_score)
        if final_score >= self.max_score:
            self.max_score = final_score
            self.save_model(str(final_score))
            SAVING = "SAVING"
            self.io.write(filename=filename, data=str(final_score))
        else:
            SAVING = ""
        print("game: {}/{}, final score: {} {}".format(self.current_game, self.no_games, final_score, SAVING))

        if self.current_game == self.no_games:

            pylab.plot(self.final_scores, 'b')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Games', fontsize=18)
            try:
                pylab.savefig("result" + ".png")
            except OSError:
                pass
            return False
        else:
            self.current_game += 1
            return True  # True = play another, False = Done


class IO:
    def read(self, filename):
        toRead = open(filename, "rb")

        out = toRead.read()
        toRead.close()

        return out

    def write(self, filename, data):
        toWrite = open(filename, "wb")

        out = toWrite.write(data.encode('utf-8'))
        toWrite.close()


if __name__ == '__main__':
    speedup = 10.0
    agent = Agent()
    g = graphical.Game(agent.ai_callback, agent.transition_callback, agent.end_of_game_callback, speedup)
    g.run()