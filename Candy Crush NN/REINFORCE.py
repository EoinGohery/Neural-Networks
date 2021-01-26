import os
import graphical, game
import numpy as np
import pylab
from keras.optimizers import Adam
import tensorflow as tf
from keras.layers import Dense, Flatten, Input
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from keras import backend as K

tf.config.experimental_run_functions_eagerly(True)

# create my save location if it doesnt exist
model_path = os.path.join(os.getcwd(), 'save_model')
if not os.path.isdir(model_path):
    os.mkdir(model_path)


# converts the board to a usable numpy array
def prep_board(board):
    s = board.replace('\n', '')
    num_array = []
    for character in s:
        number = ord(character) - 96
        num_array.append(number)
    output = np.asarray(split(num_array))
    output.shape = (1, 10, 8)
    return output


# splits the board string into individual letters
def split(word):
    return [char for char in word]


class Agent:
    def __init__(self):

        # initialised current game and set the chosen total game
        self.current_episode = 1
        self.total_episodes = 10000  # Select number of games

        # used to compare with current action
        self.last_action = 0
        self.exploratory_action = 0

        # Memory Buffer
        self.actions_memory = []
        self.states_memory = []
        self.rewards_memory = []

        # pylab Buffer
        self.final_rewards_memory = []

        # number of outputs
        self.n_action = (10*8*2) # 160 potential moves

        # define action space to iterate through
        self.action_space = [i for i in range(self.n_action)]

        # learning variables
        self.lr = 0.0001
        self.gamma = 0.98
        self.G = 0
        self.opt = optimizers.Adam(self.lr)
        self.state_size = (1, 10, 8)

        # initialise and load in model with weights
        self.policy, self.predict = self.build_model()
        self.load_model()

    def build_model(self):
        x_input = Input(shape=self.state_size)
        advantages = Input(shape=[1])
        flat = Flatten()(x_input)
        dense = Dense(1024, activation='relu')(flat)
        dense1 = Dense(512, activation='relu')(dense)
        dense2 = Dense(256, activation='relu')(dense1)
        probs = Dense(self.n_action, activation='softmax')(dense2)

        def loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*advantages)

        policy = Model(inputs=[x_input, advantages], outputs=[probs])
        policy.compile(optimizer=Adam(lr=self.lr), loss=loss)#"categorical_crossentropy")

        predict = Model(inputs=[x_input], outputs=[probs])

        policy.summary()

        return policy, predict

    # Load in model weights from saved file
    def load_model(self):
        try:
            self.policy.load_weights(os.path.join(model_path, "policy.h5"))
        except OSError:
            print("no weights found")

    # save model weights to file
    def save_model(self):
        self.policy.save_weights(os.path.join(model_path, "policy.h5"))

    # Re-adjust model at the end of each game
    def replay(self):
        #  Convert the memory buffer to usable numpy arrays
        state_memory = np.asarray(self.states_memory)
        actions_memory = np.asarray(self.actions_memory)
        rewards_memory = np.asarray(self.rewards_memory)

        actions = np.zeros([len(actions_memory), self.n_action])
        actions[np.arange(len(actions_memory)), actions_memory] = 1

        #  Reinforcement Baseline to adjust Loss
        G = np.zeros_like(rewards_memory)
        for t in range(len(rewards_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards_memory)):
                G_sum += rewards_memory[k]*discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G-mean)/std

        #  Train for all moves in the previous game
        cost = self.policy.train_on_batch([state_memory, self.G], actions)

        #  Clear the memory buffer
        self.states_memory, self.actions_memory, self.rewards_memory, self.exploratory_action = [], [], [], 0

    def ai_callback(self, board, score, moves_left):
        state = prep_board(board)

        # predict probabilities of moves
        probabilities = self.predict.predict(state)[0]
        probabilities = np.hstack(probabilities)

        # Actor picks an action. if it is the same as the previous action then a random "exploritory" action is chosen
        action = np.random.choice(160, p=probabilities)
        if action == self.last_action:
            action = np.random.choice(160)
            self.exploratory_action += 1
        self.last_action = action

        # add action to the memory buffer
        self.actions_memory.append(action)

        # convert 0-159 value to a usable (x, y, vert) output
        val = 0
        x = 0
        y = 0
        vert = False
        for i in range(8):
            if val != action:
                x = i
                val += 1
                for j in range(10):
                    if val != action:
                        y = j
                        val += 1
                        for k in range(2):
                            if val != action:
                                vert = ~vert
                                val += 1

        move = (x, y, vert)
        return move

    def transition_callback(self, board, move, score_delta, next_board, moves_left):
        #  Add the results of the previous turn to the memory buffer
        state = prep_board(board)
        self.states_memory.append(state)
        self.rewards_memory.append((score_delta+5)*10)
        pass

    pylab.figure(figsize=(18, 9))

    def end_of_game_callback(self, boards, scores, moves, final_score):
        # add final score to the final score memory for final graph
        self.final_rewards_memory.append(final_score)

        # save model every 100 games
        if self.current_episode % 100 == 0:
            self.save_model()
            SAVING = "SAVING"

            # save a graph every 100 games of final score trends
            pylab.plot(self.final_rewards_memory, 'b')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Games', fontsize=18)
            "result.png"
            try:
                pylab.savefig(model_path + "/" + "result.png")
            except OSError:
                pass

        else:
            SAVING = ""

        # Print episode/ Game results
        print("game: {}/{}, final score: {} {}".format(self.current_episode, self.total_episodes, final_score, SAVING))

        # end simulation if total games is reached
        if self.current_episode == self.total_episodes:
            return False
        else:
            self.current_episode += 1
            self.replay()
            return True


# main
if __name__ == '__main__':
    speedup = 3.0
    agent = Agent()
    g = graphical.Game(agent.ai_callback, agent.transition_callback, agent.end_of_game_callback, speedup)
    g.run()