from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from collections import deque
import numpy as np
import random


class DQN:
    def __init__(self, engine,
                 layers=((16, (3, 3)), 32),
                 activations=('relu', 'relu', 'linear'),
                 loss='mse',
                 optimizer='adam',
                 gamma=0.95,
                 epsilon=1,
                 epsilon_stop=0.01,
                 epsilon_episodes=1000,
                 max_exp=10000,
                 replay_start_size=None):
        self.engine = engine
        self.model = self._init(layers, activations, loss, optimizer)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_stop = epsilon_stop
        self.epsilon_episodes = epsilon_episodes
        self.experiences = deque(maxlen=max_exp)

    def _init(self, layers, activations, loss, optimizer):
        model = Sequential()

        model.add(Conv2D(layers[0][0], layers[0][1], activation=activations[0], input_shape=(self.engine.width, self.engine.height, 1)))
        model.add(MaxPooling2D((2, 2)))

        if not isinstance(layers[1], tuple):
            model.add(Flatten())

        for i in range(1, len(layers)):
            if isinstance(layers[i], tuple):
                model.add(Conv2D(layers[0][0], layers[0][1], activation=activations[0], input_shape=(self.engine.width, self.engine.height, 1)))
                model.add(MaxPooling2D((2, 2)))

                if not isinstance(layers[i + 1], tuple):
                    model.add(Flatten())
            else:
                model.add(Dense(layers[i], activation=activations[i]))

        model.add(Dense(self.engine.nb_actions, activation=activations[-1]))
        model.compile(loss=loss, optimizer=optimizer)

        return model

    def save_experience(self, current_state, action, next_state, reward, done):
        self.experiences.append((current_state, action, next_state, reward, done))

    def choose_action(self, state, episodes_done):
        thresh = self.epsilon - (self.epsilon - self.epsilon_stop) * episodes_done / self.epsilon_episodes
        if random.random() <= thresh:
            return random.randint(0, self.engine.nb_actions - 1)

        return self.model.predict(state).argmax()

    def train(self, batch_size=128, min_exp=0):
        n = len(self.experiences)

        if n >= min_exp and n >= batch_size:
            batch = random.sample(self.experiences, batch_size)
            next_states = np.array([x[2] for x in batch]).reshape((batch_size, self.engine.width, self.engine.height, 1))
            predicted_qs = [x for x in self.model.predict(next_states)]

            x = []
            y = []
            for i, (state, action, state_n, reward, done) in enumerate(batch):
                if not done:
                    q = reward + self.gamma * predicted_qs[i]
                else:
                    q = np.array([0] * self.engine.nb_actions)
                    q[action] = reward

                x.append(state.reshape((self.engine.width, self.engine.height, 1)))
                y.append(q)

            self.model.fit(np.array(x), np.array(y), batch_size=batch_size, verbose=0)

    def save(self, path='./checkpoints'):
        save_model(self.model, path)

    def load(self, path):
        self.model = load_model(path)
