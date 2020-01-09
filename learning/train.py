from tetrisRL.engine import TetrisEngine
from learning.dqn_agent import DQN
from tqdm import tqdm
import numpy as np

width, height = 10, 20
engine = TetrisEngine(width, height)
episodes = 1500
batch_size = 512
min_exp = 10000
ln = 50

agent = DQN(engine,
            layers=((16, (3, 3)), 32),
            activations=('relu', 'relu', 'linear'),
            loss='mse',
            optimizer='adam',
            gamma=0.95,
            epsilon=1,
            epsilon_stop=0,
            epsilon_episodes=1000,
            max_exp=20000)
scores = []
score_log = []

for episode in tqdm(range(episodes)):
    state = engine.clear()
    done = False
    steps = 0
    score = 0

    while not done:
        best_action = agent.choose_action(np.array([state.reshape((width, height, 1))]), episode)

        last_state = state
        state, reward, done = engine.step(best_action)
        agent.save_experience(last_state, best_action, state, reward, done)
        score += int(reward)
        steps += 1

    scores.append(score)
    agent.train(batch_size=batch_size, min_exp=min_exp)

    if episode % ln == 0:
        score_log.append((np.mean(scores[-ln:]), min(scores[-ln:]), max(scores[-ln:])))
        print(score_log[-1])
score_log.append((np.mean(scores[-ln:]), min(scores[-ln:]), max(scores[-ln:])))
print(score_log[-1])
