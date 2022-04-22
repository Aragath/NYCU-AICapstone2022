from board_env import Board
from RL_brain import QLearningTable


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(observation, action, reward, observation_)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.clear()

if __name__ == "__main__":

    # initial env
    env = Board()

    # initial Q table
    RL = QLearningTable(actions=list(range(env.n_actions)))
