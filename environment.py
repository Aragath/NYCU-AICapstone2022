import gym
import numpy as np
from gym import spaces
import math
import random
from typing import List
from math import floor
from kaggle_environments import make
from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction, Board, Direction
from typing import Union, Tuple, Dict
from reward_utils import get_board_value
#import basics
from helper import *
#from basics import capture_shipyard
from config import (
    N_FEATURES,
    ACTION_SIZE,
    GAME_AGENTS,
    GAME_CONFIG,
    DTYPE,
    MAX_OBSERVABLE_KORE,
    MAX_OBSERVABLE_SHIPS,
    MAX_ACTION_FLEET_SIZE,
    MAX_KORE_IN_RESERVE,
    WIN_REWARD,
)

class KoreGymEnv(gym.Env):
    """An openAI-gym env wrapper for kaggle's kore environment. Can be used with stable-baselines3.

    There are three fundamental components to this class which you would want to customize for your own agents:
        The action space is defined by `action_space` and `gym_to_kore_action()`
        The state space (observations) is defined by `state_space` and `obs_as_gym_state()`
        The reward is computed with `compute_reward()`

    Note that the action and state spaces define the inputs and outputs to your model *as numpy arrays*. Use the
    functions mentioned above to translate these arrays into actual kore environment observations and actions.

    The rest is basically boilerplate and makes sure that the kaggle environment plays nicely with stable-baselines3.

    Usage:
        >>> from stable_baselines3 import PPO
        >>>
        >>> kore_env = KoreGymEnv()
        >>> model = PPO('MlpPolicy', kore_env, verbose=1)
        >>> model.learn(total_timesteps=100000)
    """

    def __init__(self, config=None, agents=None, debug=None):
        super(KoreGymEnv, self).__init__()

        if not config:
            config = GAME_CONFIG
        if not agents:
            agents = GAME_AGENTS
        if not debug:
            debug = True

        self.agents = agents
        self.env = make("kore_fleets", configuration=config, debug=debug)
        self.config = self.env.configuration
        self.trainer = None
        self.raw_obs = None
        self.previous_obs = None

        # Define the action and state space
        # Change these to match your needs. Normalization to the [-1, 1] interval is recommended. See:
        # https://araffin.github.io/slides/rlvs-tips-tricks/#/13/0/0
        # See https://www.gymlibrary.ml/content/spaces/ for more info on OpenAI-gym spaces.
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=ACTION_SIZE,
            dtype=DTYPE
        )

        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.config.size ** 2 * N_FEATURES + 3,),
            dtype=DTYPE
        )

        self.strict_reward = config.get('strict', False)    # is it strict or not, used when evaluating trained agent

        # Debugging info - Enable or disable as needed
        self.reward = 0
        self.n_steps = 0
        self.n_resets = 0
        self.n_dones = 0
        self.last_action = None
        self.last_done = False

    def reset(self) -> np.ndarray:
        """Resets the trainer and returns the initial observation in state space. Used when training & evaluting

        Returns:
            self.obs_as_gym_state: the current observation encoded as a state in state space
        """
        # agents = self.agents if np.random.rand() > .5 else self.agents[::-1]  # Randomize starting position
        self.trainer = self.env.train(self.agents)
        self.raw_obs = self.trainer.reset()
        self.n_resets += 1
        return self.obs_as_gym_state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action in the trainer and return the results.

        Args:
            action: The action in action space, i.e. the output of the stable-baselines3 agent

        Returns:
            self.obs_as_gym_state: the current observation encoded as a state in state space
            reward: The agent's reward
            done: If True, the episode is over
            info: A dictionary with additional debugging information
        """
        kore_action = self.gym_to_kore_action(action)
        self.previous_obs = self.raw_obs
        self.raw_obs, _, done, info = self.trainer.step(kore_action)  # Ignore trainer reward, which is just delta kore
        self.reward = self.compute_reward(done)

        # Debugging info
        # with open('logs/tmp.log', 'a') as log:
        #    print(kore_action.action_type, kore_action.num_ships, kore_action.flight_plan, file=log)
        #    if done:
        #        print('done', file=log)
        #    if info:
        #        print('info', file=log)
        self.n_steps += 1
        self.last_done = done
        self.last_action = kore_action
        self.n_dones += 1 if done else 0

        return self.obs_as_gym_state, self.reward, done, info

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        pass

    @property
    def board(self):
        return Board(self.raw_obs, self.config)

    @property
    def previous_board(self):
        return Board(self.previous_obs, self.config)
    

    def gym_to_kore_action(self, gym_action: np.ndarray) -> Dict[str, str]:
        """Decode an action in action space as a kore action.

        In other words, transform a stable-baselines3 action into an action compatible with the kore environment.

        This method is central - It defines how the agent output is mapped to kore actions.
        You can modify it to suit your needs.

        Our gym_action is a 1-dimensional vector of size 2 (as defined in self.action_space). 
        We will interpret the values as follows:
        gym_action[0] represents the identity of the launched fleet or for shipyards to build ships
        gym_action[0]:
        - -1 ~ -0.6: shipyard defender
        - -0.6 ~ -0.2: attacker(include fleets / shipyards)
        - -0.2 ~ 0.2: shipyard builder
        - 0.2 ~ 0.6: greedy spawner
        - 0.6 ~ 1: miner
        abs(gym_action[1]) encodes the number of ships to build/launch.
        gym_action[2] the target to go (x axis)
        gym_action[3] the target to go (y axis)

        Notes: The same action is sent to all shipyards, though we make sure that the actions are valid.

        Args:
            gym_action: The action produces by our stable-baselines3 agent.

        Returns:
            The corresponding kore environment actions or None if the agent wants to wait.

        """         
        action_launch = gym_action[0] > 0
        action_build = gym_action[0] < 0
        # Mapping the number of ships is an interesting exercise. Here we chose a linear mapping to the interval
        # [1, MAX_ACTION_FLEET_SIZE], but you could use something else. With a linear mapping, all values are
        # evenly spaced. An exponential mapping, however, would space out lower values, making them easier for the agent
        # to distinguish and choose, at the cost of needing more precision to accurately select higher values.
        number_of_ships = int(
            clip_normalize(
                x=abs(gym_action[1]),
                low_in=0,
                high_in=1,
                low_out=1,
                high_out=MAX_ACTION_FLEET_SIZE
            )
        )
        gym_action[2] = int(
            clip_normalize(
                x=gym_action[2],
                low_in=-1,
                high_in=1,
                low_out=0,
                high_out=GAME_CONFIG['size']-1
            )
        )
        gym_action[3] = int(
            clip_normalize(
                x=gym_action[3],
                low_in=-1,
                high_in=1,
                low_out=0,
                high_out=GAME_CONFIG['size']-1
            )
        )

        # Broadcast the same action to all shipyards
        board = self.board
        me = board.current_player
        for shipyard in me.shipyards:
            action = None
            # Shipyard defenser, note: now does the same as greedy spawner, should solve the shipyard problem first
            if -1 <= gym_action[0] < -0.6:
                # Limit the number of ships to the maximum that can be actually built
                max_spawn = shipyard.max_spawn
                max_purchasable = floor(me.kore / self.config["spawnCost"])
                number_of_ships = min(number_of_ships, max_spawn, max_purchasable)
                if number_of_ships:
                    action = ShipyardAction.spawn_ships(number_ships=number_of_ships)
#             # Attacker
#             if -0.6 <= gym_action[0] < -0.2:
#                 # Limit the number of ships to the amount that is actually present in the shipyard
#                 shipyard_count = shipyard.ship_count
#                 number_of_ships = min(number_of_ships, shipyard_count)
                
#                 # Decide where to attack
#                 if number_of_ships:
#                     # try capture shipyard
#                     flight_plan = capture_shipyard(number_of_ships=number_of_ships, agent=me, board=self.board)
#                     if flight_plan !=None and flight_plan !="":
#                         print(flight_plan)
#                         action = ShipyardAction.launch_fleet_with_flight_plan(number_ships=number_of_ships, 
#                                                                               flight_plan=flight_plan)
                    # try direct attack
#                     elif flight_plan == "":
#                         flight_plan = direct_attack(number_of_ships, me, board)
#                         action = ShipyardAction.launch_fleet_with_flight_plan(number_ships=number_of_ships, 
#                                                                               flight_plan=flight_plan)
#                     # try adjacent attack:
#                     elif flight_plan == "":
#                         flight_plan = adjacent_attack(number_of_ships, me, board)
#                         action = ShipyardAction.launch_fleet_with_flight_plan(number_ships=number_of_ships, 
#                                                                               flight_plan=flight_plan)
                #action = ShipyardAction.spawn_ships(number_ships=number_of_ships)           
            # Greedy Spawner
            elif 0.2 <= gym_action[0] < 0.6:
                # Limit the number of ships to the maximum that can be actually built
                max_spawn = shipyard.max_spawn
                max_purchasable = floor(me.kore / self.config["spawnCost"])
                number_of_ships = min(number_of_ships, max_spawn, max_purchasable)
                if number_of_ships:
                    action = ShipyardAction.spawn_ships(number_ships=number_of_ships)
            # Miner
            elif 0.6 <= gym_action[0] <= 1:
                # Get number of ships to launch
                shipyard_count = shipyard.ship_count
                number_of_ships = min(number_of_ships, floor(shipyard_count * 2 / 3)) # *2/3 for not sending every fleet out
                if number_of_ships:
#                     direction = round((gym_action[1] + 1) * 1.5)  # int between 0 (North) and 3 (West)
#                     action = ShipyardAction.launch_fleet_in_direction(number_ships=number_of_ships,
#                                                                       direction=Direction.from_index(direction))
                    target_pos = Point(gym_action[2], gym_action[3])
                    flight_plan = getFlightPlan(shipyard.position, target_pos, number_of_ships, self.board)
                    print(gym_action[2], gym_action[3])
                    print("######### flight plan: ", flight_plan)
                    # if flight plan too long, go get max kore
                    if(len(flight_plan) > max_flight_plan_len(number_of_ships)):
                        target_pos = getNearestLargestKore(shipyard.position, self.board)
                        flight_plan = getFlightPlan(shipyard.position, target_pos, number_of_ships, self.board)
                        print("######### do go fetch max kore, flight plan: ", flight_plan) 
                    # flight plan still too long, do greedy mine
                    if(len(flight_plan) > max_flight_plan_len(number_of_ships)):
                        target_pos = getNearbyLargestKore(shipyard.position, self.board)
                        flight_plan = getFlightPlan(shipyard.position, target_pos, number_of_ships, self.board)
                        print("######### do greedy, flight plan: ", flight_plan) 
                    # if flight plan empty or still too long, random choose a direction
                    if not flight_plan or len(flight_plan) > max_flight_plan_len(number_of_ships):
                        print("######### random flight plan: ", flight_plan)
                        action = ShipyardAction.launch_fleet_in_direction(number_ships=number_of_ships,
                                                                          direction=Direction.random_direction())
                    # launch flight plan if nonempty
                    else:
                        print("######### launch flight plan: ", flight_plan)
                        action = ShipyardAction.launch_fleet_with_flight_plan(number_ships=number_of_ships, 
                                                                              flight_plan=flight_plan)
            shipyard.next_action = action

        return me.next_actions

    @property
    def obs_as_gym_state(self) -> np.ndarray:
        """Return the current observation encoded as a state in state space.

        In other words, transform a kore observation into a stable-baselines3-compatible np.ndarray.

        This property is central - It defines how the kore board is mapped to our state space.
        You can modify it to include as many features as you see convenient.

        Let's keep start with something easy: Define a 21x21x(4+3) state (size x size x n_features and 3 extra features).
        # Feature 0: How much kore there is in a cell
        # Feature 1: How many ships there are in a cell (>0: friendly, <0: enemy)
        # Feature 2: Fleet direction
        # Feature 3: Is a shipyard present? (1: friendly, -1: enemy, 0: no)
        # Feature 4: Progress - What turn is it?
        # Feature 5: How much kore do I have?
        # Feature 6: How much kore does the opponent have?

        We'll make sure that all features are in the range [-1, 1] and as close to a normal distribution as possible.

        Note: This mapping doesn't tackle a critical issue in kore: How to encode (full) flight plans?
        """
        # Init output state
        gym_state = np.ndarray(shape=(self.config.size, self.config.size, N_FEATURES))

        # Get our player ID
        board = self.board
        our_id = board.current_player_id

        for point, cell in board.cells.items():                 # board.cells =  Dict[Point, Cell]
            # Feature 0: How much kore
            gym_state[point.y, point.x, 0] = cell.kore

            # Feature 1: How many ships (>0: friendly, <0: enemy)
            # Feature 2: Fleet direction
            fleet = cell.fleet
            if fleet:
                modifier = 1 if fleet.player_id == our_id else -1
                gym_state[point.y, point.x, 1] = modifier * fleet.ship_count
                gym_state[point.y, point.x, 2] = fleet.direction.value
            else:
                # The current cell has no fleet
                gym_state[point.y, point.x, 1] = gym_state[point.y, point.x, 2] = 0

            # Feature 3: Shipyard present (1: friendly, -1: enemy)
            shipyard = cell.shipyard
            if shipyard:
                gym_state[point.y, point.x, 3] = 1 if shipyard.player_id == our_id else -1
            else:
                # The current cell has no shipyard
                gym_state[point.y, point.x, 3] = 0

        # Normalize features to interval [-1, 1]
        # Feature 0: Logarithmic scale, kore in range [0, MAX_OBSERVABLE_KORE]
        gym_state[:, :, 0] = clip_normalize(
            x=np.log2(gym_state[:, :, 0] + 1),
            low_in=0,
            high_in=np.log2(MAX_OBSERVABLE_KORE)
        )

        # Feature 1: Ships in range [-MAX_OBSERVABLE_SHIPS, MAX_OBSERVABLE_SHIPS]
        gym_state[:, :, 1] = clip_normalize(
            x=gym_state[:, :, 1],
            low_in=-MAX_OBSERVABLE_SHIPS,
            high_in=MAX_OBSERVABLE_SHIPS
        )

        # Feature 2: Fleet direction in range (1, 4)
        gym_state[:, :, 2] = clip_normalize(
            x=gym_state[:, :, 2],
            low_in=1,
            high_in=4
        )

        # Feature 3 is already as normal as it gets

        # Flatten the input (recommended by stable_baselines3.common.env_checker.check_env)
        output_state = gym_state.flatten()

        # Extra Features: Progress, how much kore do I have, how much kore does opponent have
        player = board.current_player
        opponent = board.opponents[0]
        progress = clip_normalize(board.step, low_in=0, high_in=GAME_CONFIG['episodeSteps'])
        my_kore = clip_normalize(np.log2(player.kore+1), low_in=0, high_in=np.log2(MAX_KORE_IN_RESERVE))
        opponent_kore = clip_normalize(np.log2(opponent.kore+1), low_in=0, high_in=np.log2(MAX_KORE_IN_RESERVE))

        return np.append(output_state, [progress, my_kore, opponent_kore])

    def compute_reward(self, done: bool, strict=False) -> float:
        """Compute the agent reward. Welcome to the fine art of RL.

        We'll compute the reward as the current board value and a final bonus if the episode is over. If the player
        wins the episode, we'll add a final bonus that increases with shorter time-to-victory.
        If the player loses, we'll subtract that bonus.

        Args:
            done: True if the episode is over
            strict: If True, count only wins/loses (Useful for evaluating a trained agent)

        Returns:
            The agent's reward
        """
        board = self.board
        previous_board = self.previous_board

        if strict:
            if done:
                # Who won?
                # Ugly but 99% sure correct, see https://www.kaggle.com/competitions/kore-2022/discussion/324150#1789804
                agent_reward = self.raw_obs.players[0][0]
                opponent_reward = self.raw_obs.players[1][0]
                return int(agent_reward > opponent_reward)
            else:
                return 0
        else:
            if done:
                # Who won?
                agent_reward = self.raw_obs.players[0][0]
                opponent_reward = self.raw_obs.players[1][0]
                if agent_reward is None or opponent_reward is None:
                    we_won = -1
                else:
                    we_won = 1 if agent_reward > opponent_reward else -1
                win_reward = we_won * (WIN_REWARD + 5 * (GAME_CONFIG['episodeSteps'] - board.step))
            else:
                win_reward = 0

            return get_board_value(board) - get_board_value(previous_board) + win_reward


def clip_normalize(x: Union[np.ndarray, float],
                   low_in: float,
                   high_in: float,
                   low_out=-1.,
                   high_out=1.) -> Union[np.ndarray, float]:
    """Clip values in x to the interval [low_in, high_in] and then MinMax-normalize to [low_out, high_out].

    Args:
        x: The array of float to clip and normalize
        low_in: The lowest possible value in x
        high_in: The highest possible value in x
        low_out: The lowest possible value in the output
        high_out: The highest possible value in the output

    Returns:
        The clipped and normalized version of x

    Raises:
        AssertionError if the limits are not consistent

    Examples:
        >>> clip_normalize(50, low_in=0, high_in=100)
        0.0

        >>> clip_normalize(np.array([-1, .5, 99]), low_in=-1, high_in=1, low_out=0, high_out=2)
        array([0., 1.5, 2.])
    """
    assert high_in > low_in and high_out > low_out, "Wrong limits"

    # Clip outliers
    try:
        x[x > high_in] = high_in
        x[x < low_in] = low_in
    except TypeError:
        x = high_in if x > high_in else x
        x = low_in if x < low_in else x

    # y = ax + b
    a = (high_out - low_out) / (high_in - low_in)
    b = high_out - high_in * a

    return a * x + b
