from time import time
from boardgame2 import ReversiEnv
from keras import callbacks
import numpy as np
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from collections import Counter
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam, SGD
import keras.backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

class PPO:
    def __init__(self,board_shape=4, n_experience_episodes=10,sum_board=True ,gamma=0.999, epochs=1, lr=0.001,entropy_loss=5e-4, hidden_layer_neurons=128,batch_size=32, iterations=2000, eval_period=50, train_randomly=True, algorithm='PPO'):
        self.epsilon = 1e-12
        # How many complete iterations its going to make (experience episode -> update).
        self.iterations = iterations
        # Experience and eval episodes.
        self.n_experience_episodes = n_experience_episodes
        self.eval_episodes = eval_period
        self.episode = 0
        # Model hiperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.entropy_loss = entropy_loss
        self.epochs = epochs
        self.lr = lr
        self.sum_board = sum_board
        # second best action TRUE if we want that the opponent always take the best action in training episodes
        self.second_best_action = False
        self.train_randomly = train_randomly
        # Env params
        self.env = ReversiEnv(board_shape=board_shape)
        self.nA = (board_shape**2)
        self.board_shape = board_shape
        # Instance both models, actor and critic
        self.actor_model = self.get_policy_model(lr=lr, hidden_layer_neurons=hidden_layer_neurons, input_shape=[self.nA] ,output_shape=self.nA)
        self.critic_model = self.get_critic_model(lr=lr, hidden_layer_neurons=128, input_shape=[self.nA] ,output_shape=1)
        
        self.actor_model2 = self.get_policy_model(lr=lr, hidden_layer_neurons=hidden_layer_neurons, input_shape=[self.nA] ,output_shape=self.nA)
        self.critic_model2 = self.get_critic_model(lr=lr, hidden_layer_neurons=128, input_shape=[self.nA] ,output_shape=1)
       
    def reset_env(self):
        (self.board, self.current_player)  = self.env.reset()
        # Reset both reward list
        self.first_reward = []
        self.second_reward = []
        return self.board, self.current_player

    def sample_valid_actions(self, state, player):
        # We need to know valid actions
        valid_actions = np.argwhere(self.env.get_valid((state, player)) == 1)
        return valid_actions[np.random.randint(len(valid_actions))]
        
    def encode_second_action(self, action):
        action_one_hot = np.zeros(self.nA)
        action_number = action[0] * self.board_shape + action[1]
        second_action_one_hot = action_one_hot[action_number]
        return second_action_one_hot


    def get_experience_episodes(self):
      
        # FIRST PLAYER STATS
        first_observations = []
        first_actions = []
        first_predictions = []
        first_discounted_rewards = []
        # SECOND PLAYER STATS
        second_observations = []
        second_actions = []
        second_predictions = []
        second_discounted_rewards = []
        # BOTH
        time_steps = []
        episodes_lenghts = []
        ts_count = 0
        # Reset env and start playing episodes
        self.board, self.current_player = self.reset_env()
        exp_episodes = 0
        while exp_episodes < self.n_experience_episodes:
            
            # Save the board in a flatten way
            flatten_board = self.board.reshape(-1)
            first_observations.append(flatten_board)

            # Predict action for that board state
            action, action_one_hot, mask_prediction, prediction = self.get_action(eval=False, player=self.current_player)
            # Save one hot action, mask prediction and prediction
            first_actions.append(action_one_hot)
            first_predictions.append(prediction.flatten())
            # Transform the action to coordinates
            action = self.encode_action(action)
            (self.board, self.current_player), reward, done, _ = self.env.step(action)
            # time step
            ts_count += 1
            # Save reward
            self.first_reward.append(reward)

            # In the same step we need to take an action from de local player (opponent)
            while not done and (self.current_player == -1):
                if not self.train_randomly:
                    # Look for valid action. If second best action is True, we are going to take the argmax
                    second_action, second_action_one_hot, second_mask_prediction, second_prediction = self.get_action(eval=self.second_best_action, player=-1)
                    second_action = self.encode_action(second_action)
                    second_observations.append(self.board.reshape(-1) * -1)
                    second_actions.append(second_action_one_hot)
                    second_predictions.append(second_prediction.flatten())
                else:
                    second_action = self.sample_valid_actions(self.board, player=-1)
                # Take action
                (self.board, self.current_player), reward, done, _ = self.env.step(second_action)
                ts_count += 1
                time_steps.append(ts_count)
                self.second_reward.append(reward)



            if done:
                exp_episodes += 1
                # We sum the board so we know who won, and for how much did it.
                if self.sum_board:
                    final_result = self.board.sum()
                else:
                    final_result = reward
                # We replace the last reward for this one and calculate the discount reward for the whole episodes
                self.first_reward[-1] = final_result
                first_discounted_reward = self.get_discounted_rewards(self.first_reward)
                first_discounted_rewards += first_discounted_reward
                # We make the same for the opponent, so we have his observations but seeing as we are player 1
                self.second_reward[-1] = -1 *final_result
                second_discounted_reward = self.get_discounted_rewards(self.second_reward)
                second_discounted_rewards += second_discounted_reward
                # The lenght of the episode
                ep_len = len(first_discounted_reward) + len(second_discounted_reward)
                episodes_lenghts.append(ep_len)
                # Reset env and play again!
                self.board, self.current_player = self.reset_env()
                ts_count = 0

        if not self.train_randomly:
            first_observations = first_observations + second_observations
            first_actions = first_actions + second_actions
            first_predictions = first_predictions + second_predictions
            first_discounted_rewards = first_discounted_rewards + second_discounted_rewards
        # We concat player 1 and player 2 stats. Remember that all of them are from the point of view of player 1

        return np.array(first_observations), np.array(first_actions), np.array(first_predictions), np.array(first_discounted_rewards), \
               np.array(second_observations), np.array(second_actions), np.array(second_predictions), np.array(second_discounted_rewards), \
               np.array(episodes_lenghts), np.array(time_steps)
    
            
    def get_eval_episode(self):

        observations = []
        actions = []
        predictions = []
        masks_predictions = []
        time_steps = []
        discounted_rewards = []
        total_wins_count = []
        episodes_lenghts = []
        EVAL_EPISODES = 0
        ts_count = 0
        self.reset_env()
        while self.eval_episodes > EVAL_EPISODES:
            
            flatten_board = self.board.reshape(-1)
            observations.append(flatten_board)
            # Becouse we are evaluating the policy, we take the argmax, the best action.
            action, action_one_hot, mask_prediction, prediction = self.get_action(eval=True, player=self.current_player)
            action = self.encode_action(action)

            (self.board, self.current_player), reward, done, _ = self.env.step(action)

            # Save info
            self.first_reward.append(reward)
            actions.append(action_one_hot)
            predictions.append(prediction.flatten())
            masks_predictions.append(mask_prediction)
            ts_count+=1

            # In the same step we need to take an action from de local player
            while not done and (self.current_player == -1):
                opponent_action = self.sample_valid_actions(self.board, player=-1)
                (self.board, self.current_player), reward, done, _ = self.env.step(opponent_action)

            if done:
                EVAL_EPISODES += 1
                discounted_reward = self.get_discounted_rewards(self.first_reward)
                discounted_rewards = np.hstack([discounted_rewards, discounted_reward])
                ep_len = len(discounted_reward)
                episodes_lenghts.append(ep_len)
                # Save the episode winner
                total_wins_count.append(reward)
                ts_count = 0
                self.board, self.current_player = self.reset_env()

        return np.array(total_wins_count), np.array(observations), np.array(actions), np.array(predictions), np.array(discounted_rewards), np.array(episodes_lenghts), np.array(time_steps), np.array(masks_predictions)

    # Create the PPO Loss
    def proximal_policy_optimization_loss(self, advantage, old_prediction, LOSS_CLIPPING=0.2):
        # We have the y_true and the y_pred, both of them being probabilities.
        def loss(y_true, y_pred):
            # Calculate the prob of taking that action, and the old prob of taking it
            prob = K.sum(y_true * y_pred, axis=-1)
            old_prob = K.sum(y_true * old_prediction, axis=-1)
            # ratio 
            r = prob/(old_prob + 1e-10)
            # We use the clip method for PPO, and also add an entropy loss
            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + self.entropy_loss * -(prob * K.log(prob + 1e-10)))
        return loss
    
    def get_policy_model(self, lr=0.001, hidden_layer_neurons = 128, input_shape=[16], output_shape=16):

        def loss_metric(y_true, y_pred):
            y_true_norm = K.sign(y_true)
            return K.categorical_crossentropy(y_true_norm, y_pred)
        # We need to create inputs to the architecture for the advantage and old_predictions, so we can use them in the loss function
        state_input = Input(shape=input_shape)
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(output_shape,))

        X = Dense(hidden_layer_neurons, activation='relu', kernel_initializer='he_uniform')(state_input)
        X = Dense(hidden_layer_neurons, activation="relu", kernel_initializer='he_uniform')(X)
        out_actions = Dense(output_shape, activation='softmax', name='output')(X)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(Adam(lr), loss=[self.proximal_policy_optimization_loss(advantage, old_prediction)], metrics=[loss_metric])
        return model

    def encode_action(self, action):
        return [action // self.board_shape, action % self.board_shape]

    # We need to have a mask for the valid actions
    def get_actions_mask(self, board):
        player = 1
        valid_actions = self.env.get_valid((board, player))
        return valid_actions.reshape(-1)  

    def get_action(self, eval=False, player=1):
        DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, self.nA)), np.zeros((1, 1))
        board = self.board.reshape(1, self.nA)
        # If the board is for the second player, we change it so the model see it as player 1
        if player == -1:
            board = board * -1
        
        p = self.actor_model.predict([board, DUMMY_VALUE, DUMMY_ACTION])
        # predict probabilities for each action, but not all of them are valid
        # Keep only valid actions and recalcuated their probs
        valid_actions = self.get_actions_mask(board.reshape(self.board_shape, self.board_shape))

        mask_prob = p.reshape(valid_actions.shape)*valid_actions
        mask_prob = mask_prob/np.sum(mask_prob)
        # If we are evaluating the policy, we must take the best action
        if eval is False:
            try:
                action = np.random.choice(self.nA, p=mask_prob) #np.nan_to_num(p[0])
            except:
                print(mask_prob.reshape(8, 8))
        else: 
            action = np.argmax(mask_prob)
        action_one_hot = np.zeros(self.nA)
        action_one_hot[action] = 1

        return action, action_one_hot,mask_prob, p

    # Create critic model, it will be separeted from the actor model
    def get_critic_model(self, lr=0.001, hidden_layer_neurons = 128, input_shape=[16], output_shape=1):
        model = Sequential()
        model.add(Dense(hidden_layer_neurons, input_shape=input_shape, activation='tanh'))
        model.add(Dense(output_shape, activation='linear'))
        model.compile(Adam(lr), loss=['mse'])
        return model
    
    def get_entropy(self, preds, epsilon=1e-12):
        entropy = np.mean(-np.sum(np.log(preds+epsilon)*preds, axis=1)/np.log(self.nA))
        return entropy
    
    def get_discounted_rewards(self, r):

        r = np.array(r, dtype=float)
        """Take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return list(discounted_r) 

    def format_as_pandas(self, time_step, obs, preds, actions, disc_sum_rews, decimals = 3):
        df = pd.DataFrame({'step': time_step})
        df['observation'] = [np.array(r*10**decimals, dtype=int)/(10**decimals) for r in obs]
        df['policy_distribution']=[np.array(r*10**decimals, dtype=int)/(10**decimals) for r in preds]
        df['sampled_action'] = [np.array(r, dtype=int) for r in actions]
        df['discounted_sum_rewards']=np.array(disc_sum_rews*10**decimals, dtype=int)/(10**decimals)
    
        return df

    def custom_LearningRate_schedular(self,epoch):
        if epoch < 5:
            return self.lr


    

    def run(self, return_stats=False):

        entropy = []
        dataframes = []
        total_wins = []
        MAX_WIN = 0
        episodes_len = []
        for ITERATION in tqdm(range(self.iterations)):
            # Get experience episodes
            obs1, actions1, preds1, disc_sum_rews1,obs2, actions2, preds2, disc_sum_rews2, ep_len, time_steps = self.get_experience_episodes()
            episodes_len.append(ep_len)
            # Predict values for each state
            values1 = self.critic_model.predict(obs1)
            # Calculate the advantage
            advantage1 = disc_sum_rews1.reshape(-1, 1) - values1
            # Before we fit, we change the actual preds to old preds
            old_prediction1 = preds1
            # Update policy
            callback = LearningRateScheduler(self.custom_LearningRate_schedular)
            self.actor_model.fit([obs1, advantage1, old_prediction1], actions1, verbose=0, 
                                    epochs=self.epochs, batch_size=self.batch_size, callbacks=[callback])

            self.critic_model.fit(obs1, disc_sum_rews1, verbose=0, epochs=self.epochs, batch_size=32)


            if (ITERATION % 25) == 0:
                entropy.append(self.get_entropy(preds1))
                # Evaluate policy
                wins, *_ = self.get_eval_episode()
                wins_p = Counter(wins)[1]
                wins_p = wins_p/len(wins)
                # Save percentage of winnings
                total_wins.append(wins_p)
                print(f'Iteration {ITERATION}: Win % {wins_p}')
                if wins_p > MAX_WIN:
                    # Save the best policy
                    MAX_WIN = wins_p
                    max_iteration = ITERATION
                    print(f'Best model at iteration {max_iteration}: {MAX_WIN}')
                    self.actor_model.save('models/PPO_actor_h5.h5', save_format='tf')
            
            
            self.second_best_action = (MAX_WIN > 0.80) and (not self.train_randomly)
    
            if MAX_WIN > 0.90:
                self.lr = 0.00001
            

        print(f'MAX WINS: {MAX_WIN}. In iteration: {max_iteration}')  
        
         
        # Create a df with iteration, winning percentage and entropy
        wins_entropy_df = pd.DataFrame({'iteration':[x for x in range(1,len(total_wins)+1)],'wins':total_wins, 'entropy':entropy})


        return wins_entropy_df


class TorchPlayer():

    def __init__(self, model_path='models/PPO_actor_h5.h5', player=1, board_shape=None, env=None, deterministic=True,flatten_action=False, only_valid=True, mcts=False, iterationLimit=None, timeLimit=None):
        # we load our dictionary we generated with policy gradient
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.DUMMY_ACT = np.zeros((1, board_shape**2))
        self.DUMMY_VAL = np.zeros((1, 1))
        if env is None:
            env = ReversiEnv(board_shape=board_shape)
        self.player = player
        self.env = env
        self.board_shape = board_shape

    
    
    def predict(self, board):
        board_predict = board.reshape(1,self.board_shape ** 2) * self.player
        p = self.model.predict([board_predict, self.DUMMY_VAL, self.DUMMY_ACT])
        valid_actions = self.get_actions_mask(board)
        mask_prob = p.reshape(valid_actions.shape)*valid_actions
        mask_prob = mask_prob/np.sum(mask_prob)
        action = np.argmax(mask_prob)
        action = self.encode_action(action)
        return action
    
    def encode_action(self,action):
        return [action // 8, action % 8]

    def get_actions_mask(self, board):
        valid_actions = self.env.get_valid((board, self.player))
        return valid_actions.reshape(-1)  