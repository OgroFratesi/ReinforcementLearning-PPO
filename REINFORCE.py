from time import time
import moviepy.editor as mpy
from boardgame2 import ReversiEnv
import numpy as np
from tqdm import tqdm
from collections import Counter
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
import keras.backend as K
from tensorboardX import SummaryWriter


class REINFORCE:
    def __init__(self,logdir_root='logs',board_shape=4, n_experience_episodes=10, gamma=0.999, epochs=1, batch_size=32,lr=0.001, hidden_layer_neurons=128, iterations=2000, eval_period=50, algorithm='REINFORCE'):
        self.epsilon = 1e-12
        self.batch_size = batch_size
        self.logdir_root = logdir_root
        self.iterations = iterations
        self.n_experience_episodes = n_experience_episodes
        self.episode = 0
        self.gamma = gamma
        self.epochs = epochs
        self.lr = lr
        self.logdir = self.get_log_name(algorithm, logdir_root)
        self.env = ReversiEnv(board_shape=board_shape)
        self.nA = (board_shape**2)
        self.board_shape = board_shape
        self.eval_episodes = eval_period
        self.model = self.get_policy_model(lr=lr, hidden_layer_neurons=hidden_layer_neurons, input_shape=[self.nA] ,output_shape=[self.nA])
        
    
    def reset_env(self):
        # Se observa el primer estado
        self.board, self.current_player  = self.env.reset()
        # Se resetea la lista con los rewards
        return self.board, self.current_player

    def sample_valid_actions(self, state, player):
        # np.argwhere junto con env.get_valid y randint solucionan el problema en pocas lineas pero puede usar otra estrategia
        valid_actions = np.argwhere(self.env.get_valid((state, player)) == 1)

        return valid_actions[np.random.randint(len(valid_actions))]
        

    def encode_action(self, action):
        return [action // self.board_shape, action % self.board_shape]

    def get_actions_mask(self, board):
        player = 1
        valid_actions = self.env.get_valid((board, player))
        return valid_actions.reshape(-1)  


    def get_experience_episodes(self):
      
        # Antes de llamar esta función hay que asegurarse de que el env esta reseteado
        observations = []
        actions = []
        predictions = []
        masks_predictions = []
        time_steps = []
        discounted_rewards = []
        episodes_returns = []
        episodes_lenghts = []
        total_wins = []
        exp_episodes = 0
        ts_count = 0
        # Juega n_experience_episodes episodios
        (self.board, self.current_player) = self.env.reset()

        while exp_episodes < self.n_experience_episodes:
            

            # Guardo la observacion
            flatten_board = self.board.reshape(-1)
            observations.append(flatten_board)
            # Obtengo acción
            action, action_one_hot, mask_prediction, prediction = self.get_action(eval=True)
            action = self.encode_action(action)
            # Ejecuto acción
            (self.board, self.current_player), reward, done, _ = self.env.step(action)
            # Guardo reward obtenido por acción
            actions.append(action_one_hot)
            predictions.append(prediction.flatten())
            masks_predictions.append(mask_prediction)

            ts_count += 1
    
                    

            # In the same step we need to take an action from de local player
            while not done and (self.current_player == -1):
                opponent_action = self.sample_valid_actions(self.board, -1)
                (self.board, self.current_player), reward, done, _ = self.env.step(opponent_action)

            if done:
                exp_episodes += 1
                if reward == 0:
                    reward = -0.5
                discounted_rewards += [reward for x in range(ts_count)]
                time_steps.append(ts_count)
                total_wins.append(reward)
                self.board, self.current_player = self.reset_env()
                ts_count = 0

        
        return np.array(total_wins),np.array(observations), np.array(actions), np.array(predictions), np.array(discounted_rewards), np.array(episodes_returns), np.array(episodes_lenghts), np.array(time_steps), np.array(masks_predictions)
    
            
    def get_eval_episode(self):
        self.reset_env()
        observations = []
        actions = []
        predictions = []
        masks_predictions = []
        rewards = []
        time_steps = []
        discounted_rewards = []
        total_wins_count = []
        episodes_lenghts = []
        EVAL_EPISODES = 0
        ts_count = 0
    
        while self.eval_episodes > EVAL_EPISODES:
            
            flatten_board = self.board.reshape(-1)
            observations.append(flatten_board)
            # Obtengo acción
            action, action_one_hot, mask_prediction, prediction = self.get_action(eval=True)
            action = self.encode_action(action)
            ts_count+=1
            
            # Ejecuto la MEJOR action, estamos evaluando la policy.
            (self.board, self.current_player), reward, done, _ = self.env.step(action)
            
            # Guardo reward obtenido por acción
            rewards.append(reward)
            actions.append(action_one_hot)
            predictions.append(prediction.flatten())
            masks_predictions.append(mask_prediction)
        

            # In the same step we need to take an action from de local player
            while not done and (self.current_player == -1):
                opponent_action = self.sample_valid_actions(self.board, player=-1)
                (self.board, self.current_player), reward, done, _ = self.env.step(opponent_action)

            if done:
                EVAL_EPISODES += 1
                if reward == 0:
                    reward = -0.5
                total_wins_count.append(reward)
                discounted_rewards += [reward for x in range(ts_count)]
                self.board, self.current_player = self.reset_env()
                time_steps.append(ts_count)
                ts_count = 0

        return np.array(total_wins_count), np.array(observations), np.array(actions), np.array(predictions), np.array(discounted_rewards), np.array(episodes_lenghts), np.array(time_steps), np.array(masks_predictions)

    
    def get_policy_model(self, lr=0.001, hidden_layer_neurons = 128, input_shape=[16], output_shape=[16]):
        ## Defino métrica - loss normalizada (sin el retorno multiplicando)
        def loss_metric(y_true, y_pred):
            y_true_norm = K.sign(y_true)
            return K.categorical_crossentropy(y_true_norm, y_pred)
        model = Sequential()
        model.add(Dense(hidden_layer_neurons, input_shape=input_shape, activation='relu'))
    
        model.add(Dense(output_shape, activation='softmax'))
        model.compile(Adam(), loss=['categorical_crossentropy'], metrics=[loss_metric])
       
        return model
    
    def get_action(self, eval=False):
        p = self.model.predict([self.board.reshape(1, -1)])
        valid_actions = self.get_actions_mask(self.board)
        mask_prob = p.reshape(valid_actions.shape)*valid_actions
        mask_prob = mask_prob/np.sum(mask_prob)
        if eval is False:
            action = np.random.choice(self.nA, p=mask_prob) 
        else: 
            action = np.argmax(mask_prob)
        action_one_hot = np.zeros(self.nA)
        action_one_hot[action] = 1
        return action, action_one_hot,mask_prob, p
    
    def get_entropy(self, preds, epsilon=1e-12):
        entropy = np.mean(-np.sum(np.log(preds+epsilon)*preds, axis=1)/np.log(self.nA))
        return entropy
    

    def run(self):

        total_wins = []
        total_wins_train = []
        entropy = []
        time_step = []
        for _ in tqdm(range(self.iterations)):

            wins, obs, actions, preds, disc_sum_rews, ep_returns, ep_len, time_steps, mask_predictions = self.get_experience_episodes()

            wins = Counter(wins)[1]
            total_wins_train.append(wins)

            entropy.append(self.get_entropy(preds))

            pseudolabels = actions*disc_sum_rews.reshape(-1, 1)

            self.model.fit(obs, pseudolabels,verbose=0, epochs=self.epochs, batch_size=self.batch_size)
            
            wins, obs, actions, preds, disc_sum_rews, ep_len, time_steps, mask_predictions = self.get_eval_episode()
            wins_p = Counter(wins)[1]
            total_wins.append(wins_p/len(wins))
            time_step_mean = sum(time_steps) / len(time_steps)
            time_step.append(time_step_mean)

        wins_entropy_df = pd.DataFrame({'iteration':[x for x in range(1,len(total_wins)+1)],'wins':total_wins, 'entropy':entropy, 'time_steps':time_step})

        return wins_entropy_df


class RunningVariance:
    # Keeps a running estimate of variance

    def __init__(self):
        self.m_k = None
        self.s_k = None
        self.k = None

    def add(self, x):
        if not self.m_k:
            self.m_k = x
            self.s_k = 0
            self.k = 0
        else:
            old_mk = self.m_k
            self.k += 1
            self.m_k += (x - self.m_k) / self.k
            self.s_k += (x - old_mk) * (x - self.m_k)

    def get_variance(self, epsilon=1e-12):
        return self.s_k / (self.k - 1 + epsilon) + epsilon
    
    def get_mean(self):
        return self.m_k