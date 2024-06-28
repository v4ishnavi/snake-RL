import torch 
import random
import numpy as np

from game import SnakeGameai, Direction, Point
from collections import deque
from model import Linear_QNet, Qtrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
BLOCK_SIZE = 20

class Agent: 
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #randomless
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen = MAX_MEMORY) # automatically popleft
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = Qtrainer(self.model, lr = LR, gamma = self.gamma )
    def get_state(self, game):
        head = game.snake[0] # first element in the snake
        # create some points next to the head in all directions 
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        l = game.direction == Direction.LEFT
        r = game.direction == Direction.RIGHT
        u = game.direction == Direction.UP
        d = game.direction == Direction.DOWN

       # we store all 11 possible states 
        state = [
            # Danger straight
            (r and game.is_collision(point_r)) or 
            (l and game.is_collision(point_l)) or 
            (u and game.is_collision(point_u)) or 
            (d and game.is_collision(point_d)),

            # Danger right
            (u and game.is_collision(point_r)) or 
            (d and game.is_collision(point_l)) or 
            (l and game.is_collision(point_u)) or 
            (r and game.is_collision(point_d)),

            # Danger left
            (d and game.is_collision(point_r)) or 
            (u and game.is_collision(point_l)) or 
            (r and game.is_collision(point_u)) or 
            (l and game.is_collision(point_d)),
            
            # Move direction
            l,
            r,
            u,
            d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int) #makes the bool into ints 
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE: 
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
            # we just get a batchsize length list of tuples 
        else: 
            mini_sample = self.memory 
            
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # take all of them together and train the trainer on it 
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        # random moves: exploration / exploitation trade off 
        self.epsilon = 80 - self.n_games 
        # after the number of games increase more we want to explore less. 
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:  # experiment
            move = random.randint(0,2)
            final_move[move] = 1 
            # randomly checking which direction to turn 
        else: # predict 
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() # most likely pred
            final_move[move] = 1
        
        return final_move
            
            
    

def train():
    plot_scores = []
    plot_mean_scores = []
    
    
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameai()
    while True: 
        # get old state 
        state_old = agent.get_state(game)
        #get move
        final_move = agent.get_action(state_old)
        
        #erform move and get new state 
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # train short memory (1 step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # remember all this
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done: 
            # train long memory, plot results 
            game.reset()
            
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()
                
            print('Game ', agent.n_games, 'Score ', score, 'Record ', record)
            plot_scores.append(score)
            total_score += score
            mean_score  = total_score/ agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()