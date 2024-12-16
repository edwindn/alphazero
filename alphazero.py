import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler

import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from IPython.display import clear_output
import wandb
import sys
import math

plt.ion()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#run = wandb.init(project='alpha-zero',)
    
class Game:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.state_size = (3, 3)
        self.action_size = 9
        self.action_map = {0:[0,0], 1:[0,1], 2:[0,2], 3:[1,0], 4:[1,1], 5:[1,2], 6:[2,0], 7:[2,1], 8:[2,2]}
        
    def get_initial_state(self):
        return np.zeros(self.state_size)
    
    def update_state(self, action, player, state):
        state[tuple(self.action_map[action])] = player
        return state
    
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, action, state):
        player = state[tuple(self.action_map[action])]
        if (np.any(np.sum(state, axis=0) == 3*player) or np.any(np.sum(state, axis=1) == 3*player) or
            np.sum(np.diag(state)) == 3*player or np.sum(np.diag(np.fliplr(state))) == 3*player):
            return True
        return False
    
    def check_terminated(self, action, state):
        if self.check_win(action, state):
            return True, 1 # ! rather than winner, True (since game is always from player's perspective)
        elif np.sum(self.get_valid_moves(state)) == 0:
            return True, 0
        return False, 0
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_view(self, state, player):
        return state*player
    
    def encode_state(self, state):
        return np.stack((state==-1, state==0, state==1)).astype(np.float32)


class Node:
    def __init__(self, game, args, state, parent=None, action=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior # initial model policy when child was created
        self.children = []

        self.wins = 0
        self.value = 0
        self.visits = visit_count

    def is_terminal(self):
        return len(self.children) == 0 # since we expand in all directions at once

    def select(self):
        best_child = None
        best_ucb = -np.inf
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        return best_child

    def get_ucb(self, child):
        if child.visits == 0:
            Q = 0
        else:
            Q = 1 - (child.value / child.visits + 1) / 2 # want child's score to be minimal since players alternate
        return Q + self.args['C'] * np.sqrt(self.visits) / (1 + child.visits) * child.prior

    def expand(self, policy):
        # new logic: return the node with the highest prob
        highest_prob = 0
        #best_child = None
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.update_state(action, 1, child_state)
                child_state = self.game.get_opponent_view(child_state, player=-1) # as seen from the other player

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
                if prob > highest_prob:
                    best_child = child
        return child
        #return best_child

    def simulate(self): # no longer used
        terminal, winner = self.game.check_terminated(self.action, self.state)

        if terminal:
            return -winner # check this - function only returns the negative of the player

        rollout_state = self.state.copy()
        rollout_player = 1
        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state = self.game.update_state(action, rollout_player, rollout_state)

            terminal, winner = self.game.check_terminated(action, rollout_state)
            if terminal:
                if rollout_player == -1:
                    winner = self.game.get_opponent(winner)
                return winner

            rollout_player = -rollout_player

    def backpropagate(self, winner):
        self.visits += 1
        self.value += winner
        winner = self.game.get_opponent(winner) # or value more generally

        if self.parent is not None:
            self.parent.backpropagate(winner)

class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)

        # give the root initial child nodes
        policy, _ = self.model(
            torch.tensor(self.game.encode_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        # give the policy some noise
        policy = (1 - self.args['epsilon']) * policy + self.args['epsilon'] * np.random.dirichlet([self.args['alpha']] * self.game.action_size)
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

        for _ in range(self.args['num_searches']):
            node = root

            while not node.is_terminal():
                node = node.select()
            
            terminal, winner = self.game.check_terminated(node.action, node.state)
            winner = self.game.get_opponent(winner)

            if not terminal:
                policy, value = self.model(torch.tensor(self.game.encode_state(node.state), device=self.model.device).unsqueeze(0))
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                winner = value.item()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                node = node.expand(policy)
                #winner = node.simulate()

            node.backpropagate(winner)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action] = child.visits
        action_probs /= np.sum(action_probs)
        return action_probs # action probs for taking the next action in the game based on how often the simulation visited that node

class ResNet(nn.Module):
    def __init__(self, game, num_blocks, hidden_dim, device):
        super().__init__()
        self.game = game
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.device = device

        self.start_block = nn.Sequential(
            nn.Conv2d(3, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList(
            [ResBlock(self.hidden_dim) for i in range(self.num_blocks)]
        )
        
        self.policy_head = nn.Sequential( # outputs policy choices for actions
            nn.Conv2d(self.hidden_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*3*3, game.action_size)                                                                                  
        )

        self.value_head = nn.Sequential( # outputs value of the node
            nn.Conv2d(self.hidden_dim, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*3*3, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        x = self.start_block(x)
        for b in self.blocks:
            x = b(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        residual = x
        x = self.block(x)
        x += residual
        x = F.relu(x)
        return x


class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def play(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.get_opponent_view(state, player) # always give from prespective of current player
            action_probs = self.mcts.search(neutral_state)
            memory.append((neutral_state, action_probs, player))

            action_probs = action_probs ** (1 / (self.args['temperature'] + 0.000001))
            action_probs /= np.sum(action_probs)
            action = np.random.choice(self.game.action_size, p=action_probs)
            #print(f'chosen action: {action}')
            state = self.game.update_state(action, player, state)
            terminal, winner = self.game.check_terminated(action, state)

            if terminal:
                return_memory = []
                for hist_state, hist_action_probs, hist_player in memory:
                    return_player = winner if hist_player == player else - winner
                    return_memory.append((
                        self.game.encode_state(hist_state),
                        hist_action_probs,
                        return_player
                    ))

                return return_memory

            player = -player

    def train(self, memory):
        # shuffle training data
        random.shuffle(memory)
        tot_loss = 0
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            batch = memory[batch_idx:min(batch_idx+self.args['batch_size'], len(memory)-1)]
            state, policy_targets, value_targets = zip(*batch)
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets) #since these are probs
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            tot_loss += loss.item()
        
        return tot_loss / len(memory)

    def learn(self):
        losses = []
        fig, ax = plt.subplots()
        line, = ax.plot(losses, label='loss')

        for iter in range(self.args['num_iterations']):
            memory = []

            print(f'iteration {iter+1}')
            print('playing ...')
            self.model.eval()
            for play_iter in tqdm(range(self.args['num_game_iterations'])):
                memory += self.play()

            print('training ...') 
            self.model.train()
            for epoch in tqdm(range(self.args['num_epochs'])):
                loss = self.train(memory)
                losses.append(loss)

                line.set_ydata(losses)
                line.set_xdata(range(len(losses)))
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.01)
                clear_output(wait=True)
                #wandb.log({'loss': loss})
            
        torch.save(self.model.state_dict(), f'model_{iter+1}.pth')
        print(f'---- saved model {iter+1} ----')

def test_run(weight_path):
    game = Game()
    model = ResNet(game, 4, 64, device)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
    model.eval()
    state = game.get_initial_state()

    player = 1
    while True:
        print(state)
        valid_moves = game.get_valid_moves(state)
        if player == 1:
            print(f'valid moves: {(np.argwhere(valid_moves)).flatten().tolist()}')
            action = int(input(f'player {player}: '))
            if valid_moves[action] == 0:
                print('action not valid')
                continue
        
        else:
            neutral_state = -state
            action_probs, _ = model(torch.tensor(game.encode_state(neutral_state)).unsqueeze(0))
            action_probs = action_probs.squeeze(0)
            action_probs = torch.softmax(action_probs.detach(), dim=0)
            plt.bar(np.arange(1, 10), action_probs.numpy())
            plt.show()
            plt.pause(1)
            action = torch.argmax(action_probs)
            action = action.item()

        state = game.update_state(action, player, state)
        terminal, winner = game.check_terminated(action, state)

        if terminal:
            print(state)
            if winner == 0:
                print('draw')
            else:
                print(f'player {winner} wins.')
            break

        player = -player

    quit()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        weight_path = sys.argv[1]
        test_run(weight_path)
    game = Game()
    model = ResNet(game, 4, 64, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        'C': 2,
        'num_searches': 60,
        'num_epochs': 4,
        'num_iterations': 10,
        'num_game_iterations': 500,
        'batch_size': 64,
        'temperature': 1.25,
        'epsilon': 0.25,
        'alpha': 0.3
    }
    a0 = AlphaZero(model, optimizer, game, args)

    a0.learn()
