import chess
import numpy as np
import random
from stockfish import Stockfish
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os

# Chess Environment
class ChessGame:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return self.board.fen()

    def step(self, move):
        self.board.push_san(move)

        if self.board.is_game_over():
            result = self.board.result()
            if result == "1-0":
                reward = 1  # Win
            elif result == "0-1":
                reward = -1  # Loss
            else:
                reward = 0  # Draw
            return self.board.fen(), reward, True, {}

        return self.board.fen(), 0, False, {}

# Neural Network for Aimy
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# Aimy (The AI)
class Aimy:
    def __init__(self, input_size, output_size, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)
        self.model = SimpleNN(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, legal_moves):
        if np.random.rand() <= self.epsilon:
            return random.choice(legal_moves)  # Explore: random move
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            q_values = {move: q_values[0][i].item() for i, move in enumerate(legal_moves)}
            return max(q_values, key=q_values.get)  # Exploit: best move

    def learn(self, batch_size, stockfish):
        if len(self.memory) < batch_size:
            return
        mini_batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Get Q-values from the model
        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        # Use Stockfish to evaluate the "true" Q-values for the next state
        target_q = []
        for i, next_state in enumerate(next_states):
            if dones[i]:
                target_q.append(rewards[i])
            else:
                # Convert the encoded state back to FEN
                board = chess.Board()
                board.clear()
                piece_to_value = {
                    1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
                    -1: 'p', -2: 'n', -3: 'b', -4: 'r', -5: 'q', -6: 'k'
                }
                for square, value in enumerate(next_state):
                    value = int(value.item())  # Convert tensor to integer
                    if value != 0:
                        piece = piece_to_value[value]
                        board.set_piece_at(square, chess.Piece.from_symbol(piece))
                
                # Set the FEN position in Stockfish
                stockfish.set_fen_position(board.fen())

                # Get Stockfish evaluation for the next state
                evaluation = stockfish.get_evaluation()
                if evaluation["type"] == "cp":  # Centipawn evaluation
                    target_q.append(rewards[i] + self.gamma * (evaluation["value"] / 100.0))
                elif evaluation["type"] == "mate":  # Mate evaluation
                    target_q.append(rewards[i] + self.gamma * (1 if evaluation["value"] > 0 else -1))
                else:
                    target_q.append(rewards[i])

        target_q = torch.FloatTensor(target_q)

        # Compute loss and update the model
        loss = self.loss_fn(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename="aimy_model.pth"):
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}.")

    def load_model(self, filename="aimy_model.pth"):
        if os.path.exists(filename):
            self.model.load_state_dict(torch.load(filename))
            print(f"Model loaded from {filename}.")
        else:
            print(f"No model found at {filename}. Starting from scratch.")

# Encode the board state as a numerical vector
def encode_board(fen):
    board = chess.Board(fen)
    piece_to_value = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
    }
    encoded = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            encoded.append(piece_to_value[piece.symbol()])
        else:
            encoded.append(0)  # Empty square
    return np.array(encoded)

# Train Aimy with Stockfish
def train_aimy(aimy, game, episodes=1000, batch_size=32, save_interval=100):
    wins = 0
    losses = 0
    draws = 0
    win_streak = 0
    episode = 0

    while win_streak < 100:  # Continue until Aimy achieves a 100-win streak
        episode += 1
        print(f"\n--- Episode {episode} ---")

        state = game.reset()
        state = encode_board(state)
        done = False
        total_reward = 0

        while not done:
            print(f"Move {game.board.fullmove_number}:")
            legal_moves = [str(move) for move in game.board.legal_moves]
            if not legal_moves:
                print("No legal moves available. Game over.")
                done = True
                break

            # Aimy's turn (both white and black)
            action = aimy.choose_action(state, legal_moves)
            print(f"Aimy's move: {action}")

            next_state, reward, done, _ = game.step(action)
            next_state = encode_board(next_state)
            aimy.remember(state, legal_moves.index(action), reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                result = game.board.result()
                if result == "1-0":
                    wins += 1
                    win_streak += 1
                    print(f"Aimy (White) wins! Win streak: {win_streak}")
                elif result == "0-1":
                    losses += 1
                    win_streak = 0
                    print("Aimy (Black) loses. Win streak reset.")
                else:
                    draws += 1
                    win_streak = 0
                    print("Draw. Win streak reset.")

                print(f"Episode {episode}: Total Reward = {total_reward}, Epsilon = {aimy.epsilon}, Wins = {wins}, Losses = {losses}, Draws = {draws}")
                break

            aimy.learn(batch_size, stockfish)
            print(f"Memory size: {len(aimy.memory)}")  # Debugging: Print memory size
            print("Learning step completed.")  # Debugging: Print after learning

        if (episode + 1) % save_interval == 0:
            aimy.save_model()

    print(f"Aimy achieved a 100-win streak in {episode} episodes!")

# Main program
if __name__ == "__main__":
    stockfish = Stockfish(path=r" ") #paste your stockfish path
    game = ChessGame()
    input_size = 64  # 8x8 board
    output_size = 4672  # Maximum number of possible moves
    aimy = Aimy(input_size, output_size)
    aimy.load_model()  # Load previous training if available

    # Train Aimy
    print("Training Aimy...")
    train_aimy(aimy, game, episodes=1000)

    # Save the final model
    aimy.save_model()
