import random as rd
import numpy as np
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, LeakyReLU, Dropout, Flatten, Conv2D
from keras.optimizers import Adam
from tkinter import *
import math
from copy import deepcopy

# UI

SIZE = 800
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
SCORE_TEXT_COLOR = "#7CFC00"
BACKGROUND_COLOR_DICT = {2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563", \
                         32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61", \
                         512: "#edc850", 1024: "#edc53f", 2048: "#edc22e"}

CELL_COLOR_DICT = {2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2", \
                   32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2", \
                   512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2"}

FONT = ("Verdana", 40, "bold")


class DrawUI:
    def __init__(self):
        self.grid_cells = []
        self.tk = Tk()
        self.__init_grid()

    def __init_grid(self):
        self.tk.grid()
        self.background = Frame(master=self.tk, bg=BACKGROUND_COLOR_GAME, width=SIZE, height=SIZE)
        self.background.grid(row=0)
        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = Frame(self.background, bg=BACKGROUND_COLOR_CELL_EMPTY, width=SIZE / GRID_LEN, height=SIZE / GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING, pady=GRID_PADDING)
                t = Label(master=cell, text="", bg=BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=FONT, width=4,
                          height=2)
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)
        self.__create_text()
        self.tk.grid_columnconfigure(0, weight=1)
        self.tk.grid_columnconfigure(1, weight=1)

    def __create_text(self):
        cell = Frame(self.tk, bg=BACKGROUND_COLOR_GAME)
        cell.grid(row=1, sticky='ew')
        t = Label(master=cell, text="Score: 0", bg=BACKGROUND_COLOR_CELL_EMPTY, fg=SCORE_TEXT_COLOR, justify=CENTER, font=FONT)
        t.pack()
        self.text = t

    def __update_grid_cells(self, board, score):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = board[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(new_number), bg=BACKGROUND_COLOR_DICT[new_number],
                                                    fg=CELL_COLOR_DICT[new_number])
        self.text.configure(text=str(score), bg=BACKGROUND_COLOR_GAME, fg=SCORE_TEXT_COLOR)
        self.tk.update_idletasks()

    def draw_board(self, board, score):
        self.__update_grid_cells(board, score)
        self.tk.update()
        self.tk.after(100)

    def init_board(self, board):
        self.__update_grid_cells(board, 0)
        self.tk.update()
        self.tk.after(5)

    def destroy(self):
        self.tk.after(500)
        self.tk.destroy()

# Game


def new_game(n):
    return np.zeros([n, n])


def add_two(state):
    empty_cells = []

    for i in range(len(state)):
        for j in range(len(state[0])):
            if state[i][j] == 0:
                empty_cells.append((i, j))

    if len(empty_cells) == 0:
        return state

    index_pair = empty_cells[rd.randint(0, len(empty_cells) - 1)]

    prob = rd.random()
    if prob >= 0.9:
        state[index_pair[0]][index_pair[1]] = 4
    else:
        state[index_pair[0]][index_pair[1]] = 2
    return state


def game_over(state):
    for i in range(len(state) - 1):
        for j in range(len(state[0]) - 1):
            if state[i][j] == state[i + 1][j] or state[i][j + 1] == state[i][j]:
                return False

    for i in range(len(state)):
        for j in range(len(state[0])):
            if state[i][j] == 0:
                return False

    for k in range(len(state) - 1):
        if state[len(state) - 1][k] == state[len(state) - 1][k + 1]:
            return False

    for j in range(len(state) - 1):
        if state[j][len(state) - 1] == state[j + 1][len(state) - 1]:
            return False

    return True


def reverse(state):
    new_state = []
    for i in range(len(state)):
        new_state.append([])
        for j in range(len(state[0])):
            new_state[i].append(state[i][len(state[0]) - j - 1])
    return new_state


def transpose(state):
    return np.transpose(state)


def cover_up(state):
    new = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    done = False
    for i in range(4):
        count = 0
        for j in range(4):
            if state[i][j] != 0:
                new[i][count] = state[i][j]
                if j != count:
                    done = True
                count += 1
    return new, done


def merge(state):
    done = False
    score = 0
    for i in range(4):
        for j in range(3):
            if state[i][j] == state[i][j + 1] and state[i][j] != 0:
                state[i][j] *= 2
                score += state[i][j]
                state[i][j + 1] = 0
                done = True
    return state, done, score


def up(game):
    game = transpose(game)
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    game = transpose(game)
    return game, done, temp[2]


def down(game):
    game = reverse(transpose(game))
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    game = transpose(reverse(game))
    return game, done, temp[2]


def left(game):
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    return game, done, temp[2]


def right(game):
    game = reverse(game)
    game, done = cover_up(game)
    temp = merge(game)
    game = temp[0]
    done = done or temp[1]
    game = cover_up(game)[0]
    game = reverse(game)
    return game, done, temp[2]


controls = {0: up, 1: left, 2: right, 3: down}


def change_values(state):
    power_mat = np.zeros(shape=(1, 4, 4, 16), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            if state[i][j] == 0:
                power_mat[0][i][j][0] = 1.0
            else:
                power = int(math.log(state[i][j], 2))
                power_mat[0][i][j][power] = 1.0
    return power_mat


def count_empty_cells(state):
    count = 0
    for i in range(len(state)):
        for j in range(len(state)):
            if state[i][j] == 0:
                count += 1
    return count


def make_int_state(mat):
    for i in range(len(mat)):
        for j in range(len(mat)):
            mat[i][j] = int(mat[i][j])
    return mat


class DQNAgent:
    def __init__(self):
        self.state_size = 16
        self.action_size = 4
        self.learning_rate = 0.001
        self.model = None
        self.gamma = 0.9
        self.epsilon = 0.9

    def build_model(self):
        input_dimensions = (self.action_size, self.action_size, self.state_size)
        output_dimensions = self.action_size

        inputs = Input(shape=input_dimensions, name='inputs')
        hidden_1 = Dense(64, activation='linear', name='hidden_1')(inputs)
        hidden_2 = LeakyReLU(0.001, name="hidden_2")(hidden_1)
        hidden_3 = Dropout(rate=0.125, name="hidden_3")(hidden_2)
        hidden_4 = Dense(32, name='hidden_4')(hidden_3)
        hidden_5 = LeakyReLU(0.001, name="hidden_5")(hidden_4)
        outputs = Dense(output_dimensions, activation='linear', name='outputs')(hidden_5)

        self.model = Model(inputs=inputs, outputs=outputs)

        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(loss='mse', optimizer=optimizer, metrics=['acc'])


def train(n_episodes, agent):
    maximum = -1
    episode = -1
    best_board = new_game(4)
    total_iters = 1

    for ep in range(n_episodes+1):
        global board
        board = new_game(4)
        add_two(board)
        add_two(board)
        finish = False
        total_score = 0
        while not finish:
            prev_board = deepcopy(board)
            state = deepcopy(board)
            state = change_values(state)
            state = np.array(state, dtype=np.float32).reshape(1, 4, 4, 16)
            control_scores = agent.model.predict(state)
            control_buttons = np.flip(np.argsort(control_scores), axis=1)
            control_buttons = control_buttons[0][0]
            control_scores = control_scores[0][0]
            labels = deepcopy(control_scores[0])
            num = rd.uniform(0, 1)
            prev_max = np.max(prev_board)
            if num < agent.epsilon:
                legal_moves = list()
                for i in range(4):
                    temp_board = deepcopy(prev_board)
                    temp_board, _, _ = controls[i](temp_board)
                    if np.array_equal(temp_board, prev_board):
                        continue
                    else:
                        legal_moves.append(i)
                if len(legal_moves) == 0:
                    finish = True
                    continue
                con = rd.sample(legal_moves, 1)[0]
                temp_state = deepcopy(prev_board)
                temp_state, _, score = controls[con](temp_state)
                total_score += score
                finish = game_over(temp_state)
                empty1 = count_empty_cells(prev_board)
                empty2 = count_empty_cells(temp_state)
                if not finish:
                    temp_state = add_two(temp_state)
                board = deepcopy(temp_state)
                next_max = np.max(temp_state)
                labels[con] = (math.log(next_max, 2)) * 0.1
                if next_max == prev_max:
                    labels[con] = 0
                labels[con] += (empty2 - empty1)
                temp_state = change_values(temp_state)
                temp_state = np.array(temp_state, dtype=np.float32).reshape(1, 4, 4, 16)
                temp_scores = agent.model.predict(temp_state)
                max_qvalue = np.max(temp_scores)
                labels[con] = (labels[con] + agent.gamma * max_qvalue)
            else:
                for con in control_buttons[0]:
                    prev_state = deepcopy(prev_board)
                    temp_state, _, score = controls[con](prev_state)
                    if np.array_equal(prev_board, temp_state):
                        labels[con] = 0
                        continue
                    empty1 = count_empty_cells(prev_board)
                    empty2 = count_empty_cells(temp_state)
                    temp_state = add_two(temp_state)
                    board = deepcopy(temp_state)
                    total_score += score
                    next_max = np.max(temp_state)
                    labels[con] = (math.log(next_max, 2)) * 0.1
                    if next_max == prev_max:
                        labels[con] = 0
                    labels[con] += (empty2 - empty1)
                    temp_state = change_values(temp_state)
                    temp_state = np.array(temp_state, dtype=np.float32).reshape(1, 4, 4, 16)
                    temp_scores = agent.model.predict(temp_state)
                    max_qvalue = np.max(temp_scores)
                    labels[con] = (labels[con] + agent.gamma * max_qvalue)
                    break
                if np.array_equal(prev_board, board):
                    finish = True

            if (ep > 10000) or (agent.epsilon > 0.1 and total_iters % 2500 == 0):
                agent.epsilon = agent.epsilon / 1.005

            total_iters += 1

        print("Episode " + str(ep) + " finished with score " + str(total_score) + ", board : " + str(board) +
              ", epsilon  : " + str(agent.epsilon))

        if maximum < total_score:
            maximum = total_score
            episode = ep
            best_board = board

    print("Maximum Score: " + str(maximum) + " ,Episode: " + str(episode) + " ,Board: " + str(best_board))
    agent.model.save("Model17.h5")
    

def test(n_episodes=5000, model_path="Model17.h5"):
    agent = DQNAgent()
    if model_path:
        print(f"Load model from: {model_path}")
        agent.model = load_model(model_path)
    else:
        agent.build_model()
        train(n_episodes, agent)
    print("Start playing...")
    ui = DrawUI()
    board = new_game(4)
    board = add_two(board)
    board = add_two(board)
    ui_board = make_int_state(board)
    ui.init_board(ui_board)
    finish = False
    total_score = 0
    while not finish:
        prev_board = deepcopy(board)
        state = deepcopy(board)
        state = change_values(state)
        state = np.array(state, dtype=np.float32).reshape(1, 4, 4, 16)
        control_scores = agent.model.predict(state)
        control_buttons = np.flip(np.argsort(control_scores), axis=1)
        control_buttons = control_buttons[0][0]
        control_scores = control_scores[0][0]
        labels = deepcopy(control_scores[0])
        num = rd.uniform(0, 1)
        prev_max = np.max(prev_board)
        if num < agent.epsilon:
            legal_moves = list()
            for i in range(4):
                temp_board = deepcopy(prev_board)
                temp_board, _, _ = controls[i](temp_board)
                if np.array_equal(temp_board, prev_board):
                    continue
                else:
                    legal_moves.append(i)
            if len(legal_moves) == 0:
                finish = True
                continue
            con = rd.sample(legal_moves, 1)[0]
            temp_state = deepcopy(prev_board)
            temp_state, _, score = controls[con](temp_state)
            total_score += score
            finish = game_over(temp_state)
            empty1 = count_empty_cells(prev_board)
            empty2 = count_empty_cells(temp_state)
            if not finish:
                temp_state = add_two(temp_state)
            board = deepcopy(temp_state)
            next_max = np.max(temp_state)
            labels[con] = (math.log(next_max, 2)) * 0.1
            if next_max == prev_max:
                labels[con] = 0
            labels[con] += (empty2 - empty1)
            temp_state = change_values(temp_state)
            temp_state = np.array(temp_state, dtype=np.float32).reshape(1, 4, 4, 16)
            temp_scores = agent.model.predict(temp_state)
            max_qvalue = np.max(temp_scores)
            labels[con] = (labels[con] + agent.gamma * max_qvalue)
        else:
            for con in control_buttons[0]:
                prev_state = deepcopy(prev_board)
                temp_state, _, score = controls[con](prev_state)
                if np.array_equal(prev_board, temp_state):
                    labels[con] = 0
                    continue
                empty1 = count_empty_cells(prev_board)
                empty2 = count_empty_cells(temp_state)
                temp_state = add_two(temp_state)
                board = deepcopy(temp_state)
                total_score += score
                next_max = np.max(temp_state)
                labels[con] = (math.log(next_max, 2)) * 0.1
                if next_max == prev_max:
                    labels[con] = 0
                labels[con] += (empty2 - empty1)
                temp_state = change_values(temp_state)
                temp_state = np.array(temp_state, dtype=np.float32).reshape(1, 4, 4, 16)
                temp_scores = agent.model.predict(temp_state)
                max_qvalue = np.max(temp_scores)
                labels[con] = (labels[con] + agent.gamma * max_qvalue)
                break
            if np.array_equal(prev_board, board):
                finish = True
        ui_board = make_int_state(board)
        ui_score = int(total_score)
        ui.draw_board(ui_board, ui_score)

        if agent.epsilon > 0.1:
            agent.epsilon = agent.epsilon / 1.005

    print("Finished with score " + str(total_score) + ", board: " + str(board))


test(20000)

