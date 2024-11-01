import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
import random
import os

# Konfigūruojame TensorFlow naudoti GPU, jei yra galimybė
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU is available: {gpus[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available.")


# Kryžiukų-nuliukų žaidimo logikos apibrėžimas
class TicTacToe:
    def __init__(self):
        # Inicijuojame žaidimo lentą kaip 3x3 matricos nulių masyvą, kur 0: tuščias langelis, 1: X, 2: O
        self.board = np.zeros((3, 3), dtype=int)
        # Inicijuojame dabartinį žaidėją kaip 1 (X)
        self.current_player = 1  # 1: X, 2: O

    def reset(self):
        # Atstatome žaidimo lentą į pradinę būseną (visi langeliai tušti)
        self.board = np.zeros((3, 3), dtype=int)
        # Nustatome dabartinį žaidėją į 1 (X)
        self.current_player = 1
        # Grąžiname atstatytą lentą
        return self.board

    def step(self, action):
        # Patikriname, ar veiksmas galimas (langelis turi būti tuščias)
        if self.board[action] != 0:
            raise ValueError(f"Invalid action: {action}")

        # Atliekame veiksmą - įrašome dabartinį žaidėją į pasirinktą langelį
        self.board[action] = self.current_player
        # Patikriname žaidimo būseną po veiksmo
        reward, done = self.check_game_status()
        # Pakeičiame dabartinį žaidėją (3 - 1 = 2; 3 - 2 = 1)
        self.current_player = 3 - self.current_player
        # Grąžiname atnaujintą lentą, atlygį ir žaidimo pabaigos flagą
        return self.board, reward, done

    def check_game_status(self):
        # Patikriname, ar kuris nors žaidėjas laimėjo pagal eilutes ir stulpelius
        for i in range(3):
            if all(self.board[i, :] == self.current_player) or all(
                self.board[:, i] == self.current_player
            ):
                return 1, True  # Pergalė

        # Patikriname, ar kuris nors žaidėjas laimėjo pagal įstrižaines
        if (
            self.board[0, 0]
            == self.board[1, 1]
            == self.board[2, 2]
            == self.current_player
            or self.board[0, 2]
            == self.board[1, 1]
            == self.board[2, 0]
            == self.current_player
        ):
            return 1, True  # Pergalė

        # Patikriname, ar nėra tuščių langelių (lygiųjų atvejis)
        if not (self.board == 0).any():
            return 0.5, True  # Lygiosios

        # Jei žaidimas dar tęsiasi, grąžiname atlygį 0 ir flagą False
        return 0, False  # Žaidimas tęsiasi

    def get_available_actions(self):
        # Surenkame visus galimus veiksmus (tuščius langelius)
        return list(zip(*np.where(self.board == 0)))


# Neuroninio tinklo modelio apibrėžimas
def create_model():
    # Sukuriame sekos modelį
    model = Sequential(
        [
            # "Flatten" sluoksnis ištiesina 3x3 įėjimo matricą į 9 elementų vektorių
            Flatten(input_shape=(3, 3)),
            # Pirmasis tankus (Dense) sluoksnis su 128 neuronais ir "relu" aktyvacijos funkcija
            Dense(128, activation="relu"),
            # Antrasis tankus sluoksnis su 128 neuronais ir "relu" aktyvacijos funkcija
            Dense(128, activation="relu"),
            # Trečiasis tankus sluoksnis su 9 neuronais (po vieną kiekvienam galimam veiksmui) ir "linear" aktyvacijos funkcija
            Dense(9, activation="linear"),
        ]
    )
    # Kompiliuojame modelį, naudojant "adam" optimizatorių ir "mse" nuostolių funkciją
    model.compile(optimizer="adam", loss="mse")
    # Grąžiname sukurtą modelį
    return model


# Sustiprinto mokymosi agento apibrėžimas
class Agent:
    def __init__(self, model):
        # Inicijuojame agentą su duotu neuroninio tinklo modeliu
        self.model = model
        # Pradinis "epsilon" reikšmė, skirta atsitiktiniams veiksmams (epsilon-greedy strategija)
        self.epsilon = 1.0
        # Epsilon mažėjimo koeficientas
        self.epsilon_decay = 0.995
        # Minimalus epsilon reikšmė
        self.epsilon_min = 0.1
        # Diskonto faktorius, naudojamas Q-mokyme
        self.gamma = 0.95

    def get_action(self, state):
        # Jei atsitiktinis skaičius mažesnis už epsilon, atliekame atsitiktinį veiksmą
        if np.random.rand() < self.epsilon:
            action = random.choice(env.get_available_actions())
            print(f"Random action: {action}")
            return action

        # Kitu atveju, numatome Q-vertes dabartinei būsenai
        q_values = self.model.predict(state[np.newaxis])
        # Surūšiuojame veiksmus pagal Q-vertes mažėjančia tvarka
        action_indices = np.argsort(q_values[0])[::-1]
        # Pasirenkame geriausią galimą veiksmą
        for flat_action in action_indices:
            action = divmod(flat_action, 3)
            if action in env.get_available_actions():
                print(f"Predicted action: {action}")
                return action

        # Jei nėra galimų veiksmų, pasirenkame atsitiktinį veiksmą (gali būti atsarginis variantas)
        action = random.choice(env.get_available_actions())
        print(f"Fallback random action: {action}")
        return action

    def train(self, state, action, reward, next_state, done):
        # Pradinis Q-vertės atnaujinimas yra lygi atlygiam
        q_update = reward
        # Jei žaidimas dar nesibaigė, pridedame diskontuotą maksimalios Q-vertės kitai būsenai reikšmę
        if not done:
            q_update += self.gamma * np.max(self.model.predict(next_state[np.newaxis]))
        # Numatome Q-vertes dabartinei būsenai
        q_values = self.model.predict(state[np.newaxis])
        # Atnaujiname Q-vertę veiksmui, kurį atlikome
        q_values[0, action[0] * 3 + action[1]] = q_update
        # Treniruojame modelį, kad atitiktų atnaujintas Q-vertes
        self.model.fit(state[np.newaxis], q_values, verbose=0)
        # Mažiname epsilon, kad mažiau pasikliautume atsitiktiniais veiksmais ateityje
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Modelio išsaugojimo funkcija
def save_model(model, path):
    model.save(path)


# Modelio įkėlimo iš kelio funkcija
def load_model_from_path(path):
    if os.path.exists(path):
        return load_model(path)
    else:
        return create_model()


# Inicializuojame aplinką ir modelį
env = TicTacToe()
model_path = "tic_tac_toe_model.h5"
model = load_model_from_path(model_path)
agent = Agent(model)


# Konsolės išvalymo funkcija
def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


# Lentos atvaizdavimo funkcija su X ir O
def display_board(board):
    symbols = {0: " ", 1: "X", 2: "O"}
    board_str = "\n".join(
        [" | ".join([symbols[cell] for cell in row]) for row in board]
    )
    print(board_str)
    print("\n")


# Funkcija, skirta žaisti prieš vartotoją
def play_against_user(agent, env):
    model_path = "tic_tac_toe_model.h5"
    agent.model = load_model_from_path(
        model_path
    )  # Įkeliame apmokytą modelį žaidimui prieš vartotoją
    state = env.reset()
    state = state.astype(np.float32)
    done = False
    while not done:
        clear_console()
        print(f"Current board:")
        display_board(state)
        if env.current_player == 1:
            # AI gauna veiksmą
            action = agent.get_action(state)
            print(f"AI plays: {action}")
        else:
            while True:
                try:
                    # Vartotojas įveda savo veiksmą
                    action = tuple(
                        map(int, input("Enter your move (row col): ").split())
                    )
                    # Patikriname, ar veiksmas yra galimas
                    if action not in env.get_available_actions():
                        raise ValueError(
                            "Invalid move. The position is either occupied or out of bounds."
                        )
                    break
                except (ValueError, IndexError):
                    print(
                        "Invalid input. Please enter a valid move in the format 'row col'."
                    )

        # Atlikite veiksmą ir gaukite kitą būseną, atlygį ir žaidimo pabaigos flagą
        next_state, reward, done = env.step(action)
        state = next_state.astype(np.float32)
        clear_console()
        print(f"Next board:")
        display_board(state)
        if done:
            # Patikriname žaidimo pabaigos būseną ir spausdiname rezultatą
            if reward == 1:
                print("AI wins!" if env.current_player == 2 else "You win!")
            elif reward == 0.5:
                print("It's a draw!")
            else:
                print("AI loses!" if env.current_player == 2 else "You lose!")
            break


play_against_user(agent, env)
