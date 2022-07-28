import os
from snakeGame import SnakeGameAI, Direction, Point
import keras
import numpy as np
import tensorflow as tf
from keras import layers
import keras.initializers as initializers
from keras.optimizers import Adam

def create_network():
    inputs = layers.Input(shape=(11))
    hidden = layers.Dense(units=128, activation='relu')(inputs)
    q_layer = layers.Dense(units=3, activation='linear')(hidden)

    return keras.Model(inputs=inputs, outputs=q_layer)


def get_state(game):
    head = game.snake[0]
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # Danger straight
        (dir_r and game.is_collision(point_r)) or
        (dir_l and game.is_collision(point_l)) or
        (dir_u and game.is_collision(point_u)) or
        (dir_d and game.is_collision(point_d)),

        # Danger right
        (dir_u and game.is_collision(point_r)) or
        (dir_d and game.is_collision(point_l)) or
        (dir_l and game.is_collision(point_u)) or
        (dir_r and game.is_collision(point_d)),

        # Danger left
        (dir_d and game.is_collision(point_r)) or
        (dir_u and game.is_collision(point_l)) or
        (dir_r and game.is_collision(point_u)) or
        (dir_l and game.is_collision(point_d)),

        # Move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,

            # Food location
        game.food.x < game.head.x,  # food left
        game.food.x > game.head.x,  # food right
        game.food.y < game.head.y,  # food up
        game.food.y > game.head.y  # food down
    ]

    return np.array(state, dtype=int)

def mean_squared_error_loss(q_value: tf.Tensor, reward: tf.Tensor) -> tf.Tensor:
    # Compute mean squared error loss
    loss = 0.5 * (q_value - reward) ** 2

    return loss




def train():
    agent = create_network()
    agent.summary()
    game = SnakeGameAI()
    learning_rate = 0.01
    exploration_rate = 0.8

    # Optimizer
    opt = Adam(learning_rate=learning_rate)

    while True:
        with tf.GradientTape() as tape:

            curr_state = get_state(game)

            curr_state_tf = tf.convert_to_tensor(curr_state)
            curr_state_tf = tf.expand_dims(curr_state_tf, 0)
            q_values = agent(curr_state)




if __name__ == '__main__':
    train()