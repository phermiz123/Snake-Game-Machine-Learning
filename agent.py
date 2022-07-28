import os
from snakeGame import SnakeGameAI, Direction, Point
import keras
import numpy as np
import tensorflow as tf
from keras import layers
import keras.initializers as initializers
from keras.optimizers import Adam
from keras import  optimizers
import matplotlib.pyplot as plt


def create_network():
    inputs = layers.Input(shape=(11,))
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
    agent_target = create_network()
    game = SnakeGameAI()

    learning_rate = 0.01
    epsilon = 0.8
    gamma = 0.81
    batch_size = 10

    # Experience Replay
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    games = []
    game_number = 1
    scores = []
    episode_reward_history = []
    episode_reward = 0
    episode_count = 0
    record = 0
    max_memory_length = 10_000

    # Optimizer
    opt = Adam(learning_rate=learning_rate)

    # Loss function
    loss_function = keras.losses.Huber()

    while True:
        with tf.GradientTape() as tape:

            curr_state = get_state(game)

            if epsilon > np.random.rand(1)[0]:
                # take a random action
                action = np.random.choice(3)
            else:
                # get the predicted action
                curr_state_tf = tf.convert_to_tensor(curr_state)
                curr_state_tf = tf.expand_dims(curr_state_tf, 0)
                q_values = agent(curr_state_tf, training=False)
                action = tf.argmax(q_values[0]).numpy()

            epsilon -= 0.001
            epsilon = max(epsilon, 0)

            move = [0,0,0]
            move[action] = 1

            reward, game_over, score = game.play_step(move)
            next_state = get_state(game)

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(curr_state)
            state_next_history.append(next_state)
            done_history.append(game_over)
            rewards_history.append(reward)
            state = next_state

            if game_over:
                episode_reward = 0
                game.reset()

                if score > record:
                    record = score
                    scores.append(score)
                    games.append(game_number)
                    game_number += 1
                    plt.plot(games, scores)
                    plt.show()



            if len(done_history) > batch_size:
                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = agent_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, 3)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = agent(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, agent.trainable_variables)
                opt.apply_gradients(zip(grads, agent.trainable_variables))

                agent_target.set_weights(agent.get_weights())

                if len(rewards_history) > max_memory_length:
                    del rewards_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]









if __name__ == '__main__':
    train()