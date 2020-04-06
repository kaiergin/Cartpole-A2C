import gym
import tensorflow as tf
import numpy as np
import random

from tensorflow.keras import layers, models

GAMMA = 0.99
EPSILON = .2
MINIBATCH_SIZE = 5
OBS_SPACE = 4
ACTION_SPACE = 2
RENDER = True

actor_opt = tf.keras.optimizers.Adam(3e-4)
critic_opt = tf.keras.optimizers.Adam(3e-4)

env = gym.make('CartPole-v0')

# Takes in state, output action
def create_model_actor():
    model = models.Sequential()
    model.add(layers.Dense(200, activation='relu', input_shape=(4,)))
    model.add(layers.Dense(2, activation='softmax'))
    return model

# Takes in state action pair, output estimated reward
def create_model_critic():
    model = models.Sequential()
    model.add(layers.Dense(200, input_shape=(6,)))
    model.add(layers.Dense(1))
    return model

def eval_actor(actor, obs):
    return np.random.choice(2, p=np.squeeze(actor(np.expand_dims(obs,0))))

def eval_critic(critic, obs, move):
    return tf.cast(critic(tf.concat((obs, move), 1)), tf.float64)

def fill_qvals(rewards, qvals):
    future = 0
    for reward in reversed(rewards):
        value = future * GAMMA + reward
        qvals.insert(0, value)
        future = value

@tf.function
def learn(actor, actor_old, critic, obs, move, qval):
    with tf.GradientTape() as act_tape, tf.GradientTape() as crit_tape:
        prob = tf.cast(tf.boolean_mask(actor(obs), move), tf.float64)
        prob_old = tf.cast(tf.boolean_mask(actor_old(obs), move), tf.float64)
        div = prob / prob_old
        clipped = tf.clip_by_value(div, 1.0 - EPSILON, 1.0 + EPSILON)
        V = eval_critic(critic, obs, move)
        advantage = qval - V
        critic_loss = tf.math.square(advantage)
        actor_loss = -1 * tf.math.minimum(advantage*clipped, advantage*div)
    gradients_of_actor = act_tape.gradient(actor_loss, actor.trainable_variables)
    actor_opt.apply_gradients(zip(gradients_of_actor, actor.trainable_variables))
    gradients_of_critic = crit_tape.gradient(critic_loss, critic.trainable_variables)
    critic_opt.apply_gradients(zip(gradients_of_critic, critic.trainable_variables))

def train(actor, actor_old, critic, obs, moves, qvals):
    indices = list(range(len(qvals)))
    random.shuffle(indices)
    it = 0
    minibatch = (np.zeros((MINIBATCH_SIZE, OBS_SPACE)), np.zeros((MINIBATCH_SIZE, ACTION_SPACE)), np.zeros((MINIBATCH_SIZE, 1)))
    for x in indices:
        minibatch[0][it % MINIBATCH_SIZE] = obs[x]
        minibatch[1][it % MINIBATCH_SIZE] = np.array([1,0]) if moves[x] == 0 else np.array([0,1])
        minibatch[2][it % MINIBATCH_SIZE] = qvals[x]
        if it % MINIBATCH_SIZE == MINIBATCH_SIZE - 1:
            learn(actor, actor_old, critic, *minibatch)
        it += 1

actor = create_model_actor()
actor_old = create_model_actor()
critic = create_model_critic()
num_games = 5000

for _ in range(num_games):
    obs = env.reset()
    game_states = [obs]
    moves_chosen = []
    rewards = []
    dones = []
    qvals = []
    it = 0
    while True:
        action = eval_actor(actor, obs)
        obs, reward, done, _ = env.step(action)
        if RENDER:
            env.render()
        game_states.append(obs)
        moves_chosen.append(action)
        rewards.append(reward)
        dones.append(done)
        it += 1
        if done:
            fill_qvals(rewards, qvals)
            train(actor, actor_old, critic, game_states, moves_chosen, qvals)
            print('Cumulative reward', str(np.sum(rewards)))
            break
    actor_old.set_weights(actor.get_weights())