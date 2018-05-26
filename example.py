import os
os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = '/lib:/usr/lib:/usr/bin/lib:/' + os.environ['DYLD_FALLBACK_LIBRARY_PATH'];

import retro
import numpy as np
import pandas as pd

def adjust_obs(obs):
    return obs.astype('float32') / 255.

def savetonumpy(game, state):
    movie_path = 'human/' + game + '/contest/' + game + '-' + state + '-0000.bk2'
    movie = retro.Movie(movie_path)
    movie.step()

    env = retro.make(game=movie.get_game(), state=retro.STATE_NONE, use_restricted_actions=retro.ACTIONS_ALL)
    env.initial_state = movie.get_state()
    observation = env.reset()
    observation = adjust_obs(observation)

    obs_sequence = []
    action_sequence = []

    print('stepping movie')
    while movie.step():
        keys = []
        for i in range(env.NUM_BUTTONS):
            keys.append(movie.get_key(i))
        
        action_sequence.append(keys)
        obs_sequence.append(observation)

        observation, _rew, _done, _info = env.step(keys)
        observation = adjust_obs(observation)
        saved_state = env.em.get_state()
        env.render()

    obs_data = [obs_sequence]
    action_data = [action_sequence]

    np.save('./data/obs_data_' + game + '-' + state, obs_data)
    np.save('./data/action_data_' + game + '-' + state, action_data)

train = pd.read_csv('sonic-train.csv')
validation = pd.read_csv('sonic-validation.csv')

for row in train.iterrows():
    game = row[1]['game']
    state = row[1]['state']
    savetonumpy(game, state)