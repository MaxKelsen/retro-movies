import os
os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = '/lib:/usr/lib:/usr/bin/lib:/' + os.environ['DYLD_FALLBACK_LIBRARY_PATH'];

import retro
import numpy as np
import pandas as pd

def adjust_obs(obs):
    return obs.astype('float32') / 255.

# save n batches of size of 100 episodes
# each episode contains 1000 frames
def savetonumpy(game, state):
    movie_path = 'human/' + game + '/contest/' + game + '-' + state + '-0000.bk2'
    movie = retro.Movie(movie_path)
    movie.step()

    env = retro.make(game=movie.get_game(), state=retro.STATE_NONE, use_restricted_actions=retro.ACTIONS_ALL)
    env.initial_state = movie.get_state()
    observation = env.reset()
    observation = adjust_obs(observation)

    # declare array heirarchy of data -> batches -> sequences
    obs_data = []
    action_data = []
    obs_batch = []
    action_batch = []
    obs_sequence = []
    action_sequence = []

    step_count = 0
    episode_count = 0
    batch_count = 0

    print('stepping movie')
    while movie.step():

        # populate batches with episodes of size 1000 timesteps
        if (step_count == 300):
            obs_batch.append(obs_sequence)
            action_batch.append(action_sequence)

            # print progress
            print("Batch {} Episode {} finished after {} timesteps".format(batch_count, episode_count, step_count))
            print("Current batch contains {} observations".format(sum(map(len, obs_batch))))

            # reset step count
            step_count = 0

            # reset sequence arrays
            obs_sequence = []
            action_sequence = []

            # increment episode count
            episode_count += 1

        # save batches of size 10 episodes
        if (episode_count == 10):
            
            print("Saving dataset for batch {}".format(batch_count))
            np.save('./data/obs_data_' + game + '_' + state + '_' + str(batch_count), obs_batch)
            np.save('./data/action_data_' + game + '_' + state + '_' + str(batch_count), action_batch)
            
            # reset episode count
            episode_count = 0

            # reset batch arrays
            obs_batch = []
            action_batch = []

            # increment batch count
            batch_count += 1

        keys = []
        for i in range(env.NUM_BUTTONS):
            keys.append(movie.get_key(i))
        
        action_sequence.append(keys)
        obs_sequence.append(observation)

        step_count += 1

        observation, _rew, _done, _info = env.step(keys)
        observation = adjust_obs(observation)
        saved_state = env.em.get_state()
        env.render()

    print("Saving dataset for batch {}".format(batch_count))
    np.save('./data/obs_data_' + game + '_' + state + '_' + str(batch_count), obs_batch)
    np.save('./data/action_data_' + game + '_' + state + '_' + str(batch_count), action_batch)

    env.close()

train = pd.read_csv('sonic-train.csv')
validation = pd.read_csv('sonic-validation.csv')

for row in train.iterrows():
    game = row[1]['game']
    state = row[1]['state']
    savetonumpy(game, state)