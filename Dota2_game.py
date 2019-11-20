# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:26:06 2019
 
"""

# import libraries
import pandas as pd
import numpy as np
import ujson as json
import os
from tqdm import tqdm_notebook
import collections
import warnings
warnings.filterwarnings('ignore')

#path for loading data, change this as needed
PATH = ''

#json data

def read_matches(matches_file):
    
    MATCHES_COUNT = {
        'test_matches.jsonl': 10000,
        'train_matches.jsonl': 39675,
    }
    _, filename = os.path.split(matches_file)
    total_matches = MATCHES_COUNT.get(filename)
    
    with open(matches_file) as fin:
        for line in tqdm_notebook(fin, total=total_matches):
            yield json.loads(line)
            


MATCH_FEATURES = [
    ('game_time', lambda m: m['game_time']),
    ('game_mode', lambda m: m['game_mode']),
    ('lobby_type', lambda m: m['lobby_type']),
    ('objectives_len', lambda m: len(m['objectives'])),
    ('chat_len', lambda m: len(m['chat'])),
]

PLAYER_FIELDS = [
    'hero_id',    
    'kills',
    'deaths',
    'assists',
    'denies',    
    'gold',
    'lh',
    'xp',
    'health',
    'max_health',
    'max_mana',
    'level',
    'x',
    'y',    
    'stuns',
    'creeps_stacked',
    'camps_stacked',
    'rune_pickups',
    'firstblood_claimed',
    'teamfight_participation',
    'towers_killed',
    'roshans_killed',
    'obs_placed',
    'sen_placed'
]

def extract_features_csv(match):
    row = [
        ('match_id_hash', match['match_id_hash']),
    ]
    
    for field, f in MATCH_FEATURES:
        row.append((field, f(match)))
        
    for slot, player in enumerate(match['players']):
        if slot < 5:
            player_name = 'r%d' % (slot + 1)
        else:
            player_name = 'd%d' % (slot - 4)

        for field in PLAYER_FIELDS:
            column_name = '%s_%s' % (player_name, field)
            row.append((column_name, player[field]))
        row.append((f'{player_name}_ability_level', len(player['ability_upgrades'])))
        row.append((f'{player_name}_max_hero_hit', player['max_hero_hit']['value']))
        row.append((f'{player_name}_purchase_count', len(player['purchase_log'])))
        row.append((f'{player_name}_count_ability_use', sum(player['ability_uses'].values())))
        row.append((f'{player_name}_damage_dealt', sum(player['damage'].values())))
        row.append((f'{player_name}_damage_received', sum(player['damage_taken'].values())))
            
    return collections.OrderedDict(row)
    
def extract_targets_csv(match, targets):
    return collections.OrderedDict([('match_id_hash', match['match_id_hash'])] + [
        (field, targets[field])
        for field in ['game_time', 'radiant_win', 'duration', 'time_remaining', 'next_roshan_team']
    ])

# read train data from json
print('Reading train data from json files...')
df_new_features = []
df_new_targets = []

for match in read_matches(os.path.join(PATH,'train_matches.jsonl')):
    match_id_hash = match['match_id_hash']
    features = extract_features_csv(match)
    targets = extract_targets_csv(match, match['targets'])
    
    df_new_features.append(features)
    df_new_targets.append(targets)

df_new_features = pd.DataFrame.from_records(df_new_features).set_index('match_id_hash')
df_new_targets = pd.DataFrame.from_records(df_new_targets).set_index('match_id_hash')

print('Reading test data from json files...')
test_new_features = []
for match in read_matches(os.path.join(PATH,'test_matches.jsonl')):
    match_id_hash = match['match_id_hash']
    features = extract_features_csv(match)
    
    test_new_features.append(features)
test_new_features = pd.DataFrame.from_records(test_new_features).set_index('match_id_hash')

train_df = df_new_features
test_df = test_new_features
train_full = train_df.merge(pd.DataFrame(df_new_targets['radiant_win']),how='outer',left_index=True,right_index=True)

# Transforming the data

# combining test and train data for a unified transformation
idx_split = train_df.shape[0]
full_df = pd.concat([train_df, test_df])
# creating a new dataframe for feature engineering
new_features = pd.DataFrame(index=full_df.index)

# Buling features based on 'hero_id'
print('Making features from hero_id...')
#hero id columns names
ls_r_hero_id = ['r{}_hero_id'.format(i) for i in range(1,6)]
ls_d_hero_id = ['d{}_hero_id'.format(i) for i in range(1,6)]
ls_hero_id = ls_r_hero_id + ls_d_hero_id
#sub data frame of hero ids and target
hero_ids = train_full[ls_hero_id +['radiant_win']]
hero_ids_rad_win = hero_ids[hero_ids['radiant_win'] == True] #rad wins
hero_ids_rad_lose = hero_ids[hero_ids['radiant_win'] == False] #rad loses
winning_hero_ids1 = hero_ids_rad_win[ls_r_hero_id]
winning_hero_ids2 = hero_ids_rad_lose[ls_d_hero_id]
losing_hero_ids1 = hero_ids_rad_win[ls_d_hero_id]
losing_hero_ids2 = hero_ids_rad_lose[ls_r_hero_id]
winning_hero_ids1.rename(columns = {'r1_hero_id':'1_id', 'r2_hero_id':'2_id', 
                              'r3_hero_id':'3_id','r4_hero_id':'4_id',
                                    'r5_hero_id':'5_id'}, inplace = True)
winning_hero_ids2.rename(columns = {'d1_hero_id':'1_id', 'd2_hero_id':'2_id', 
                              'd3_hero_id':'3_id','d4_hero_id':'4_id',
                                    'd5_hero_id':'5_id'}, inplace = True)
losing_hero_ids1.rename(columns = {'d1_hero_id':'1_id', 'd2_hero_id':'2_id', 
                              'd3_hero_id':'3_id','d4_hero_id':'4_id',
                                    'd5_hero_id':'5_id'}, inplace = True)
losing_hero_ids2.rename(columns = {'r1_hero_id':'1_id', 'r2_hero_id':'2_id', 
                              'r3_hero_id':'3_id','r4_hero_id':'4_id',
                                    'r5_hero_id':'5_id'}, inplace = True)
#for all games, df of winner's hero ids only
winning_hero_ids = pd.concat([winning_hero_ids1, winning_hero_ids2], axis=0)
#for all games, df of loser's hero ids only
losing_hero_ids = pd.concat([losing_hero_ids1, losing_hero_ids2], axis=0)

#by hero, in how many games did the hero win / lose
winning_hero_counts = winning_hero_ids['1_id'].value_counts().sort_index() + winning_hero_ids['2_id'].value_counts().sort_index() + winning_hero_ids['3_id'].value_counts().sort_index() + winning_hero_ids['4_id'].value_counts().sort_index() + winning_hero_ids['5_id'].value_counts().sort_index()
losing_hero_counts = losing_hero_ids['1_id'].value_counts().sort_index() + losing_hero_ids['2_id'].value_counts().sort_index() + losing_hero_ids['3_id'].value_counts().sort_index() + losing_hero_ids['4_id'].value_counts().sort_index() + losing_hero_ids['5_id'].value_counts().sort_index()

#to dictionary, key is hero id, value is win / loss count
winning_hero_counts = winning_hero_counts.sort_values()
winning_hero_dict = winning_hero_counts.to_dict()
losing_hero_counts = losing_hero_counts.sort_values()
losing_hero_dict = losing_hero_counts.to_dict()

#now subtract wins - loses by hero (this will be one feature)
hero_counts_win_minus_lose = winning_hero_counts.sort_index() - losing_hero_counts.sort_index()
diff_hero_dict = hero_counts_win_minus_lose.to_dict()

#normalize by dividing by total number of games played (this is another feature)
from collections import Counter
total_games_dict = Counter(winning_hero_dict) + Counter(losing_hero_dict)
hero_id_normalize_dict = {k: (diff_hero_dict[k] / total_games_dict[k]) for k in diff_hero_dict}

#add the two new features to the data frame
for col in ls_hero_id:
    full_df[col+'success'] = full_df[col].map(diff_hero_dict)
    full_df[col+'norm'] = full_df[col].map(hero_id_normalize_dict)
    
# transforming 'health' column
print('Making features from health...')   
def are_dead(x):
    ans = 0 
    for el in x:
        if el == 0:
            ans += 1
    return ans

# add number of dead players at the end of the game in each team
d_health = ['d{}_health'.format(i) for i in range(1,6)]
new_features['d_dead'] = full_df[d_health].apply(are_dead, axis = 1)
r_health = ['r{}_health'.format(i) for i in range(1,6)]
new_features['r_dead'] = full_df[r_health].apply(are_dead, axis = 1)

# calculate the proportion of health
percentage_health = 0
for j in range(5):
    max_health = '{}_max_health'.format(d_health[j].split('_')[0])
    health = 'd{}_health'.format(j+1)
    ph = full_df[health]/full_df[max_health]
    percentage_health += 1/5*ph
new_features['d_health_avg'] = percentage_health

percentage_health = 0
for j in range(5):
    max_health = '{}_max_health'.format(r_health[j].split('_')[0])
    health = 'r{}_health'.format(j+1)
    ph = full_df[health]/full_df[max_health]
    percentage_health += 1/5*ph

new_features['r_health_avg'] = percentage_health

# transform the coordinates of the players
print('Making features from coordinates...')
def in_opponent_base(x,y,opponent = 'Radiant'):

    '''
returns indicator function of whether the player is in the base of the opponent
opponent = {'Radiant', 'Dire'} - opponent team
    
    '''
    radiant_base_x = 96
    radiant_base_y = 100

    dire_base_x = 156
    dire_base_y = 156

    if opponent == 'Radiant':
        if x <= radiant_base_x and y <= radiant_base_y:
            return 1
        else:
            return 0
    elif opponent == 'Dire':
        if x >= dire_base_x and y >= dire_base_y:
            return 1
        else:
            return 0
    else:
        return NaN
    
r_x = ['r{}_x'.format(j) for j in range(1,6)]
r_y = ['r{}_y'.format(j) for j in range(1,6)]
d_x = ['d{}_x'.format(j) for j in range(1,6)]
d_y = ['d{}_y'.format(j) for j in range(1,6)]

# getting indicator function for each player
for j in range(5):
    rx = r_x[j]
    ry = r_y[j]
    dx = d_x[j]
    dy = d_y[j]
    new_features['r{}_in_d_base'.format(j+1)] = full_df.loc[:,[rx,ry]].apply(lambda x: in_opponent_base(x = x[rx], y = x[ry], opponent = 'Dire'), axis = 1)
    new_features['d{}_in_r_base'.format(j+1)] = full_df.loc[:,[dx,dy]].apply(lambda x: in_opponent_base(x = x[dx], y = x[dy], opponent = 'Radiant'), axis = 1)
    new_features['r{}_in_r_base'.format(j+1)] = full_df.loc[:,[rx,ry]].apply(lambda x: in_opponent_base(x = x[rx], y = x[ry], opponent = 'Radiant'), axis = 1)
    new_features['d{}_in_d_base'.format(j+1)] = full_df.loc[:,[dx,dy]].apply(lambda x: in_opponent_base(x = x[dx], y = x[dy], opponent = 'Dire'), axis = 1)
   
# getting total amount of players of each team at each base
r_in_d_base = pd.Series(0,index = full_df.index)
d_in_r_base = pd.Series(0,index = full_df.index)
r_in_r_base = pd.Series(0,index = full_df.index)
d_in_d_base = pd.Series(0,index = full_df.index)


# aggregate features amoung teams
print('Agreggating features across the team...')
# Create a feature that is sum of r1 + ... + r5 for all r features, and same for all d features.
# Features that do not make sence are removed.
new_feats = [i.replace("1","")  for i in full_df.columns.values if (i.startswith('r1')==1 or i.startswith('d1')==1)] 
rem_feat = ['r_health', 'r_max_health','r_level', 'r_x', 'r_y', 'r_teamfight_participation', 'd_health','d_max_health', 'd_level', 'd_x', 'd_y', 'd_teamfight_participation']

for feat in rem_feat:
    new_feats.remove(feat)

for feature in new_feats:
    feat_names = [(feature[0] + '{}_'.format(i) +feature[2:]) for i in range(1,6)]
    new_features[feature] = full_df.loc[:,feat_names].sum(axis=1)

#aggregate level of players
print('Building features from level...')   
d_levels = ['d{}_level'.format(j) for j in range(1,6)] 
d_avg_level = full_df.loc[:,d_levels].mean(axis = 1)
d_min_level = full_df.loc[:,d_levels].min(axis = 1)
d_max_level = full_df.loc[:,d_levels].max(axis = 1)
new_features['d_avg_level'] = d_avg_level
new_features['d_min_level'] = d_min_level
new_features['d_max_level'] = d_max_level


r_levels = ['r{}_level'.format(j) for j in range(1,6)] 
r_avg_level = full_df.loc[:,r_levels].mean(axis = 1)
r_min_level = full_df.loc[:,r_levels].min(axis = 1)
r_max_level = full_df.loc[:,r_levels].max(axis = 1)
new_features['r_avg_level'] = r_avg_level
new_features['r_min_level'] = r_min_level
new_features['r_max_level'] = r_max_level

# adding old features to new ones
new_features = new_features.merge(full_df[['chat_len','game_time','game_mode','lobby_type']], how='outer', left_index=True, right_index=True)

# assisting ratio
new_features['ratio_assists'] = new_features['r_assists']/(new_features['d_assists']+1)

# taking logs of skewed features
print('Making log-transform for skewed variables...')
log_tags = ['_kills', '_gold', '_lh', '_xp', '_max_mana', '_creeps_stacked', 
               '_camps_stacked','_rune_pickups', '_purchase_count','_count_ability_use', '_damage_dealt',
            '_damage_received', '_max_hero_hit']
prefix = ['r', 'd']
col_for_log = []
for tag in log_tags:
    for p in prefix:
        col_for_log.append(p+tag)
col_for_log += ['chat_len', 'game_time','ratio_assists']

import math

for col in col_for_log:
    new_log_index = 'log_{}'.format(col)
    new_log_col = new_features[col].apply(lambda x: math.log(x+1))
    new_features[new_log_index] = new_log_col
    
# take squares of some features
    
col_for_sq = ['r_gold', 'r_xp', 'd_gold', 'd_xp']

for col in col_for_sq:
    new_sq_index = 'sq_{}'.format(col)
#     print(new_features[col].min())
    new_sq_col = new_features[col].apply(lambda x: x**2)
    new_features[new_sq_index] = new_sq_col

# splitting back the data
train = new_features.iloc[:idx_split, :]
test = new_features.iloc[idx_split:, :]
y = df_new_targets['radiant_win']
  
#define categorical and numerical features
cat_feats = ['game_mode','lobby_type', 'r_firstblood_claimed','d_firstblood_claimed']
cat_feats  += ['r{}_in_d_base'.format(j) for j in range(1,6)]+['r{}_in_r_base'.format(j) for j in range(1,6)]
cat_feats += ['d{}_in_d_base'.format(j) for j in range(1,6)]+['d{}_in_r_base'.format(j) for j in range(1,6)]

num_feats = [col for col in train.columns if col not in cat_feats]
numerical_columns =[]
for j in range(len(train.columns)):
    if train.columns[j] in num_feats:
        numerical_columns.append(j)
        

#building the pipeline
print('Building pipeline: OneHotEncoder, StandardScaler, LogisticRegression...')
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer

preprocess = make_column_transformer(
    (StandardScaler(), num_feats),
    (OneHotEncoder(),cat_feats))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score


logreg = LogisticRegression(C = 10,penalty = 'l1',random_state = 150 )
pipe = make_pipeline(preprocess, logreg)

#fit the model
print('Fitting the model...')
pipe.fit(train,y)

#making prediction
print('Making predictions on test data...')
prediction = pipe.predict_proba(test)[:, 1]

df_submission = pd.DataFrame({'radiant_win_prob': prediction},index=test.index)

import datetime
submission_filename = 'submission_{}.csv'.format(
    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
df_submission.to_csv(submission_filename)
print('Submission saved to {}'.format(submission_filename))
