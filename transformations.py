# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:28:01 2019

@author: Tatiana
"""

def are_dead(x):
    
    '''
    counts number of dead players among indicated health columns
    x - columns of health
    
    '''
    ans = 0 
    for el in x:
        if el == 0:
            ans += 1
    return ans


def in_opponent_base(x,y,opponent = 'Radiant'):

    '''
    returns indicator function of whether the player is in the base of the opponent
    opponent = {'Radiant', 'Dire'} - opponent team
    
    '''
#     x = coordinates[0]
#     y = coordinates[1]
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
        return None