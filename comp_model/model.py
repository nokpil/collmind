
import sys
import gc
from argparse import ArgumentParser
from datetime import datetime
from copy import deepcopy

import numpy as np
from numpy.random import rand
import random
import scipy.spatial as spatial
import scipy.stats as stats
import pandas as pd

import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interact_manual

from collections import defaultdict, Counter
from copy import deepcopy
import pickle
from os.path import join
from tqdm.notebook import tqdm
import networkx as nx
import powerlaw

from cProfile import Profile
from pstats import SortKey, Stats

## utility functions

def harmonic_approx(N):
    return np.log(N) + 0.57721 + 1/(2*N) - 1/(12*N**2)

class rng():
    def __init__(self, num_samples=int(1e7)):
        self.num_samples = num_samples
        self.samples = np.random.uniform(0, 1, num_samples)
        self.counter = 0
        
class powerlaw_rng(rng):
    def __init__(self, num_samples=int(1e7)):
        super().__init__(num_samples)
        
    def sample(self, xmin, xmax):
        self.counter+=1
        return xmin * (xmax / xmin) ** self.samples[self.counter%self.num_samples]
    
class lognormal_rng(rng):
    def __init__(self, num_samples=int(1e7)):
        super().__init__(num_samples)
        
    def sample(self, mu, sigma):
        self.counter+=2
        return np.exp(mu + sigma * np.sqrt(-2 * np.log(self.samples[self.counter%self.num_samples])) * np.cos(2 * np.pi * self.samples[(self.counter+1)%self.num_samples]))
    
class exponential_rng(rng):
    def __init__(self, num_samples=int(1e7)):
        super().__init__(num_samples)
        
    def sample(self, Lambda, xmin, xmax):
        self.counter+=1
        return -1/Lambda * np.log(1 - self.samples[self.counter%self.num_samples] * (1 - np.exp(-Lambda * (xmax - xmin)))) + xmin

class stretched_exponential_rng(rng):
    def __init__(self, num_samples=int(1e7)):
        super().__init__(num_samples)
        
    def sample(self, Lambda, beta, xmin):
        self.counter+=1
        return (1/Lambda)* ((Lambda*xmin)**beta-np.log(1-self.samples[self.counter%self.num_samples]))**(1/beta)
        
# update weight in symmetric weight matrix
def update_weight(W, i, j, weight):
    W[i, j] = weight
    W[j, i] = weight
    return W

# given list A, return all pairs of objects
def get_pairs(A):
    return [(A[i], A[j]) for i in range(len(A)) for j in range(i+1, len(A))]

def power_law_pmf(alpha, xmin, xmax):
    x = np.arange(xmin, xmax, dtype='float')
    pmf = x**(-alpha)
    pmf /= pmf.sum()
    
    return pmf

def power_law_sample(alpha, xmin, xmax, size):
    # Generate uniform random numbers
    u = np.random.uniform(0, 1, size)

    # Inverse transform sampling for power-law distribution
    if alpha == 1:
        samples = np.exp(u * (np.log(xmax) - np.log(xmin)) + np.log(xmin))
    else:
        samples = ((u * (xmax**(-alpha + 1) - xmin**(-alpha + 1)) + xmin**(-alpha + 1)) ** (1 / (-alpha + 1)))
        
    return list(samples)

def exp_func(x, a, b):
    return a * np.exp(-b * x)

def get_ranking(scores):
    temp = (-scores).argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(scores))+1
    return ranks

def events_topic_counter(events, topic_num):
    counter_list = []
    for i in range(events.shape[1]):
        c = Counter(events[:, i])
        for j in range(topic_num):
            if j not in c.keys():
                c[j] = 0
        counter_list.append(np.array(sorted(c.items(), key=lambda x: x[0]))[:, 1])
    return np.array(counter_list)

def adjust_to_sum(vec, target_sum):
    rounded_vec = np.round(vec).astype(int)
    diff = target_sum - np.sum(rounded_vec)
    
    # Sort indices based on the fractional parts
    fractional_parts = vec - np.floor(vec)
    indices_sorted = np.argsort(fractional_parts)[::-1]
    
    for i in range(abs(diff)):
        if diff > 0:
            rounded_vec[indices_sorted[i]] += 1
        elif diff < 0:
            rounded_vec[indices_sorted[i]] -= 1
    
    return rounded_vec

color_list = ['#e84d8a', '#feb326', '#60bd68', '#64c5eb', '#7f58af']

# distribution / functions

## Initial distribution of topic graph

# given topic number, assign proper probability according to exponential distribution with exp(-exponent * x)
def freq_dist(topic_num, alpha=1):
    pmf = power_law_pmf(alpha=alpha, xmin=1, xmax=topic_num+1)
    return pmf

# given topic number, assign proper probability according to lognormal distribution with loc and scale
def weight_dist(topic_num, loc=-0.6, scale=1.0):
    # 0.1177 -0.6500 0.9985
    # mean = loc + scale * exp(ln(scale) + 0.5 * s^2)
    s, loc, scale = 0.12, -0.65, 1
    weight = stats.lognorm.rvs(s=s, loc=loc, scale=scale, size=(topic_num, topic_num))
    np.fill_diagonal(weight, 0)
    weight = np.clip(weight, 0, 0.8, out=weight)
    return  np.triu(weight) + np.triu(weight, 1).T

# intervention functions

'''
IV_dict = {'iv_type': iv_type,
            'iv_t1': iv_t1,
            'iv_s1': iv_s1,
            'iv_t2': iv_t2,
            'iv_s2': iv_s2,
            'iv_rank': iv_rank,
            'iv_tier': iv_tier}

'''

# intervention 0. World event
def IV_0(orig_general_frequency, orig_general_weight, IV_dict, time):
    
    # iv_s1 = frequency modifier (at t1, pulse-like)
    # iv_s2 = weight modifier (at t1, pulse-like)
    
    if time < IV_dict['iv_t1']:
        return orig_general_frequency, orig_general_weight

    modified_general_frequency = deepcopy(orig_general_frequency)
    modified_general_weight = deepcopy(orig_general_weight)
    
    iv_type = IV_dict['iv_type']
    
    if iv_type[1] == '1':
        s1 = IV_dict['iv_s1'] if time == IV_dict['iv_t1'] else 1
        s2 = IV_dict['iv_s2'] if time == IV_dict['iv_t1'] else 1
        
    elif iv_type[1] == '2':
        decay = np.exp(-0.5*(time-IV_dict['iv_t1']))
        s1 = 1 + (IV_dict['iv_s1']-1) * decay
        s2 = 1 + (IV_dict['iv_s2']-1) * decay
        
    modified_general_frequency[IV_dict['iv_rank']] = s1 * orig_general_frequency[IV_dict['iv_rank']]
    modified_general_weight[IV_dict['iv_rank']] =  s2 * orig_general_weight[IV_dict['iv_rank']]
    modified_general_weight[:, IV_dict['iv_rank']] = s2 * orig_general_weight[:, IV_dict['iv_rank']]
    
    return modified_general_frequency, modified_general_weight
    
# intervention 1. Alignment (filter strength)
def IV_1(strength, IV_dict, time):
    
    # iv_s1 = filter strength (after t1)
    # iv_s2 = filter strength (after t2)
    
    if time < IV_dict['iv_t1']:                                     
        return strength
                                 
    return IV_dict['iv_s1'] if time < IV_dict['iv_t2'] else IV_dict['iv_s2']
                            
# intervention 2 : Amplification
def IV_2(general_frequency, IV_dict, time):
    
    # iv_s1 = comment frequency multiplier (after t1)
    # iv_s2 = comment frequency multiplier (after t2)
    
    if time < IV_dict['iv_t1']:                                   
        return general_frequency
 
    modified_frequency = deepcopy(general_frequency) 
    
    if iv_type[1] == '1':
        modified_frequency[IV_dict['iv_rank']] *= IV_dict['iv_s1'] if time < IV_dict['iv_t2'] else IV_dict['iv_s2']
    else:
        modified_frequency[IV_dict['iv_rank']] *= IV_dict['iv_s1'] * np.exp(-IV_dict['iv_s2'] * (time - IV_dict['iv_t1']))
        
    return modified_frequency / modified_frequency.sum()
    
# intervention 3 : Reframing / Distortion
def IV_3(filtered_events, IV_dict, time):
    
    # iv_s1 = replace prob. (after t1)
    # iv_s2 = replace prob. (after t2)
    
    if time < IV_dict['iv_t1']:
        return filtered_events
    
    modified_events = deepcopy(filtered_events)
    replace_mask = np.random.binomial(1, (IV_dict['iv_s1'] if time < IV_dict['iv_t2'] else IV_dict['iv_s2']), size=modified_events.shape[0])
    modified_events[:, IV_dict['iv_tier']-1][replace_mask.astype(bool)] = IV_dict['iv_rank']
    return modified_events

# intervention 4. Membership turnover (memory strength)
def IV_4(strength, IV_dict, time):
    
    # iv_s1 = filter strength (after t1)
    # iv_s2 = filter strength (after t2)
    
    if time < IV_dict['iv_t1']:                                     
        return strength
                                 
    return IV_dict['iv_s1'] if time < IV_dict['iv_t2'] else IV_dict['iv_s2']


# intervention 5: Troll
def IV_5(raw_frequency, comment_num, event, IV_dict, time):
    # iv_s1 = troll strength (after t1)
    # iv_s2 = 4-1 : troll strength (after t2)

    if time < IV_dict['iv_t1']:
        return raw_frequency, comment_num
    
    iv_type = IV_dict['iv_type']
    iv_rank = IV_dict['iv_rank']
    modified_raw_frequency = deepcopy(raw_frequency)
    modified_comment_num = comment_num
    
    modifier = IV_dict['iv_s1'] if time < IV_dict['iv_t2'] else IV_dict['iv_s2']
    
    modified_comment_num += raw_frequency[iv_rank] * (modifier - 1)
    modified_raw_frequency[iv_rank] *= modifier
        
    return modified_raw_frequency, modified_comment_num


# intervention 6 : Troll + Counterspeech
def IV_6(raw_frequency, comment_num, event, IV_dict, time):
    # iv_s1 = troll strength (after t1)
    # iv_s2 = counterspeech strength (after t2), troll strength keeps iv_s1

    if time < IV_dict['iv_t1']:
        return raw_frequency, comment_num
    
    iv_type = IV_dict['iv_type']
    iv_rank = IV_dict['iv_rank']
    modified_raw_frequency = deepcopy(raw_frequency)
    modified_comment_num = comment_num
    
    modifier = IV_dict['iv_s1']
    modified_comment_num += raw_frequency[iv_rank] * (modifier - 1)
    modified_raw_frequency[iv_rank] *= modifier
    
    if time >= IV_dict['iv_t2']:
        
        if iv_type[1] == '1':  # all articles
            if iv_rank in event:
                for rank in event:
                    if iv_rank != rank:
                        modified_comment_num += raw_frequency[rank] * (IV_dict['iv_s2'] - 1)
                        modified_raw_frequency[rank] *= IV_dict['iv_s2']
            else:
                for rank in event:
                    modified_comment_num += raw_frequency[rank] * (IV_dict['iv_s2'] - 1)
                    modified_raw_frequency[rank] *= IV_dict['iv_s2']
        
        if iv_type[1] == '2':  # only corresponding articles
            if iv_rank in event:
                for rank in event:
                    if iv_rank != rank:
                        modified_comment_num += raw_frequency[rank] * (IV_dict['iv_s2'] - 1)
                        modified_raw_frequency[rank] *= IV_dict['iv_s2']
            else:
                pass
            
        if iv_type[1] == '3':  # only off-corresponding articles
            if iv_rank in event:
                pass
            else:
                for rank in event:
                    modified_comment_num += raw_frequency[rank] * (IV_dict['iv_s2'] - 1)
                    modified_raw_frequency[rank] *= IV_dict['iv_s2']
                    
    return modified_raw_frequency, modified_comment_num



### Main classes

class topic_graph:
    def __init__(self, topic_num, filter_strength=0.2, memory_strength=0.8, sampling_ratio_list=[0.5, 0.5], power_constant_list=[-0.4, -0.2, -0.1], IV_dict={}, frequency=None, weight=None):
        
        self.WEIGHT_MIN_VAL = 1e-6
        self.WEIGHT_MAX_VAL = 0.8
        self.WEIGHT_LR = 10
        self.WEIGHT_NOISE_STD =  1e-3
        self.EVENT_MEMORY_STRENGTH = 0.0
        
        self.topic_num = topic_num
        self.filter_strength = filter_strength
        self.memory_strength = memory_strength
        
        self.sampling_ratio_list = sampling_ratio_list
        self.power_constant_list = power_constant_list
        self.IV_dict=IV_dict
        
        self.harmonic_num = harmonic_approx(topic_num)
        
        # initializations
        if frequency is None:
            self.frequency = freq_dist(topic_num)
            self.default_freq = deepcopy(self.frequency)
        else:
            self.frequency = frequency
            self.default_freq = freq_dist(topic_num)
            
        if weight is None:
            self.weight = weight_dist(topic_num)
        else:
            self.weight = weight
        
    def copy_from(self, topic_graph):
        self.set_frequency(topic_graph.frequency)
        self.set_weight(topic_graph.weight)
    
    ### Frequency
    
    def set_frequency(self, frequency):
        # set frequency from external source and fit it into default frequency
        self.frequency = self.default_freq[(-frequency).argsort().argsort()]

    def perturb_frequency(self, perturbation):  # multiplicative
        assert perturbation.shape == self.frequency.shape
        self.frequency *= perturbation
        self.frequency = self.default_freq[(-self.frequency).argsort().argsort()]
        
    def perturb_frequency_lognormal(self, std):
        perturbation = np.random.lognormal(0, std, self.frequency.shape)
        self.perturb_frequency(perturbation)
        
    def get_ranking(self):
        return get_ranking(self.frequency)
    
    def get_normalized_ranking(self):
        return get_ranking(self.frequency)/self.topic_num
        
    # Edge weights (similarity)
    
    def set_weight(self, weight):
        # set weight from external source
        np.fill_diagonal(weight, 0)
        np.clip(weight, self.WEIGHT_MIN_VAL, self.WEIGHT_MAX_VAL, out=weight)
        self.weight = deepcopy(weight)
    
    def perturb_weight(self, perturbation):   # additive
        assert perturbation.shape == self.weight.shape
        self.set_weight(self.weight + perturbation)
        
    def perturb_weight_gaussian(self, std):
        perturbation = np.random.normal(0, std, self.weight.shape)
        self.perturb_weight(perturbation)
        
    ### event generation (for general)
    
    def generate_events(self, event_num, event_topic_num, previous_events=None):
        
        events = np.zeros((event_num, event_topic_num)).astype(int)
        normalized_ranking = self.get_normalized_ranking()

        if previous_events is None:
            events[:, 0] = np.random.choice(self.topic_num, p=-np.log(normalized_ranking)/np.sum(-np.log(normalized_ranking)), size=event_num)

            for j in range(1, event_topic_num):
                for i in range(event_num):
                    tmp_normalized_ranking = np.array([normalized_ranking[k] for k in range(self.topic_num) if k not in events[i][:j]])
                    events[i][j] = np.random.choice([k for k in range(self.topic_num) if k not in events[i][:j]], p=-np.log(tmp_normalized_ranking)/np.sum(-np.log(tmp_normalized_ranking)))

            return events, None
        
        else:
            new_events = np.random.choice(self.topic_num, p=-np.log(normalized_ranking)/np.sum(-np.log(normalized_ranking)), size=(event_num, event_topic_num))

            counter_previous = events_topic_counter(previous_events, self.topic_num)
            counter_new = events_topic_counter(new_events, self.topic_num)
            counter_interp = np.array([adjust_to_sum(self.EVENT_MEMORY_STRENGTH * counter_previous[i] + (1-self.EVENT_MEMORY_STRENGTH) * counter_new[i], event_num) for i in range(event_topic_num)])
            
            pool_list = [[num for num, freq in enumerate(counter_interp[i]) for _ in range(freq)] for i in range(len(counter_interp))]

            # Initialize list to store the triplets
            events = []
            check_retry = 0
            check_internal = 0
            # Generate triplets
            while len(events) < event_num:
                check_internal += 1
                # Randomly sample one number from each pool
                candidate = [random.choice(pool_list[i]) for i in range(event_topic_num)]
                
                # Check if all three numbers are distinct
                if len(candidate) == len(set(candidate)):
                    events.append(candidate)
                    for i in range(event_topic_num):
                        pool_list[i].remove(candidate[i])
                        
                if check_internal > 2*event_num:
                    # reset
                    pool_list = [[num for num, freq in enumerate(counter_interp[i]) for _ in range(freq)] for i in range(len(counter_interp))]
                    events=[]
                    check_internal = 0
                    check_retry += 1

            events = np.array(events)
            return events, (counter_previous, counter_new, counter_interp, check_retry, check_internal)
    
        ## Frequency update process

    ### Comment number distribution

    def comment_num_sampler(self, events):
        pass

    ### Frequency multiplier distribution

    def frequency_mult_sampler(self, event, comment_num):
        pass

    def func_generate_freq(self, event, comment_num, time):
        frequency = self.frequency
        frequency_multipliers = self.frequency_mult_sampler(event, comment_num)
        
        # current : (1 - \sum m_i * f_i) = \sum_{j!=i} f_j
        # goal : (1 - \sum m_i * f_i) = \sum_{j!=i} m_j * f_j
        
        raw_frequency = frequency * (1-min(np.sum(frequency_multipliers * frequency[event]), 1)) / (1-np.sum(frequency[event]))
            
        off_corr_freq = raw_frequency
        corr_freq = np.zeros_like(raw_frequency)
        for t, m in zip(event, frequency_multipliers):
            raw_frequency[t] = m * frequency[t]
            corr_freq[t] = m * frequency[t]
            
        # sub_peak_multiplied_frequency = 0  # Not yet implemented
        #raw_frequency = raw_frequency / np.sum(raw_frequency) # re-normalization
        
        return raw_frequency, np.sum(frequency_multipliers * frequency[event]), (corr_freq, off_corr_freq)
        
    def func_update_freq(self, raw_freq, time):
        norm_freq = raw_freq / raw_freq.sum()
        if self.IV_dict['iv_type'] == '41':  # IV_4, membership turnover (memory strength)
            memory_strength = IV_4(self.memory_strength, self.IV_dict, time)
        else:
            memory_strength = self.memory_strength
        new_freq = memory_strength * self.frequency + (1 - memory_strength) * norm_freq
        self.set_frequency(new_freq)

    ## Weight update process

    def func_generate_adj(self, event, comment_num):
        adj = np.zeros((self.topic_num, self.topic_num))
        for i, j in get_pairs(event):
            adj[i, j] += comment_num
            adj[j, i] += comment_num
        return adj

    def func_update_weight(self, raw_adj, comment_num_list):

        weight = self.weight
        event_topic_num = events.shape[-1]

        norm_adj = raw_adj / (np.sum(comment_num_list) * (event_topic_num) * (event_topic_num-1) / 2)
        hebb_term = (self.WEIGHT_MAX_VAL-weight) * (norm_adj + np.random.normal(0, self.WEIGHT_NOISE_STD, weight.shape))
        decaying_term = (np.sum(hebb_term) / np.sum(weight)) * weight

        delta_weight = self.WEIGHT_LR * (hebb_term - decaying_term)
        updated_weight = weight + delta_weight
        self.set_weight(updated_weight)
    
    def update_topic_graph(self, events, time):
        
        iv_type = self.IV_dict['iv_type']
        
        raw_frequency = np.zeros(self.topic_num)
        raw_adj = np.zeros((self.topic_num, self.topic_num))
        mult_frequency_list = np.zeros(len(events))

        comment_num_list = self.comment_num_sampler(events)
        corr_list = []
        
        if iv_type == '51':  # IV_5, Trolls
            for i in range(len(events)):
                event = events[i]
                comment_num = comment_num_list[i]
                raw_f, mult_f, corr = self.func_generate_freq(event, comment_num, time)
                corr_list.append(corr)
                
                rf, comment_num = IV_5(raw_f * comment_num, comment_num, event, IV_dict, time)
                raw_frequency += rf
                comment_num_list[i] = comment_num

                raw_adj += self.func_generate_adj(event, comment_num)
        
        elif iv_type[0] == '6':  # IV_6, Trolls + Counterspeech
            for i in range(len(events)):
                event = events[i]
                comment_num = comment_num_list[i]
                raw_f, mult_f, corr = self.func_generate_freq(event, comment_num, time)
                corr_list.append(corr)
                
                rf, comment_num = IV_6(raw_f * comment_num, comment_num, event, IV_dict, time)
                raw_frequency += rf
                comment_num_list[i] = comment_num

                raw_adj += self.func_generate_adj(event, comment_num)
                
        else:  # Every other normal cases
            for i in range(len(events)):
                event = events[i]
                comment_num = comment_num_list[i]
                raw_f, mult_f, corr = self.func_generate_freq(event, comment_num, time)
                corr_list.append(corr)
                
                raw_frequency += raw_f * comment_num
                mult_frequency_list[i] = mult_f * comment_num
                
                raw_adj += self.func_generate_adj(event, comment_num)
        
        self.func_update_freq(raw_frequency, time)  # via normalization
        self.func_update_weight(raw_adj, comment_num_list)  # via Hebbian learning
        
        return raw_frequency, raw_adj, comment_num_list, mult_frequency_list, corr_list


class topic_graph_simple(topic_graph):
    
    def __init__(self, topic_num, filter_strength=0.2, memory_strength=0.8, sampling_ratio_list=[0.5, 0.5], power_constant_list=[-0.4, -0.2, -0.1], IV_dict={}, frequency=None, weight=None, rng_list=[]):
        super().__init__(topic_num, filter_strength, memory_strength, sampling_ratio_list, power_constant_list, IV_dict, frequency, weight)
        self.zero_ratio_list = [0.7, 0.9, 0.9]
        
        self.comment_mu = -9
        self.comment_sigma = 1.4
        
        self.lambda_slope_list = [5e-3, 1e-2, 2e-2]
        self.lambda_exponent = 0.8
        self.mean_total_comment_num = 1e6        # mean total comment number per timestep (month), dependent on the scale of the comment number
        
        self.rng_lognormal = rng_list[0]
        self.rng_exp = rng_list[1]
        
    ### Comment number distribution

    def comment_num_sampler(self, events):
        return np.array([self.rng_lognormal.sample(mu=self.comment_mu, sigma=self.comment_sigma) for i in range(len(events))])

    ### Frequency multiplier distribution

    def frequency_mult_sampler(self, event, comment_num):
        ranking = self.get_ranking()
        mult_list = []
        for tier in range(len(event)):
            r = ranking[event[tier]]
            normalized_ranking = r/self.topic_num
            if np.random.rand() < self.zero_ratio_list[tier] * normalized_ranking:
                mult_list.append(0)
            else:
                c = (r * (np.log(r) + np.euler_gamma))
                mult_list.append(self.rng_exp.sample(Lambda=self.lambda_slope_list[tier]/normalized_ranking**self.lambda_exponent, 
                                                         xmin= c / (comment_num * self.mean_total_comment_num),
                                                         xmax= c))
        return mult_list

    
class topic_graph_empirical(topic_graph):
    def __init__(self, topic_num, filter_strength=0.2, memory_strength=0.8, sampling_ratio_list=[0.5, 0.5], power_constant_list=[-0.4, -0.2, -0.1], IV_dict={}, frequency=None, weight=None, rng=None):
        
        super().__init__(topic_num, filter_strength, memory_strength, sampling_ratio_list, power_constant_list, IV_dict, frequency, weight)
        self.rng = rng
        
        # Following empirical data comes from Thehill, but qualitative tendency from Breitbart data is also similar (see comments).
        # Typically, 1st and 2nd/3rd shows qualitatively different distribution. 
        # Constants are rounded up to 4 decimal digits.
        
        # comment_num : [histogram] stretched exponential (Lambda, beta) (adopted from fitting with 2015~) lambda * exp(-t^beta)
        # Lambda ~ [ranking] linear (slope, intercept) 
        # beta ~ [ranking] linear (slope, intercept) 
        self.comment_lambda_slope = [8190.7294, 5051.3898, 3267.6738]   # Lambda, slope                             [1572.3855, 916.1964, 629.4866]
        self.comment_lambda_intercept = 5657.6310                       # mean value of all tier's intercept        [4607.05107]
        self.comment_beta_slope = [-0.0503, -0.0373, -0.0255]           # beta, slope                               [-0.0166, -0.01589, -0.0067]
        self.comment_beta_intercept = 0.5811                            # mean value of all tier's intercept        [0.6521]
        self.min_comment_ratio = 6e-6                                   # minimum possible comment_ratio, dependent on the scale of the comment number
        
        # zero ratio : [comment_num] exponential (slope, intercept), before/after zr_point
        # slope ~ [ranking] linear (slope, intersept) 
        # exponent ~ [ranking] exponential (slope, exponent))
        self.zr_point = 1/5                                             # nr where slope change occurs              [1/5]
        self.zss1_const_list = [1.1229, 2.1468, 1.7321]                 # zero, slope, slope,  x < zr_point         [1.3142, 2.2949, 2.3375]
        self.zsi1_const_list = [0.2995, 0.3105, 0.4482]                 # zero, slope, intercept, x < zr_point      [0.1146, 0.2297, 0.3463]
        self.zss2_const_list = [0.2323, 0.2150, 0.2025]                 # zero, slope, slope, x >= zr_point         [0.1938, 0.2795, 0.2224]
        self.zsi2_const_list = [0.4488, 0.6597, 0.7122]                 # zero, slope, intercept, x >= zr_point     [0.3622, 0.5968, 0.7106]
        self.zes1_const_list = [48838.9375, 21308.4141, 37649.1613]     # zero, exponent, slope, x < zr_point       [56264.9701, 42433.8542 42721.7776]
        self.zee1_const_list = [3.6546, 7.9685, 17.5380]                # zero, exponent, intercept, x < zr_point   [12.4497, 18.0671, 27.6995]
        self.zes2_const_list = [53625.7959, 6409.8930, 1935.8821]       # zero, exponent, slope, x >= zr_point      [14043.7855, 6758.55678, 3333.4954]
        self.zee2_const_list = [7.9152, 5.2564, 3.2722]                 # zero, exponent, slope, x >= zr_point      [4.2267, 4.0999, 3.2930]
        
        # non-zero multiplier : [histogram] stretched exponential (Lambda, beta)
        # Lambda ~ [ranking] power-law (slope ~ [comment_num] power-law (slope, exponent), exponent ~ [comment_num] power-law (slope, exponent)) 
        # beta ~ [ranking] power-law (slope ~ [comment_num] power-law (slope, exponent), exponent ~ [comment_num] power-law (slope, exponent)) 
        self.Lss_const_list = [0.1470, 1.9313, 1.8861]                  # Lambda, slope, slope                      [0.0698, 11.8541, 6.2492]
        self.Lse_const_list = [-0.2779, -0.3509, -0.271]                # Lambda, slope, exponent                   [-0.2768,-0.5645, -0.414]
        self.Les_const_list = [0.3027, 0.0073, 0.0046]                  # Lambda, exponent, slope                   [0.4545, 0.0014, 0.0002]
        self.Lee_const_list = [0.0784, 0.3873, 0.3911]                  # Lambda, exponent, exponent                [0.0513, 0.5513, 0.7113]
        self.bss_const_list = [0.2453, 0.1397, 0.1734 ]                 # beta, slope, slope                        [0.1711, 0.0641, 0.0801]
        self.bse_const_list = [0.1179, 0.1250, 0.1013]                  # beta, slope, exponent                     [0.1557, 0.2078, 0.1756]
        self.bes_const_list = [0.6527, 0.7173, 0.5598]                  # beta, exponent, slope                     [0.5598, 1.0245, 0.8681]
        self.bee_const_list = [-0.1310, -0.1118, -0.0969]               # beta, exponent, exponent                  [-0.0969, -0.1457, -0.1330]
        self.mean_total_comment_num = 1e6                               # mean total comment number per timestep (month), dependent on the scale of the comment number
        
    ### Comment number distribution

    def comment_num_sampler(self, events):
        normalized_ranking = self.get_normalized_ranking()
        comment_num_list = []
        for i in range(len(events)):
            Lambda = self.comment_lambda_intercept + np.mean(self.comment_lambda_slope * normalized_ranking[events[i]])
            beta = self.comment_beta_intercept + np.mean(self.comment_beta_slope * normalized_ranking[events[i]])
            comment_num_list.append(self.rng.sample(Lambda=Lambda, beta=beta, xmin=self.min_comment_ratio))
        return np.array(comment_num_list)

    ### Frequency multiplier distribution

    def frequency_mult_sampler(self, event, comment_num):
        ranking = self.get_ranking()
        mult_list = []
        for tier in range(len(event)):
            normalized_ranking = ranking[event[tier]]/self.topic_num
            if normalized_ranking < self.zr_point:
                zero_slope = self.zss1_const_list[tier]*normalized_ranking + self.zsi1_const_list[tier]
                zero_exp = self.zes1_const_list[tier]*np.exp(-self.zee1_const_list[tier]*normalized_ranking)
            else:
                zero_slope = self.zss2_const_list[tier]*normalized_ranking + self.zsi2_const_list[tier]
                zero_exp = self.zes2_const_list[tier]*np.exp(-self.zee2_const_list[tier]*normalized_ranking)
            if np.random.rand() < zero_slope*np.exp(-zero_exp*comment_num):
                mult_list.append(0)
            else:
                # er = 1/(er * comment_num * mean_total_comment_num) = expected comment ratio
                # xmin = 1/(er * comment_num * mean_total_comment_num)
                Lambda = (self.Lss_const_list[tier]/(comment_num**self.Lse_const_list[tier]))/(normalized_ranking**(self.Les_const_list[tier]/(comment_num**self.Lee_const_list[tier])))
                beta = (self.bss_const_list[tier]/(comment_num**self.bse_const_list[tier]))/(normalized_ranking**(self.bes_const_list[tier]/(comment_num**self.bee_const_list[tier])))
                mult_list.append(self.rng.sample(Lambda=Lambda, beta=beta, xmin=(self.harmonic_num * normalized_ranking * self.topic_num)/(comment_num*self.mean_total_comment_num)))
        return mult_list

        
class model_system:
    def __init__(self, model_type, topic_num, comm_num, init_dict, filter_strength, memory_strength, sampling_ratio_list, power_constant_list, IV_dict):

        self.model_type = model_type
        self.topic_num = topic_num
        self.comm_num = comm_num
        self.filter_strength = filter_strength
        self.memory_strength = memory_strength
        self.IV_dict = IV_dict
        self.freq_prefix = None
        
        if model_type == 'simple':
            self.rng_list = [lognormal_rng(num_samples=int(1e7)), exponential_rng(num_samples=int(1e7))]
        elif model_type == 'empirical':
            self.rng = stretched_exponential_rng(num_samples=int(1e7))
        else:
            raise ValueError('Invalid model type')

        # generate general topic space
        general = topic_graph(topic_num=topic_num)
        if init_dict['init_type'] == 'fixed':
            with open(join('data', 'fixed_dict.pkl'), 'rb') as f:
                fixed_dict = pickle.load(f)
            self.freq_fixed = fixed_dict[init_dict['init_freq_std']]['freq']
            self.weight_list_fixed = fixed_dict[init_dict['init_freq_std']]['weight']
            denominator = self.comm_num // len(self.weight_list_fixed)

        # generate community topic spaces and perturb their embeddings from general topic space
        community_list = []

        for i in range(self.comm_num):
            if self.model_type == 'simple':
                tg = topic_graph_simple(topic_num=topic_num, filter_strength=filter_strength, memory_strength=memory_strength, sampling_ratio_list=sampling_ratio_list, power_constant_list=power_constant_list, IV_dict=IV_dict, rng_list=self.rng_list)
            elif self.model_type == 'empirical':
                tg = topic_graph_empirical(topic_num=topic_num, filter_strength=filter_strength, memory_strength=memory_strength, sampling_ratio_list=sampling_ratio_list, power_constant_list=power_constant_list, IV_dict=IV_dict, rng=self.rng)
            else:
                raise ValueError('Invalid model type')
                
            tg.copy_from(general)
            if init_dict['init_type'] == 'plain':
                pass
            elif init_dict['init_type'] == 'perturb':
                tg.perturb_frequency_lognormal(init_dict['init_freq_std'])
                tg.perturb_weight_gaussian(init_dict['init_weight_std'])
            elif init_dict['init_type'] == 'fixed':
                tg.set_frequency(self.freq_fixed)
                tg.set_weight(deepcopy(self.weight_list_fixed[i // denominator]))
                
            community_list.append(tg)
                
        self.general = general
        self.community_list = community_list
        
    def filter_process(self, events, time):
    
        iv_type = self.IV_dict['iv_type']
        
        event_topic_num = events.shape[-1]
        comm_num = len(self.community_list)
        filter_num_1 = int(len(events) * sampling_ratio_list[0])
        filter_num_2 = int(filter_num_1 * sampling_ratio_list[1])
        
        if iv_type == '11':  ## IV_1, alignment
            filter_strength = IV_1(self.filter_strength, self.IV_dict, time)
        else:
            filter_strength = self.filter_strength
        
        filtered_events_list_1 = np.zeros((comm_num, filter_num_1, event_topic_num)).astype(int)
        filtered_events_list_2 = np.zeros((comm_num, filter_num_2, event_topic_num)).astype(int)
        
        if iv_type == '21':  ## IV_2, amplification
            general_frequency = IV_2(self.general.frequency, self.IV_dict, time)
        else:
            general_frequency = self.general.frequency
            
        filtered_frequency = np.array([(1-filter_strength) * general_frequency + filter_strength * self.community_list[i].frequency for i in range(comm_num)])  
        filtered_normalized_ranking = np.array([get_ranking(filtered_frequency[i])/self.topic_num for i in range(comm_num)])
        filtered_normalized_ranking_list = np.array([[[filtered_normalized_ranking[c][i] for i in event] for event in events] for c in range(comm_num)])
        
        for c in range(comm_num):
            for i in range(len(events)):
                for j in range(len(events[i])):
                    filtered_normalized_ranking_list[c][i][j] = filtered_normalized_ranking_list[c][i][j] ** power_constant_list[j]
                    
        filtered_magnitude_list = np.prod(filtered_normalized_ranking_list, axis=-1)
        filter_probability_list = filtered_magnitude_list / np.sum(filtered_magnitude_list, axis=1).reshape(-1, 1)
        
        for c in range(comm_num):
            filtered_events_list_1[c] = events[np.random.choice(len(events), size=int(len(events) * sampling_ratio_list[0]), replace=False, p=filter_probability_list[c])]
            #filtered_events_list_1[c] = events[np.argsort(filter_probability_list[c])[::-1]][:int(len(events[c]) * sampling_ratio_list[1])]
        
        for c in range(comm_num):
            weight_sum_list = [np.prod([(1-filter_strength) * self.general.weight[i, j] + filter_strength * self.community_list[c].weight[i, j] for i, j in get_pairs(event)]) for event in filtered_events_list_1[c]]
            filtered_events_list_2[c] = filtered_events_list_1[c][np.argsort(weight_sum_list)[::-1]][:int(len(filtered_events_list_1[c]) * sampling_ratio_list[1])]
        
        if iv_type == '31':  ## IV_3, reframing
            filtered_events_list = [IV_3(np.array(filtered_events_2), self.IV_dict, time) for filtered_events_2 in filtered_events_list_2]  
        else:
            filtered_events_list = [np.array(filtered_events_2) for filtered_events_2 in filtered_events_list_2]
        
        return filtered_events_list
    
if __name__ == '__main__':

    # %%
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default='simple')
    parser.add_argument("--topic_num", type=int, default=200)
    parser.add_argument("--comm_num", type=int, default=5)
    parser.add_argument("--event_num", type=int, default=1000)
    parser.add_argument("--event_topic_num", type=int, default=3)
    parser.add_argument("--filter_strength", type=float, default=0.1)
    parser.add_argument("--memory_strength", type=float, default=0.9)
    parser.add_argument("--init_type", type=str, default='perturb')
    parser.add_argument("--init_freq_std", type=float, default=0.0)
    parser.add_argument("--init_weight_std", type=float, default=0.0) #0.05
    parser.add_argument("--init_empirical_comm", type=int, default=-1)
    
    parser.add_argument("--iv_type", type=str, default='99')
    parser.add_argument("--iv_t1", type=int, default=0)
    parser.add_argument("--iv_s1", type=float, default=0.)
    parser.add_argument("--iv_t2", type=int, default=0)
    parser.add_argument("--iv_s2", type=float, default=0.)
    parser.add_argument("--iv_rank", type=int, default=-1)
    parser.add_argument("--iv_tier", type=int, default=-1)
    
    parser.add_argument("--timestep", type=int, default=50)
    parser.add_argument("--folder_name", default='test')
    
    parser.add_argument("--store_events", type=str, default='T')
    parser.add_argument("--store_extra", type=str, default='F')
    parser.add_argument("--store_weight", type=str, default='F')
    parser.add_argument("--store_corr", type=str, default='F')
    parser.add_argument("--store_tmp", type=str, default='F')
    
    parser.add_argument("--desc", default='0')
    
    args = vars(parser.parse_args(sys.argv[1:]))
    print(args)

    # get current time for recording
    now = datetime.now()

    model_type = args['model_type']
    topic_num = args['topic_num']
    comm_num = args['comm_num']
    event_num = args['event_num']
    event_topic_num = args['event_topic_num']
    filter_strength = args['filter_strength']
    memory_strength = args['memory_strength']
    init_type = args['init_type']
    init_freq_std = args['init_freq_std']
    init_weight_std = args['init_weight_std']
    init_empirical_comm = args['init_empirical_comm']
    
    iv_type = args['iv_type']
    iv_t1 = args['iv_t1']
    iv_s1 = args['iv_s1']
    iv_t2 = args['iv_t2']
    iv_s2 = args['iv_s2']
    iv_rank = args['iv_rank']
    iv_tier = args['iv_tier']
    
    timestep = args['timestep']
    folder_name = args['folder_name']
    store_events = True if args['store_events'] == 'T' else False
    store_extra = True if args['store_extra'] == 'T' else False
    store_weight = True if args['store_weight'] == 'T' else False
    store_corr = True if args['store_corr'] == 'T' else False
    store_tmp = True if args['store_tmp'] == 'T' else False
    desc = args['desc']
    
    seed = int(datetime.now().timestamp()) + int(desc)
    np.random.seed(seed)
    
    sampling_ratio_list = [0.5, 0.5]
    power_constant_list = [-0.4, -0.2, -0.1]  # twice of [-0.2, -0.1, -0.05] (empirical one), due to the correction from the 2nd filter

    init_dict = {'init_type': init_type, 
                 'init_freq_std': init_freq_std, 
                 'init_weight_std': init_weight_std, 
                 'init_empirical_comm': init_empirical_comm}
    
    IV_dict = {'iv_type': iv_type,
                'iv_t1': iv_t1,
                'iv_s1': iv_s1,
                'iv_t2': iv_t2,
                'iv_s2': iv_s2,
                'iv_rank': iv_rank,
                'iv_tier': iv_tier}
    
    system = model_system(model_type=model_type, topic_num=topic_num, comm_num=comm_num, init_dict=init_dict, filter_strength=filter_strength, memory_strength=memory_strength, sampling_ratio_list=sampling_ratio_list, power_constant_list=power_constant_list, IV_dict=IV_dict)
    
    raw_freq_list = [[] for _ in range(comm_num)]
    freq_list = [[] for _ in range(comm_num)]
    events_list = []
    filtered_events_list_list = []
    comment_num_list_list = [[] for _ in range(comm_num)]
    mult_frequency_list_list = [[] for _ in range(comm_num)]
    
    if store_weight:
        raw_adj_list = [[] for _ in range(comm_num)]
        weight_list = [[] for _ in range(comm_num)]
        
    if store_corr:
        corr_list_list = [[] for _ in range(comm_num)] # for corr / off_corr division
    
    for i in range(comm_num):
        freq_list[i].append(system.community_list[i].frequency)
        if store_weight:
            if iv_rank != -1:
                weight_list[i].append(deepcopy(system.community_list[i].weight[iv_rank]))   # only save target rank weight
            else:
                weight_list[i].append(deepcopy(system.community_list[i].weight))           # save full weight matrix

    print('setting finished')
    orig_general_frequency = deepcopy(system.general.frequency)
    orig_general_weight = deepcopy(system.general.weight)
    
    for t in tqdm(range(timestep)):
        print(t, datetime.now())
        
        if IV_dict['iv_type'][0] == '0':  # IV_0, world event
            general_frequency, general_weight = IV_0(orig_general_frequency, orig_general_weight, IV_dict, t)
            system.general.frequency = general_frequency
            system.general.set_weight(general_weight)
        
        events, _ = system.general.generate_events(event_num=event_num, event_topic_num=event_topic_num, previous_events=events_list[-1] if t>0 else None)
        
        filtered_events_list = system.filter_process(events=events, time=t)
        events_list.append(events)
        filtered_events_list_list.append(filtered_events_list)

        for i in range(comm_num):
            raw_frequency, raw_adj, comment_num_list, mult_frequency_list, corr_list = system.community_list[i].update_topic_graph(filtered_events_list[i], time=t)
            raw_freq_list[i].append(raw_frequency)
            freq_list[i].append(deepcopy(system.community_list[i].frequency))

            if store_extra:
                comment_num_list_list[i].append(comment_num_list)
                mult_frequency_list_list[i].append(mult_frequency_list)
            if store_weight:
                if iv_rank != -1:
                    weight_list[i].append(deepcopy(system.community_list[i].weight[iv_rank]))
                else:
                    weight_list[i].append(deepcopy(system.community_list[i].weight))
            if store_corr:
                corr_list_list[i].append(corr_list)
        
    # Perform garbage collection
    gc.collect()
    
    '''
    with Profile() as profile:
        Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats()
    '''

    raw_freq_list = np.array(raw_freq_list)
    freq_list = np.array(freq_list)
    
    if store_events:
        events_list = np.array(events_list)
        filtered_events_list_list = np.array(filtered_events_list_list)
    
    if store_extra:
        comment_num_list_list = np.array(comment_num_list_list)
        mult_frequency_list_list = np.array(mult_frequency_list_list)
    
    if store_weight:
        raw_adj_list = np.array(raw_adj_list)
        weight_list = np.array(weight_list)
    if store_corr:
        corr_list_list = np.array(corr_list_list)
        
    # save
    
    if desc == '':
        sim_name = f'sim_T{topic_num}_C{comm_num}_E{event_num}_ET{event_topic_num}_F{filter_strength}_M{memory_strength}_IT{init_type}_IF{init_freq_std}_IV{iv_type}_{iv_t1}_{iv_s1}_{iv_t2}_{iv_s2}_{iv_rank}_{iv_tier}_t{timestep}'
    else:
        sim_name = f'sim_T{topic_num}_C{comm_num}_E{event_num}_ET{event_topic_num}_F{filter_strength}_M{memory_strength}_IT{init_type}_IF{init_freq_std}_IV{iv_type}_{iv_t1}_{iv_s1}_{iv_t2}_{iv_s2}_{iv_rank}_{iv_tier}_t{timestep}_{desc}'

    parameter_dict = {'model_type': model_type,
                    'topic_num': topic_num,
                    'comm_num': comm_num, 
                    'event_num': event_num,
                    'event_topic_num': event_topic_num, 
                    'timestep': timestep, 
                    'weight_std': init_weight_std,
                    'freq_std': init_freq_std, 
                    'sampling_ratio_list': sampling_ratio_list,
                    'power_constant_list': power_constant_list,
                    'filter_strength': filter_strength,
                    'memory_strength': memory_strength,
                    'init_dict':init_dict,
                    'store_events':store_events,
                    'store_extra':store_extra,
                    'store_weight':store_weight,
                    'store_corr':store_corr,
                    'IV_dict': IV_dict,
                    'seed': seed
                    }

    if not store_tmp:
        with open(join('/data', 'collmind', 'model_result', folder_name, model_type, f'param_{sim_name}.pkl'), 'wb') as f:
            pickle.dump(parameter_dict, f)
    else:
        with open(join('model_result_tmp', f'param_{sim_name}.pkl'), 'wb') as f:
            pickle.dump(parameter_dict, f)
        
    result_dict = {'general_freq': system.general.frequency,
                    'general_weight': system.general.weight,
                    'raw_freq_list': raw_freq_list,
                    'freq_list': freq_list}
    
    if store_events:
        result_dict['events_list'] = events_list
        result_dict['filtered_events_list_list'] = filtered_events_list_list
    if store_extra:
        result_dict['comment_num_list_list'] = comment_num_list_list
        result_dict['mult_frequency_list_list'] = mult_frequency_list_list
    if store_weight:
        result_dict['raw_adj_list'] = raw_adj_list
        result_dict['weight_list'] = weight_list
    if store_corr:
        result_dict['corr_list_list'] = corr_list_list

    if not store_tmp:
        np.savez_compressed(join('/data', 'collmind', 'model_result', folder_name, model_type, f'result_{sim_name}'), 
                            **result_dict
                            )
    else:
        np.savez_compressed(join('model_result_tmp', f'result_{sim_name}'), 
                            **result_dict
                            )
    # get elapsed time
    elapsed_time = datetime.now() - now
    print('Time elapsed (hh:mm:ss.ms) {}'.format(elapsed_time))

    exit(0)
    