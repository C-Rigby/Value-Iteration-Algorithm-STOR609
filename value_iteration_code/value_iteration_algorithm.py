#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# value_iteration.py

# This file contains an implementation of value iteration for a Markov Decision Process (MDP)


def get_actions(P, state):
    # return list of actions for a state
    if state in P:
        return list(P[state].keys())
    else:
        return []


def is_terminal(P, state):
    # a state is terminal if it has no actions
    actions = get_actions(P, state)

    if len(actions) == 0:
        return True
    else:
        return False


def action_value(state, action, P, R, V, gamma):
    # calculate value of taking an action in a state

    total = 0

    next_states = P[state][action]

    for item in next_states:
        next_state = item[0]
        prob = item[1]

        reward = R[(state, action, next_state)]

        total = total + prob * (reward + gamma * V[next_state])

    return total


def value_iteration(S, A, P, R, gamma, epsilon, max_iterations=1000):
    # main value iteration function
    # inputs:
    # S = list of states
    # A = list of actions
    # P = transition function
    # R = reward function

    # set all values to 0 at start
    V = {}
    for s in S:
        V[s] = 0

    delta_list = []
    k = 0

    # loop until it converges or hits max iterations
    while k < max_iterations:
        k = k + 1

        delta = 0

        # copy values so we don't overwrite mid-loop
        V_new = {}
        for s in S:
            V_new[s] = V[s]

        # for each state s do
        for s in S:

            #skip terminal states
            if is_terminal(P, s):
                continue

            actions = get_actions(P, s)

            best_value = None

            for a in actions:
                val = action_value(s, a, P, R, V, gamma)

                if best_value is None:
                    best_value = val
                else:
                    if val > best_value:
                        best_value = val

            old_v = V[s]
            V_new[s] = best_value

            diff = abs(old_v - best_value)

            
            if diff > delta:
                delta = diff

        V = V_new
        delta_list.append(delta)

        # stop if small enough change
        if delta < epsilon:
            break

    # now find best policy
    pi = {}

    # for each state s do
    for s in S:

        if is_terminal(P, s):
            pi[s] = None
            continue

        actions = get_actions(P, s)

        best_action = None
        best_val = None

        for a in actions:
            val = action_value(s, a, P, R, V, gamma)

            if best_val is None:
                best_val = val
                best_action = a
            else:
                if val > best_val:
                    best_val = val
                    best_action = a

        pi[s] = best_action

    return pi, V, delta_list
