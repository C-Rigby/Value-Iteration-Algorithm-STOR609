#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#pytests

import pytest
from value_iteration_code import get_actions, is_terminal, action_value, value_iteration

def test_get_actions():
    # 1. checks that get_actions returns the correct list of actions
    # 2. if a state is not in P, it should return an empty list (terminal behavior)
    P = {"A": {"stay": [("A", 1.0)]}}

    actions_1 = get_actions(P, "A")
    actions_2 = get_actions(P, "B")

    assert actions_1 == ["stay"]
    assert actions_2 == []



def test_is_terminal():
    # a state with actions should not be terminal
    P = {"A": {"stay": [("A", 1.0)]}}

    assert is_terminal(P, "A") is False
    assert is_terminal(P, "B") is True


def test_action_value_single():
    # tests action_value with a deterministic transition
    P = {"A": {"stay": [("A", 1.0)]}}

    R = {("A", "stay", "A"): 5}

    V = {"A": 10}

    # Expected: 5 + 0.9 * 10 = 14
    val = action_value("A", "stay", P, R, V, gamma=0.9)

    assert val == pytest.approx(14.0)


def test_action_value_multiple():
    # tests action_value with probabilistic transitions
    P = {"A": {"move": [("A", 0.25), ("B", 0.75)]}}

    R = {("A", "move", "A"): 4, ("A", "move", "B"): 2}

    V = {"A": 8, "B": 6}

    # expected weighted sum of both outcomes
    val = action_value("A", "move", P, R, V, gamma=0.5)

    expected = 0.25 * (4 + 0.5 * 8) + 0.75 * (2 + 0.5 * 6)

    assert val == pytest.approx(expected)


def test_vi():
    #one state loops with constant reward
    S = ["A"]
    A = ["stay"]

    P = {"A": {"stay": [("A", 1.0)]}}

    R = {("A", "stay", "A"): 1.0}

    pi, V, delta_list = value_iteration(S, A, P, R, gamma=0.5, epsilon=1e-6)

    assert pi["A"] == "stay"
    assert V["A"] == pytest.approx(2.0)
    assert len(delta_list) >= 1


def test_vi_max_iterations():
    # forces algorithm to stop due to max_iterations rather than convergence
    S = ["A"]
    A = ["stay"]

    P = {"A": {"stay": [("A", 1.0)]}}

    R = {("A", "stay", "A"): 1}

    pi, V, delta_list = value_iteration(S, A, P, R,gamma=0.99,epsilon=1e-20,max_iterations=3)

    # should stop after exactly 3 iterations
    assert len(delta_list) == 3
    assert pi["A"] == "stay"
    assert V["A"] > 0


def test_vi_terminal_state():
    # tests that terminal states are handled correctly
    S = ["start", "end"]
    A = ["go"]

    P = {"start": {"go": [("end", 1.0)]}}

    R = {("start", "go", "end"): 5}

    pi, V, delta_list = value_iteration(S, A, P, R, gamma=0.9, epsilon=0.001)

    assert pi["start"] == "go"
    assert pi["end"] is None
    assert V["end"] == pytest.approx(0.0)


def test_vi_grid_world():
    # full test using the grid world example
    # checks both structure and correctness of policy
    S = ["TL", "TR", "BL", "BR"]
    A = ["U", "D", "L", "R"]

    P = {"TL": {"R": [("TR", 0.9), ("BL", 0.1)],"D": [("BL", 0.9), ("TR", 0.1)]},
        "TR": {"L": [("TL", 0.9), ("BR", 0.1)],"D": [("BR", 0.8), ("TL", 0.2)]},
        "BL": {"R": [("BR", 0.9), ("TL", 0.1)],"U": [("TL", 0.8), ("BR", 0.2)]}}

    R = {
        ("TL", "R", "TR"): -1,
        ("TL", "R", "BL"): -2,
        ("TL", "D", "BL"): -2,
        ("TL", "D", "TR"): -1,
        ("TR", "L", "TL"): -1.5,
        ("TR", "L", "BR"): 10,
        ("TR", "D", "BR"): 15,
        ("TR", "D", "TL"): -1,
        ("BL", "R", "BR"): 20,
        ("BL", "R", "TL"): -2.5,
        ("BL", "U", "TL"): -0.5,
        ("BL", "U", "BR"): 5}

    pi, V, delta_list = value_iteration(S, A, P, R, gamma=0.9, epsilon=0.001)

    # check valid policy outputs
    assert pi["TL"] is not None
    assert pi["TR"] is not None
    assert pi["BL"] is not None
    assert pi["BR"] is None

    # check terminal value
    assert V["BR"] == pytest.approx(0.0)

    # check known optimal policy
    assert pi["TL"] == "D"
    assert pi["TR"] == "D"
    assert pi["BL"] == "R"