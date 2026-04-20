
# value_iteration

## Description

This is a git repository which implements the value iteration algorithm for solving finite Markov Decision Processes (MDPs). The algorithm computes the optimal value function and derives the corresponding optimal policy using iterative updates based on expected future rewards.

# Parameters and Outputs

## Function

```
value_iteration(S, A, P, R, gamma, epsilon, max_iterations)
```

## Parameters
- S: list of states
- A: list of actions
- P[s][a]: transition function
	- list of (next_state, probability) pairs
- R[(s, a, s')]: reward function
- gamma: discount factor, between 0 and 1
- epsilon: convergence threshold
- max_iterations: maximum number of iterations

## Outputs
- pi: optimal policy which covers the optimal action which should be taken for each state 
- V: value function 
- delta_list: convergence history (the value changes per iteration)

# Installation

Install from github
```
pip install git+https://github.com/C-Rigby/value_iteration.git
```

# Example Usage

```
from value_iteration_algorithm import value_iteration

S = ["A"]
A = ["stay"]

P = {
    "A": {
        "stay": [("A", 1.0)]
    }
}

R = {
    ("A", "stay", "A"): 1.0
}

pi, V, delta_list = value_iteration(S, A, P, R, gamma=0.5, epsilon=0.001)

print(pi)  # {'A': 'stay'}
print(V)   # {'A': ~2.0}
```
Further examples can be found in the "examples" folder.

# Testing

Run tests using:
```
pytest -v
```
This checks:

- all relevant functions
- value updates
- convergence behaviour
- policy correctness on example MDPs

# References

The implementation is based on the value iteration section from Artificial Intelligence: Foundations of Computational Agents (2nd Edition), Section 9.5.2, Figure 9.16, url = https://artint.info/2e/html2e/ArtInt2e.Ch9.S5.SS2.html#Ch9.F16.



New line 1
New line 2
New line 3
New line 4
New line 5
New line 6
# Collaborative
