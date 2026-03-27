#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Example 9.27 data

from value_iteration_code import value_iteratio

S = ["healthy", "sick"]

A = ["relax", "party"]

P = {
    "healthy": {
        "relax": [("healthy", 0.95), ("sick", 0.05)],
        "party": [("healthy", 0.7), ("sick", 0.3)]
    },
    "sick": {
        "relax": [("healthy", 0.5), ("sick", 0.5)],
        "party": [("healthy", 0.1), ("sick", 0.9)]
    }
}

R = {
    ("healthy", "relax", "healthy"): 7,
    ("healthy", "relax", "sick"): 7,

    ("healthy", "party", "healthy"): 10,
    ("healthy", "party", "sick"): 10,

    ("sick", "relax", "healthy"): 0,
    ("sick", "relax", "sick"): 0,

    ("sick", "party", "healthy"): 2,
    ("sick", "party", "sick"): 2
}

pi, V, delta_list = value_iteration(S, A, P, R, gamma=0.9, epsilon=0.001)

print("policy =", pi)
print("values =", V)
print("iterations =", len(delta_list))
print("last delta =", delta_list[-1])