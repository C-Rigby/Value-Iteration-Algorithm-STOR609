#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Grid world data

S = ["TL", "TR", "BL", "BR"]

A = ["R", "D", "L", "U"]

P = {
    "TL": {
        "R": [("TR", 0.9), ("BL", 0.1)],
        "D": [("BL", 0.9), ("TR", 0.1)]
    },
    "TR": {
        "L": [("TL", 0.9), ("BR", 0.1)],
        "D": [("BR", 0.8), ("TL", 0.2)]
    },
    "BL": {
        "R": [("BR", 0.9), ("TL", 0.1)],
        "U": [("TL", 0.8), ("BR", 0.2)]
    }
}

R = {
    ("TL", "R", "TR"): -1,
    ("TL", "R", "BL"): -2,

    ("TL", "D", "BL"): -2,
    ("TL", "D", "TR"): -1,

    ("TR", "L", "TL"): -3/2,
    ("TR", "L", "BR"): 10,

    ("TR", "D", "BR"): 15,
    ("TR", "D", "TL"): -1,

    ("BL", "R", "BR"): 20,
    ("BL", "R", "TL"): -5/2,

    ("BL", "U", "TL"): -1/2,
    ("BL", "U", "BR"): 5
}

pi, V, delta_list = value_iteration(S, A, P, R, gamma=0.9, epsilon=1e-6)

print("policy =", pi)
print("values =", V)
print("iterations =", len(delta_list))
print("last delta =", delta_list[-1])