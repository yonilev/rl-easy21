import random
import numpy as np
from copy import deepcopy


HIT = 0
STICK = 1


def draw_card(random_color=True):
    card = random.randint(1,10)
    if random_color:
        if random.random()>2/3:
            card *= -1
    return card


def is_bust(player):
    return player < 1 or player > 21


class State:
    def __init__(self):
        self.player = draw_card(False)
        self.dealer = draw_card(False)
        self.is_terminal = False


def step(s,a):
    next_s = deepcopy(s)

    if a==HIT:
        next_s.player += draw_card()
        if is_bust(next_s.player):
            next_s.is_terminal = True
            return -1,next_s
        return 0,next_s

    while not is_bust(next_s.dealer) and next_s.dealer < 17:
        next_s.dealer += draw_card()

    next_s.is_terminal = True
    if is_bust(next_s.dealer) or next_s.player > next_s.dealer:
        return 1,next_s
    if next_s.player < next_s.dealer:
        return -1,next_s
    return 0,next_s


if __name__ == "__main__":
    pass












