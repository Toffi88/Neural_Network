import pygame
import time
import random
import pandas as pd
import numpy as np
import operator
from numpy import *
import matplotlib.pyplot as plt

import copy

from math import sqrt

import locale

locale.setlocale(locale.LC_ALL, 'de_DE')

pygame.init()

white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 100, 0)
blue = (50, 153, 213)

dis_width = 1500
dis_height = 780

dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('Evolution by Serge')

clock = pygame.time.Clock()

worm_block = 10
worm_speed = 10

font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)

# --- CONSTANTS ----------------------------------------------------------------+

settings = {}

# EVOLUTION SETTINGS
settings['bird_num'] = 50  # number of food particles
settings['worm_num'] = 100  # number of organism_worms
settings['food_num'] = 50  # number of food particles
settings['gens'] = 50  # number of generations
settings['elitism'] = 0.20  # elitism (selection bias)
settings['mutate'] = 0.10  # mutation rate

# SIMULATION SETTINGS
settings['gen_time'] = 100  # generation length         (seconds)
settings['dt'] = 0.04  # simulation time step      (dt)

settings['x_min'] = -2.0  # arena western border
settings['x_max'] = 2.0  # arena eastern border
settings['y_min'] = -2.0  # arena southern border
settings['y_max'] = 2.0  # arena northern border

settings['plot'] = True  # plot final generation?

# ORGANISM NEURAL NET SETTINGS
settings['inodes'] = 1  # number of input nodes
settings['hnodes'] = 5  # number of hidden nodes
settings['onodes'] = 2  # number of output nodes


# --- Neural Network ---------------+

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Sigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        assert isinstance(probabilities, object)
        self.output = probabilities


# --- Distance ---------------+

def dist(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# --- BORDER FOR SNAKE ---------------+
def distance(own, target_list):
    closest_target = []
    for target in target_list:
        closest_food_dist = dist(own.x, own.y, target.x, target.y)
        closest_target.append([closest_food_dist, target.x, target.y])

    food_org_dist, target_pos_x, target_pos_y = min(closest_target, key=lambda item: item[0])

    return target_pos_x, target_pos_y


# --- Organismen ---------------+

class organism_worm():
    def __init__(self, xx, yy, dense1, dense2, dense3):

        self.energy = 500
        self.life_points = 0

        self.x = random.uniform(30, dis_width - 30)
        self.y = random.uniform(30, dis_height - 30)

        if xx > 0:
            self.x = random.uniform(30, dis_width - 30)
        if yy > 0:
            self.y = random.uniform(30, dis_height - 30)

        # --- CREATE NEW BRAIN LAYER
        input_count = 8
        output_count = 4
        dense2_count = 5
        dense3_count = 5

        if dense1 == 0:
            self.dense1 = Layer_Dense(input_count, dense2_count)
            self.dense2 = Layer_Dense(dense2_count, dense3_count)
            self.dense3 = Layer_Dense(dense3_count, output_count)

        # --- CREATE NEW BRAIN LAYER
        else:
            self.dense1 = copy.deepcopy(dense1)
            self.dense2 = copy.deepcopy(dense2)
            self.dense3 = copy.deepcopy(dense3)

            dense1.weights += 0.05 * np.random.randn(input_count, 5)
            dense1.bias += 0.05 * np.random.randn(1, 5)
            dense2.weights += 0.05 * np.random.randn(5, 5)
            dense2.bias += 0.05 * np.random.randn(1, 5)
            dense3.weights += 0.05 * np.random.randn(5, output_count)
            dense3.bias += 0.05 * np.random.randn(1, output_count)

    def energy_update(self, worm_list):
        self.energy += -1
        if self.energy <= 0:
            worm_list.remove(self)

        # elif self.x < 0 or self.x > dis_width:
        #     worm_list.remove(self)
        #
        # elif self.y < 0 or self.y > dis_height:
        #     worm_list.remove(self)

    def Brain(self, X):

        Activation_Funktion = Activation_Sigmoid()

        activation1 = Activation_ReLU()
        activation2 = Activation_ReLU()
        activation3 = Activation_Funktion

        self.dense1.forward(X)
        activation1.forward(self.dense1.output)

        self.dense2.forward(activation1.output)
        activation2.forward(self.dense2.output)

        self.dense3.forward(activation2.output)
        activation3.forward(self.dense3.output)

        self.output_NN = activation3.output

    def position_update_worm(self, food_list, bird_list):
        self_list = self

        # --- Find next Food
        target_pos_x, target_pos_y = distance(self_list, food_list)

        target_pos_x = target_pos_x - self.x
        target_pos_y = target_pos_y - self.y

        target_split_pos = split_input(target_pos_x, target_pos_y)

        # --- Find next Enemy
        enemy_pos_x, enemy_pos_y = distance(self_list, food_list)

        enemy_pos_x = enemy_pos_x - self.x
        enemy_pos_y = enemy_pos_y - self.y

        enemy_split_pos = split_input(enemy_pos_x, enemy_pos_y)

        X = target_split_pos + enemy_split_pos
        print(X)

        self.Brain(X)

        self.x += (self.output_NN[0][0] - 0.5)
        self.y += (self.output_NN[0][1] - 0.5)

        border(self)


class organism_bird():
    def __init__(self, xx, yy, dense1, dense2, dense3):

        self.energy = 500
        self.life_points = 0

        self.x = random.uniform(30, dis_width - 30)
        self.y = random.uniform(30, dis_height - 30)

        if xx > 0:
            self.x = random.uniform(30, dis_width - 30)
        if yy > 0:
            self.y = random.uniform(30, dis_height - 30)

        # --- CREATE NEW BRAIN LAYER
        input_count = 4
        output_count = 4
        dense2_count = 5
        dense3_count = 5

        if dense1 == 0:
            self.dense1 = Layer_Dense(input_count, dense2_count)
            self.dense2 = Layer_Dense(dense2_count, dense3_count)
            self.dense3 = Layer_Dense(dense3_count, output_count)

        # --- CREATE NEW BRAIN LAYER
        else:
            self.dense1 = copy.deepcopy(dense1)
            self.dense2 = copy.deepcopy(dense2)
            self.dense3 = copy.deepcopy(dense3)

            dense1.weights += 0.05 * np.random.randn(input_count, 5)
            dense1.bias += 0.05 * np.random.randn(1, 5)
            dense2.weights += 0.05 * np.random.randn(5, 5)
            dense2.bias += 0.05 * np.random.randn(1, 5)
            dense3.weights += 0.05 * np.random.randn(5, output_count)
            dense3.bias += 0.05 * np.random.randn(1, output_count)

    def energy_update(self, bird_list):
        self.energy += -1
        if self.energy <= 0:
            bird_list.remove(self)

    def Brain(self, X):

        activation1 = Activation_ReLU()
        activation2 = Activation_ReLU()
        activation3 = Activation_Sigmoid()

        self.dense1.forward(X)
        activation1.forward(self.dense1.output)

        self.dense2.forward(activation1.output)
        activation2.forward(self.dense2.output)

        self.dense3.forward(activation2.output)
        activation3.forward(self.dense3.output)

        self.output_NN = activation3.output

    def position_update_bird(self, worm_list):
        self_list = self

        # --- Find next Food
        target_pos_x, target_pos_y = distance(self_list, worm_list)

        target_pos_x = target_pos_x - self.x
        target_pos_y = target_pos_y - self.y

        target_split_pos = split_input(target_pos_x, target_pos_y)

        X = target_split_pos
        # print("X: ", X)

        self.Brain(X)
        # print("NN: ", self.output_NN)

        self.x += (self.output_NN[0][0] - 0.5)
        self.y += (self.output_NN[0][1] - 0.5)

        border(self)


# --- BORDER FOR SNAKE ---------------+
def border(own):
    if own.x < 10:
        own.x = 10
        own.energy -= 10
    if own.y < 10:
        own.y = 10
        own.energy -= 10
    if own.x > dis_width - 10:
        own.x = dis_width - 10
        own.energy -= 10
    if own.y > dis_height - 10:
        own.y = dis_height - 10
        own.energy -= 10


# --- FOOD ---------------+
class food():
    def __init__(self, xx, yy):
        self.x = random.uniform(30, dis_width - 30)
        self.y = random.uniform(30, dis_height - 30)

        if xx > 0:
            self.x = xx
        if yy > 0:
            self.y = yy

    # --- SNAKE EATING ---------------+


def eating(own_list, target_list, org):
    for own in own_list:
        for target in target_list:
            food_org_dist = dist(own.x, own.y, target.x, target.y)
            if food_org_dist <= 2:
                own.energy += 500
                target_list.remove(target)
                if org == "worm":
                    new_org = organism_worm(own.x + random.uniform(-3, 3), own.y + random.uniform(-3, 3), own.dense1,
                                            own.dense2, own.dense3)
                    own_list.append(new_org)

                    if len(own_list) < 10:
                        new_org = organism_worm(own.x + random.uniform(-3, 3), own.y + random.uniform(-3, 3),
                                                own.dense1, own.dense2, own.dense3)
                        own_list.append(new_org)
                    break
                if org == "bird":
                    new_org = organism_bird(own.x + random.uniform(-3, 3), own.y + random.uniform(-3, 3), own.dense1,
                                            own.dense2, own.dense3)
                    own_list.append(new_org)

                    if len(own_list) < 10:
                        new_org = organism_bird(own.x + random.uniform(-3, 3), own.y + random.uniform(-3, 3),
                                                own.dense1,
                                                own.dense2, own.dense3)
                        own_list.append(new_org)
                    break


# --- Split input x & y in positiv or negativ value
def split_input(x, y):
    tar_x_pos = np.maximum(0, x)  # positv x or 0
    tar_x_neg = np.minimum(0, x)  # negativ x or 0
    tar_y_pos = np.maximum(0, y)  # positv y or 0
    tar_y_neg = np.minimum(0, y)  # negativ y or 0
    return [tar_x_pos, tar_x_neg, tar_y_pos, tar_y_neg]


def respawn_food(worm_list, food_list):
    pos_x = random.uniform(30, dis_width - 30)
    pos_y = random.uniform(30, dis_height - 30)

    entfernungen = []
    for worm in worm_list:
        i = dist(worm.x, worm.y, pos_x, pos_y)
        entfernungen.append(i)
    if min(entfernungen) >= 10:
        i = food(pos_x, pos_y)
        food_list.append(i)
    else:
        respawn_food(worm_list, food_list)


def best(own, best, org):
    own.life_points += 1
    if org == "worm":
        i = [own, own.life_points]
        best.append(i)


def gameLoop(settings):
    pygame.init()
    game_over = False
    game_close = False
    running = True

    # --- CREATIOM WORM ---------------+
    worm_list = []
    for i in range(settings['worm_num']):
        i = organism_worm(0, 0, 0, 0, 0)
        worm_list.append(i)

    # --- CREATIOM BIRD ---------------+
    bird_list = []
    for i in range(settings['bird_num']):
        i = organism_bird(0, 0, 0, 0, 0)
        bird_list.append(i)

    # --- START FOOD ---------------+
    food_list = []
    for i in range(settings['food_num']):
        i = food(0, 0)
        food_list.append(i)

    # --- GAME---------------+
    game_tick = 0

    while not game_over:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        dis.fill(black)

        bird_count = len(bird_list)
        worm_count = len(worm_list)
        food_count = len(food_list)
        game_tick = game_tick + 1

        # --- FOOD RESPAWN ---------------+
        if game_tick % 4 == 0:
            respawn_food(worm_list, food_list)

        # --- DATA Visual ---------------+
        display_text = "Gameround: " + str(game_tick) + " | Bird Count: " + str(bird_count) + " | Worm Count: " + str(
            worm_count) + " | Food Count: " + str(
            food_count)
        pygame.display.set_caption(display_text)

        # -- WORM Visual ---------------+
        best_worm = []
        for worm in worm_list:
            worm.position_update_worm(food_list, bird_list)
            worm.energy_update(worm_list)
            best(worm, best_worm, "worm")

            worm_colour = (0, 0, 255)

            pygame.draw.rect(dis, worm_colour, [worm.x, worm.y, 8, 8])

        best_worm_colour = (255, 215, 0)
        best_worm_obj, best_worm_points = max(best_worm, key=lambda item: item[1])
        print(max(best_worm, key=lambda item: item[1]))
        pygame.draw.rect(dis, best_worm_colour, [best_worm_obj.x, best_worm_obj.y, 8, 8])

        # --- BIRD Visual ---------------+
        best_bird = []
        for bird in bird_list:
            bird.position_update_bird(worm_list)
            bird.energy_update(bird_list)
            best(bird, best_bird, "bird")

            rect1 = pygame.Rect(bird.x, bird.y, 8, 8)

            bird_colour = (255, 0, 0)

            pygame.draw.rect(dis, bird_colour, [bird.x, bird.y, 8, 8])

        # --- FOOD Visual ---------------+
        for essen in food_list:
            pygame.draw.rect(dis, green, [essen.x, essen.y, 8, 8])

        # --- LAST INSTRUCTIONS ---------------+
        eating(worm_list, food_list, "worm")
        eating(bird_list, worm_list, "bird")

        pygame.display.update()

        clock.tick(50)

        # --- LAST INSTRUCTIONS ---------------+

        plt.plot([-1, -4.5, 16, 23, 15, 59])
        plt.show()
        # print("------------")


gameLoop(settings)
