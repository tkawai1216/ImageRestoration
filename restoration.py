# IAML Image Restoration Project
# 
# Team #11
# Members:
#   Toshiki Kawai
#   Edgar Handy
# 
# REF: https://gist.github.com/bellbind/741853
# REF: http://www.obitko.com/tutorials/genetic-algorithms/ga-basic-description.php

import random
import numpy as np
import cv2
from PIL import Image
import time
import os
import matplotlib.pyplot as plt
import sys

population_size = 100
number_of_bits = 24
max_iterations = 100
prob_crossover = 90
prob_mutation = 10
pure_img = Image.open('lena.png')
pure_img_pixel = np.array(pure_img.convert('L'))
corrupt_img = Image.open('lena_noisy.png')
corrupt_img_pixel = np.array(corrupt_img.convert('L'))
img_width, img_height = pure_img.size
row, col = np.meshgrid(np.arange(img_width), np.arange(img_height))

amp_cross_hparam = 0
freq_r_cross_hparam = 0
freq_c_cross_hparam = 0
amp_mut_hparam = 0
freq_r_mut_hparam = 0
freq_c_mut_hparam = 0

def main(argv):
    global population_size, prob_crossover, prob_mutation, max_iterations
    try:
        population_size = int(argv[1])
        prob_crossover = int(argv[2])
        prob_mutation = int(argv[3])
        max_iterations = int(argv[4])
    except TypeError:
        print('Wrong command: python restoration.py <population_size> <prob_crossover> <prob_mutation> <max_iterations>')
        quit()
    except ValueError:
        print('Wrong command: python restoration.py <population_size> <prob_crossover> <prob_mutation> <max_iterations>')
        quit()
    except IndexError:
        print('Wrong command: python restoration.py <population_size> <prob_crossover> <prob_mutation> <max_iterations>')
        quit()

    index = 0
    population = initial_population(population_size)
    best_fitness = []

    while index < max_iterations:
        fits_pops = [(fitness(ch), ch) for ch in population]
        
        best = getBest(fits_pops)
        print('Gen ' + str(index))
        print('\t Best: ' + str(best), end='')
        print('\t Params: ' + str(getParams(best)))
        
        best_fitness.append(getBest(fits_pops)[0])

        if index % int(max_iterations / 2) == 0:
            saveBestImg(getBest(fits_pops)[1], index)
        
        population = breed_population(fits_pops)
        index += 1
    
    saveBestImg(getBest(fits_pops)[1], index)
    createGraph(np.array(best_fitness))

    return population

def breed_population(fitness_population):
    parent_pairs = select_parents(fitness_population)
    size = len(parent_pairs)
    next_population = []
    for k in range(size) :
        parents = parent_pairs[k]
        cross = random.randint(0, 100) < prob_crossover
        children = crossover(parents) if cross else parents
        for ch in children:
            next_population.append(mutate(ch) if mutate else ch)
    return next_population


#Initialize population
def initial_population(population_size):
    return [format(np.random.randint(pow(2, number_of_bits)), '024b') for i in range(population_size)]
    
#Calculate total distance
def fitness(chromosome):
    noise_pixel = makeNoise(chromosome)

    noise_img_pixel = pure_img_pixel.astype(float) + noise_pixel

    #true_noise_pixel = corrupt_img_pixel.astype(float) - pure_img_pixel.astype(float)
    #print(true_noise_pixel)

    diff = (noise_img_pixel- corrupt_img_pixel.astype(float)) ** 2
    avg_e = np.sum(diff)
    avg_e /= img_width * img_height

    return -avg_e

def makeNoise(chromosome):
    amp = mapVal(int(chromosome[:8], 2), 0, 255, 0, 30)
    freq_r = mapVal(int(chromosome[8:16], 2), 0, 255, 0, 0.01)
    freq_c = mapVal(int(chromosome[16:], 2), 0, 255, 0, 0.01)

    noise_pixel = amp * np.sin( 2*np.pi * ( freq_r * row + freq_c * col ) )
    
    return noise_pixel

def mapVal(val, from_min, from_max, to_min, to_max):
    from_scale = from_max - from_min
    to_scale = to_max - to_min

    scaled_val = (val - from_min) / from_scale

    return to_min + (scaled_val * to_scale)

def getBest(fitness_population):
    best = max(fitness_population, key=lambda x: x[0])
    return best

def getParams(chromosome):
    params = [int(chromosome[1][:8], 2), int(chromosome[1][8:16], 2), int(chromosome[1][16:], 2)]
    return params

def saveBestImg(chromosome, index):
    noise_pixel = makeNoise(chromosome)

    n_pixel = np.clip(noise_pixel, 0, 255)
    img = Image.fromarray(n_pixel, mode='L')
    img.save(os.path.dirname(os.path.realpath(__file__)) + '/images/noise/noise_' + 'p_' + str(population_size) + 'c_' + str(prob_crossover) + 'm_' + str(prob_mutation) + 'i_' + str(max_iterations) + 'N_' + str(index), 'PNG')

    noise_img_pixel = pure_img_pixel.astype(float) + noise_pixel
    n_i_pixel = np.clip(noise_img_pixel, 0, 255)
    img = Image.fromarray(n_i_pixel.astype('uint8'), mode='L')
    img.save(os.path.dirname(os.path.realpath(__file__)) + '/images/lena_noise/lena_' + 'p_' + str(population_size) + 'c_' + str(prob_crossover) + 'm_' + str(prob_mutation) + 'i_' + str(max_iterations) + 'N_' + str(index), 'PNG')

    noise_diff_pixel = np.absolute( corrupt_img_pixel.astype(float) - noise_img_pixel )
    noise_diff_pixel = np.clip(noise_diff_pixel, 0, 255)
    img = Image.fromarray(noise_diff_pixel.astype('uint8'), mode='L')
    img.save(os.path.dirname(os.path.realpath(__file__)) + '/images/noise_diff/noise_diff_' + 'p_' + str(population_size) + 'c_' + str(prob_crossover) + 'm_' + str(prob_mutation) + 'i_' + str(max_iterations) + 'N_' + str(index), 'PNG')

    restore_pixel = corrupt_img_pixel.astype(float) - noise_pixel
    r_pixel = np.clip(restore_pixel, 0, 255)
    img = Image.fromarray(r_pixel.astype('uint8'))
    img.save(os.path.dirname(os.path.realpath(__file__)) + '/images/lena_restored/lena_restored_' + 'p_' + str(population_size) + 'c_' + str(prob_crossover) + 'm_' + str(prob_mutation) + 'i_' + str(max_iterations) + 'N_' + str(index), 'PNG')

    restore_diff_pixel = np.absolute( restore_pixel - pure_img_pixel.astype('float') )
    r_d_pixel = np.clip(restore_diff_pixel, 0, 255)
    img = Image.fromarray(r_d_pixel.astype('uint8'))
    img.save(os.path.dirname(os.path.realpath(__file__)) + '/images/lena_restored_diff/lena_restored_diff' + 'p_' + str(population_size) + 'c_' + str(prob_crossover) + 'm_' + str(prob_mutation) + 'i_' + str(max_iterations) + 'N_' + str(index), 'PNG')


def createGraph(fitness):
    x = np.arange(max_iterations)
    plt.plot(x, fitness)
    plt.show()

#Selection
def select_parents(fitness_population):
    # fitness_population = [ (fitness, bits), (fitness, bits), ...]
    # Tournament
    parent_1 = []
    parent_2 = []

    fitness_list = np.array([ch[0] for ch in fitness_population])
    population_list = [ch[1] for ch in fitness_population]
    fitness_list = -1 * 1 / fitness_list
    prop = [fit / np.sum(fitness_list) for fit in fitness_list]
    
    for i in range(int(population_size / 2)):
        index_1 = np.random.choice(np.arange(population_size), p=prop)
        index_2 = np.random.choice(np.arange(population_size), p=prop)
        
        parent_1.append(population_list[index_1]) if fitness_list[index_1] > fitness_list[index_2] else parent_1.append(population_list[index_2])

        index_1 = np.random.choice(np.arange(population_size), p=prop)
        index_2 = np.random.choice(np.arange(population_size), p=prop)
        
        parent_2.append(population_list[index_1]) if fitness_list[index_1] > fitness_list[index_2] else parent_2.append(population_list[index_2])

    return list(zip(parent_1, parent_2)) 

    '''
    # Select from best and worst
    fits_pops = sorted(fitness_population, key=lambda x: x[0])

    # 10 pairs of best and stupid, 40 pairs of bests
    best = [ch[1] for ch in fits_pops[-(int(population_size / 3)):]]
    worst = [ch[1] for ch in fits_pops[:-int(population_size / 3)]]

    parent_1 = np.random.choice(best, int(population_size / 2))
    parent_2_b = np.random.choice(best, int(population_size / 3))
    parent_2_w = np.random.choice(worst, int(population_size / 2) - int(population_size / 3))
    parent_2 = np.concatenate((parent_2_b, parent_2_w))
    
    #for i in range(len(parent_1)):
    #    print(parent_1[i], parent_2[i])

    return list(zip(parent_1, parent_2)) 
    '''

#Crossover
def crossover(parents):
    # Swap bits
    cross_hparam = np.random.randint(number_of_bits)

    child_1 = parents[0][:cross_hparam] + parents[1][cross_hparam:]
    child_2 = parents[1][:cross_hparam] + parents[0][cross_hparam:]

    return [child_1, child_2]

#Mutation
def mutate(chromosome):
    amp_mut_hparam = np.random.randint(8)
    freq_r_mut_hparam = np.random.randint(8)
    freq_c_mut_hparam = np.random.randint(8)

    # Mutate amp
    amp = chromosome[:8]
    if np.random.randint(100) < prob_mutation:
        amp = amp[:amp_mut_hparam] + '0' + amp[(amp_mut_hparam+1):] if amp[amp_mut_hparam] == '1' else amp[:amp_mut_hparam] + '1' + amp[(amp_mut_hparam+1):]

    # Mutate freq_r
    freq_r = chromosome[8:16]
    if np.random.randint(100) < prob_mutation:
        freq_r = freq_r[:freq_r_mut_hparam] + '0' + freq_r[(freq_r_mut_hparam+1):] if freq_r[freq_r_mut_hparam] == '1' else freq_r[:freq_r_mut_hparam] + '1' + freq_r[(freq_r_mut_hparam+1):]

    # Mutate freq_c
    freq_c = chromosome[16:]
    if np.random.randint(100) < prob_mutation:    
        freq_c = freq_c[:freq_c_mut_hparam] + '0' + freq_c[(freq_c_mut_hparam+1):] if freq_c[freq_c_mut_hparam] == '1' else freq_c[:freq_c_mut_hparam] + '1' + freq_c[(freq_c_mut_hparam+1):]

    mut_ch = amp + freq_r + freq_c

    return mut_ch

if __name__ == "__main__":
    main(sys.argv)
