# IAML Maxone Project
# 
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
        
        print('Gen ' + str(index))
        print('\t Best: ' + str(getBest(fits_pops)))
        
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

"""
class Chromosome:
    def __init__(self, bits):
        self.bits = bits
        self.amp = np.uint8(int(self.bits[:8], 2))
        self.freq_r = np.uint8(int(self.bits[8:16], 2))
        self.freq_c = np.uint8(int(self.bits[16:], 2))
        self.fitness = self.calcFitness()

    def __repr__(self):
        return '[Bits=%s, Fitness=%f]' % (self.bits, self.fitness)

    def __str__(self):
        return '[Bits=%s, Fitness=%f]' % (self.bits, self.fitness)

    def getBits(self):
        return self.bits

    def getAmp(self):
        return self.amp

    def getFreqRow(self):
        return self.freq_r

    def getFreqCol(self):
        return self.freq_c

    def getFitness(self):
        return self.fitness

    def calcFitness(self):
        noise_pixel = self.makeNoise()

        noise_img_pixel = pure_img_pixel + noise_pixel

        diff = (noise_img_pixel.astype(float) - corrupt_img_pixel.astype(float)) ** 2
        avg_e = np.sum(diff)
        avg_e /= img_width * img_height

        return -avg_e

    # Create noise pixels
    def makeNoise(self):
        amp = mapVal(self.amp, 0, 255, 0, 30)
        freq_r = mapVal(self.freq_r, 0, 255, 0, 0.01)
        freq_c = mapVal(self.freq_c, 0, 255, 0, 0.01)

        noise_pixel = amp * np.sin( 2*np.pi * ( freq_r * row + freq_c * col ) )
        noise_pixel = np.clip(noise_pixel, 0, 255)

        return noise_pixel.astype(np.dtype('uint8'))

# Initialization
def initialize():
    return [Chromosome(format(np.random.randint(pow(2, number_of_bits)), '024b')) for i in range(population_size)]

# Selection of parents
def select(population):
    parent_1 = []
    parent_2 = []

    fitness_list = np.array([ch.getFitness() for ch in population])
    fitness_list = -1 * 1 / fitness_list
    prop = [fit / np.sum(fitness_list) for fit in fitness_list]

    for i in range(int(population_size / 2)):
        parent_1.append(np.random.choice(population, p=prop))
        parent_2.append(np.random.choice(population, p=prop))


    '''
    # Sort list by fitness value to choose chromosome with a certain fitness value
    population.sort(key=lambda x: x.getFitness())
    fitness_list = [round(ch.getFitness(), -2) for ch in population]

    # Random sampling with weight to choose chromosome with a higher fitness vlaue
    distinct_fitness = list(set(fitness_list))
    distinct_fitness.sort(key=lambda x: x)
    distinct_count = [fitness_list.count(num) for num in distinct_fitness]
    prop = [fit / sum(distinct_fitness) for fit in distinct_fitness]
    print(distinct_fitness)
    print(distinct_count)
    
    for i in range(0, (int(population_size / 2))):
        start_1 = 0
        start_2 = 0
        end_1 = 0
        end_2 = 0

        val_1 = np.random.choice(distinct_fitness, p=prop)
        val_2 = np.random.choice(distinct_fitness, p=prop)

        for j in range(0, len(distinct_fitness)):
            if distinct_fitness[j] < val_1:
                start_1 += distinct_count[j]
                end_1 += distinct_count[j]
            if distinct_fitness[j] < val_2:
                start_2 += distinct_count[j]
                end_2 += distinct_count[j]
            if distinct_fitness[j] == val_1:
                end_1 += distinct_count[j]
            if distinct_fitness[j] == val_2:
                end_2 += distinct_count[j]

        id_1 = np.random.choice(np.arange(start_1, end_1))
        id_2 = np.random.choice(np.arange(start_2, end_2))

        parent_1.append(population[id_1])
        parent_2.append(population[id_2])
    '''

    return list(zip(parent_1, parent_2))

# Crossover
def crossover(parents):
    # Swap bits
    cross_hparam = np.random.randint(number_of_bits)

    child_1 = Chromosome(parents[0].getBits()[:cross_hparam] + parents[1].getBits()[cross_hparam:])
    child_2 = Chromosome(parents[1].getBits()[:cross_hparam] + parents[0].getBits()[cross_hparam:])

    '''
    # Swap bits for each parameter
    amp_cross_hparam = np.random.randint(8)
    freq_r_cross_hparam = np.random.randint(8)
    freq_c_cross_hparam = np.random.randint(8)

    amp_1 = parents[0].getBits()[:8][:amp_cross_hparam] + parents[1].getBits()[:8][amp_cross_hparam:]
    amp_2 = parents[1].getBits()[:8][:amp_cross_hparam] + parents[0].getBits()[:8][amp_cross_hparam:]

    r_1 = parents[0].getBits()[8:16][:freq_r_cross_hparam] + parents[1].getBits()[8:16][freq_r_cross_hparam:]
    r_2 = parents[1].getBits()[8:16][:freq_r_cross_hparam] + parents[0].getBits()[8:16][freq_r_cross_hparam:]

    c_1 = parents[0].getBits()[16:][:freq_c_cross_hparam] + parents[1].getBits()[16:][freq_c_cross_hparam:]
    c_2 = parents[1].getBits()[16:][:freq_c_cross_hparam] + parents[0].getBits()[16:][freq_c_cross_hparam:]

    child_1 = Chromosome(amp_1 + r_1 + c_1)
    child_2 = Chromosome(amp_2 + r_2 + c_2)
    '''

    return [child_1, child_2]

# Mutation
def mutate(chromosome):
    amp_mut_hparam = np.random.randint(8)
    freq_r_mut_hparam = np.random.randint(8)
    freq_c_mut_hparam = np.random.randint(8)

    # Mutate amp
    amp = chromosome.getBits()[:8]
    if np.random.randint(100) < prob_mutation:
        amp = amp[:amp_mut_hparam] + '0' + amp[(amp_mut_hparam+1):] if amp[amp_mut_hparam] == '1' else amp[:amp_mut_hparam] + '1' + amp[(amp_mut_hparam+1):]

    # Mutate freq_r
    freq_r = chromosome.getBits()[8:16]
    if np.random.randint(100) < prob_mutation:
        freq_r = freq_r[:freq_r_mut_hparam] + '0' + freq_r[(freq_r_mut_hparam+1):] if freq_r[freq_r_mut_hparam] == '1' else freq_r[:freq_r_mut_hparam] + '1' + freq_r[(freq_r_mut_hparam+1):]

    # Mutate freq_c
    freq_c = chromosome.getBits()[16:]
    if np.random.randint(100) < prob_mutation:    
        freq_c = freq_c[:freq_c_mut_hparam] + '0' + freq_c[(freq_c_mut_hparam+1):] if freq_c[freq_c_mut_hparam] == '1' else freq_c[:freq_c_mut_hparam] + '1' + freq_c[(freq_c_mut_hparam+1):]

    mut_ch = Chromosome(amp + freq_r + freq_c)

    '''
    # Always mutate all three parameters
    amp_mut_hparam = np.random.randint(8)
    freq_r_mut_hparam = np.random.randint(8)
    freq_c_mut_hparam = np.random.randint(8)

    amp = chromosome.getBits()[:8]
    amp = amp[:amp_mut_hparam] + '0' + amp[(amp_mut_hparam+1):] if amp[amp_mut_hparam] == '1' else amp[:amp_mut_hparam] + '1' + amp[(amp_mut_hparam+1):]
    
    freq_r = chromosome.getBits()[8:16]
    freq_r = freq_r[:freq_r_mut_hparam] + '0' + freq_r[(freq_r_mut_hparam+1):] if freq_r[freq_r_mut_hparam] == '1' else freq_r[:freq_r_mut_hparam] + '1' + freq_r[(freq_r_mut_hparam+1):]
    
    freq_c = chromosome.getBits()[16:]
    freq_c = freq_c[:freq_c_mut_hparam] + '0' + freq_c[(freq_c_mut_hparam+1):] if freq_c[freq_c_mut_hparam] == '1' else freq_c[:freq_c_mut_hparam] + '1' + freq_c[(freq_c_mut_hparam+1):]
    
    mut_ch = Chromosome(amp + freq_r + freq_c)
    '''

    return mut_ch


def mapVal(val, from_min, from_max, to_min, to_max):
    from_scale = from_max - from_min
    to_scale = to_max - to_min

    scaled_val = (val - from_min) / from_scale

    return to_min + (scaled_val * to_scale)

def applyNoise(row, col, amp, freq_r, freq_c):
    noise = amp * np.sin( 2 * np.pi * (freq_r * row + freq_c + col))
    new_pixel = pure_img_pixel + noise

    return np.uint8(new_pixel)

def getBest(population):
    best = population[0]

    for ch in population:
        if ch.getFitness() > best.getFitness():
            best = ch

    return best

def getWorst(population):
    worst = population[0]

    for ch in population:
        if ch.getFitness() < worst.getFitness():
            worst = ch

    return worst

def saveBestImg(chromosome, index):
    noise_pixel = chromosome.makeNoise()

    img_pixel = pure_img_pixel + noise_pixel

    img = Image.fromarray(img_pixel, mode='L')
    img.save(os.path.dirname(os.path.realpath(__file__)) + '/images/lena_noise/noise_' + str(index), 'PNG')

    img = Image.fromarray(noise_pixel, mode='L')
    img.save(os.path.dirname(os.path.realpath(__file__)) + '/images/noise/noise_' + str(index), 'PNG')

def createGraph(fitness):
    x = np.arange(max_iterations)
    plt.plot(x, fitness)
    plt.show()


def main():
    # Initialize population
    pop = initialize()
    
    index = 0
    fitness_list = []

    while index < max_iterations:
        print('Gen ' + str(index) + '\n\t' + 'Best: ' + str(getBest(pop)) + '\n\tWorst: ' + str(getWorst(pop)))
        fitness_list.append(getBest(pop).getFitness())

        if index % int(max_iterations / 2) == 0:
            saveBestImg(getBest(pop), index)
        
        next_pop = []

        parents = select(pop)
        
        for pair in parents:
            children = crossover(pair) if np.random.randint(100) < prob_crossover else pair
            
            for ch in children:
                ch = mutate(ch)
                #ch = mutate(ch) if np.random.randint(100) < prob_mutation else ch
                next_pop.append(ch)

        pop = next_pop
        index = index + 1  

    saveBestImg(getBest(pop), index)
    createGraph(np.array(fitness_list))
    print('Reached max iterations')

if __name__ == '__main__':
    main()
"""