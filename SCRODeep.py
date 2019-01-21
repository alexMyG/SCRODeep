import numpy as np
import json
import urllib
import os.path
import os
import time

from Individual import *
from random import shuffle
from copy import deepcopy
from scipy.io import loadmat
from keras import backend as K
from termcolor import colored
from KerasExecutor import KerasExecutor
from OperatorsDeep import complete_crossover, complete_mutation

K.set_image_dim_ordering('th')

METRICS = ["accuracy"]
EARLY_STOPPING_PATIENCE_KERAS = 10
EARLY_STOPPING_PATIENCE_SCRO = 4  # if accuracy_validation does not improve in the last 4 generations, the search stops
LOSS = "categorical_crossentropy"
MAX_GENERATIONS_SCRO = 30
TEST_SIZE = 0.5




"""
IMPORTANT NOTE

REEF IS A LIST WITH INDIVIDUALS AND NONE POSITIONS.
POPULATION IS A SET OF INDIVIDUALS WHERE THE ORDER IN THE LIST DOES NOT MATTER AND WHICH CONTAIN NONE VALUES

"""


class Configuration(object):
    def __init__(self, j):
        self.__dict__ = json.load(j)


def runSCRO():
    """
    RUNS THE SCRO ALGORITHM FOR DEEP LEARNING ARCHITECTURES OPTIMISATION
    :return:
    """

    # Loading MNIST dataset
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    mnist_path = "./mnist-original.mat"

    if not os.path.isfile(mnist_path):
        response = urllib.urlopen(mnist_alternative_url)
        with open(mnist_path, "wb") as f:
            print "Downloading data"
            content = response.read()
            f.write(content)
    mnist_raw = loadmat(mnist_path)
    dataset = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }

    print "Starting keras executor"
    ke = KerasExecutor(dataset, TEST_SIZE, METRICS, EARLY_STOPPING_PATIENCE_KERAS, LOSS)

    config_data = open('parametersGenetic.json')
    configuration = Configuration(config_data)

    ##############################
    # Initialisation
    ##############################

    reef = initialisation(Rsize=4, config=configuration, n_global_in=deepcopy(ke.n_in), n_global_out=ke.n_out, ke=ke)
    # Population is already evaluated in the initialisation function

    history = []

    max_fitness_ever = 0.0
    generations_with_no_improvement = 0

    output_file = open("EXECUTION_" + str(time.time()) + ".csv", "w")
    output_file.write("fitness_mean, fitness_std, fitness_max, fitness_min, count_evaluations, individuals_depredated, ratio_reef, time_generation, reef\n")

    ##############################
    # Loop
    ##############################
    for i in range(MAX_GENERATIONS_SCRO):

        start_time = time.time()

        print colored("GENERATION: " + str(i), "red")
        pool = []

        if len(filter(lambda w: w is not None, reef)) == 0:
            output_file.write("ALL REEF IS NONE!")
            print colored("ALL REEF IS NONE!, BREAKING EVOLUTION!", "red")
            break

        # 1 Asexual reproduction
        asexual_new_individual = asexual_reproduction(reef, configuration)
        if asexual_new_individual is not None:
            pool = pool + [asexual_new_individual]

        # 2 Sexual reproduction
        sexual_new_individuals = sexual_reproduction(reef, configuration)
        pool = pool + sexual_new_individuals

        # 3 Larvae settlement
        pool, count_evaluations = eval_population(pool, ke)
        reef, settled = larvae_settlement(reef, pool)

        # 4  Depredation
        # Todo remove returns same object
        reef, individuals_depredated = depredation(reef)

        # History

        fitness_mean, fitness_std, fitness_max, fitness_min = fitness_mean_std(reef)

        if fitness_max > max_fitness_ever:
            max_fitness_ever = fitness_max
            generations_with_no_improvement = 0
        else:
            generations_with_no_improvement += 1

        finish_time = time.time()

        time_generation = finish_time-start_time

        positions_free = len(filter(lambda w: w is not None, reef))
        positions_total = len(reef)


        history.append([fitness_mean, fitness_std, fitness_max, fitness_min, count_evaluations, individuals_depredated, str(positions_free) + "/" + str(positions_total), time_generation, deepcopy(reef)])

        output_file.write(str(fitness_mean) + "," +
                          str(fitness_std) + "," +
                          str(fitness_max) + "," +
                          str(fitness_min) + "," +
                          str(count_evaluations) + "," +
                          str(individuals_depredated) + "," +
                          str(positions_free) + "/" + str(positions_total) + ","
                          str(time_generation) + "," +
                          str(reef) + "\n")

        print colored(
            str(fitness_mean) + "," + str(fitness_std) + "," + str(fitness_max) + "," + str(fitness_min) + "," + str(
                count_evaluations) + "," + str(individuals_depredated) + "," +
                str(positions_free) + "/" + str(positions_total) + "," + str(time_generation), 'yellow')

        if generations_with_no_improvement >= MAX_GENERATIONS_SCRO:
            print colored("Stop criterion reached! " + str(generations_with_no_improvement) + "generations with no "
                                                                                              "improvement!", "red")
            break

    output_file.close()
#################################################
#################################################
#                   SCRO                        #
#################################################
#################################################



#  OPERATORS

# TODO: check: reef is a list.
def initialisation(Rsize, config, n_global_in, n_global_out, ke):
    """
    Initialisation function. It creates the first population with a reef or Rsize*Rsize.
    At first all positions are filled with random individuals.
    A set of random corals are deleted according to the formula fi not in (f_1 - sf1, 1]
    :param Rsize:
    :param rate_free_corals:
    :param config:
    :param n_global_in:
    :param n_global_out:
    :param ke:
    :return:
    """
    # Creating population of Rsize*Rsize new random individuals
    # population = [[Individual(config, n_global_in, n_global_out)]*Rsize for _ in range(Rsize)]
    reef = [Individual(config, n_global_in, n_global_out) for _ in range(Rsize*Rsize)]
    print "Reef created with " + str(len(reef)) + " solutions"
    print "Original size: " + str(len(reef))

    # Eval population

    reef, count_evaluations = eval_population(reef, ke)
    #for ind in reef:
    #    print str(ind.fitness)

    # Calculating fitness mean and std deviation
    fitness_mean, fitness_std, fitness_max, fitness_min = fitness_mean_std(reef)

    # Deleting corals according to formula
    # It is not the same that the depredation one
    # new_population = [[ind if initial_deletion_check(ind.fitness, fitness_mean, fitness_std) else None for ind in line ] for line in population]
    new_reef = [
        ind if initial_deletion_check(ind.fitness["accuracy_validation"], fitness_mean, fitness_std) else None for
        ind in reef]

    print "Population reduced to: " + str(len(filter(lambda w: w is not None, new_reef))) + " solutions"

    #for ind in filter(lambda w: w is not None, new_reef):
    #    print str(ind.fitness)

    return new_reef

def initial_deletion_check(fitness, fitness_mean, fitness_std):

    return (fitness_mean - fitness_std) < fitness <= 1


def fitness_mean_std(reef):

    # fitnesses_reef = np.array([[eval_keras(x, ke) if x is not None else None for x in line ] for line in population])


    fitness_mean = 0.0
    fitness_std = 0.0
    fitness_max = 0.0
    fitness_min = 0.0

    if len(filter(lambda w: w is not None, reef)) > 0:

        fitnesses_reef = np.array([ind.fitness["accuracy_validation"] for ind in reef if ind is not None])  # None are removed

        fitness_mean = fitnesses_reef.mean()
        fitness_std = fitnesses_reef.std()
        fitness_max = fitnesses_reef.max()
        fitness_min = fitnesses_reef.min()

    return fitness_mean, fitness_std, fitness_max, fitness_min

# Asexual reproduction

def asexual_reproduction(reef, config):

    selected_individual = asexual_selection(reef)
    new_individual = None
    if selected_individual is not None:
        new_individual = mutation(selected_individual, config)
        new_individual.fitness = None
    return new_individual

# Asexual selection
def asexual_selection(reef):
    """

    :param population: set of chromosomes
    :return: aLarvae: selected larvae
    """

    population = filter(lambda w: w is not None, reef)

    fitness_mean, fitness_std, fitness_max, fitness_min = fitness_mean_std(population)
    range_min = (fitness_mean + fitness_std)
    range_max = 1
    fragmentation = filter(lambda ind: range_min < ind.fitness["accuracy_validation"] <= range_max, population)

    if len(fragmentation) > 0:

        # sorted_population = sorted(population, key=lambda coral: coral.fitness if coral is not None else -1, reverse=True)
        # max_value = round(fa * len(filter(lambda x: x is not None, population)))
        idx = random.randrange(0, len(fragmentation))
        # TODO: What if there is nothing but holes in the population? -> Is it that possible?

        aLarvae = deepcopy(fragmentation[idx])

    else:
        aLarvae = None

    return aLarvae

# Sexual reproduction
def sexual_reproduction(reef, config):
    # population.sort(key=lambda x: x.fitness, reverse=True)

    new_population = []

    # A random fraction Fb of the individuals is selected uniformly

    not_none_population = filter(lambda w: w is not None, reef)

    fitness_mean, fitness_std, fitness_max, fitness_min = fitness_mean_std(not_none_population)

    range_min = (fitness_mean - fitness_std)
    range_max = 1

    # Population subset for EXTERNAL sexual reproduction
    # Todo: check if max bound can be removed
    external_pairs = filter(lambda ind: range_min < ind.fitness["accuracy_validation"] <= range_max, not_none_population)

    # Population subset for INTERNAL sexual reproduction
    internal_individuals = filter(lambda ind: range_min >= ind.fitness["accuracy_validation"], not_none_population)

    if len(external_pairs) % 2 == 1:
        new_random_position = random.randrange(0, len(external_pairs))

        # Todo check if this is correct
        # sorted_population = sorted(external_pairs, key=lambda ind: ind.fitness, reverse=False)
        # min_num = min(external_pairs, key=attrgetter('fitn'))


        # moving worst individual
        internal_individuals.append(external_pairs[new_random_position])
        del external_pairs[new_random_position]  # Todo check if correct

        # MOVER DE EXTERNAL PAIRS A
    # if not even number, move to other set

    # Todo assuming that external pair is even
    ########################################################
    # External
    ########################################################
    shuffle(external_pairs)

    for i in range(0, len(external_pairs), 2):

        ind1, ind2 = external_pairs[i], external_pairs[i+1]

        new_individual = crossover(ind1, ind2, config)
        new_individual.fitness = None
        new_population.append(new_individual)
    ########################################################

    ########################################################
    # Internal
    ########################################################
    for ind in internal_individuals:
        new_individual = mutation(ind, config)
        new_individual.fitness = None
        new_population.append(new_individual)

    ########################################################

    #print "NEW POL" + str(new_population)
    return new_population


# Crossover
# Todo: set this parameter according to evodeep
def crossover(ind1, ind2, config, indpb=0.5):

    # Fix crossover to get 1 individual
    new_individual1, new_individual2 = complete_crossover(ind1=ind1, ind2=ind2, indpb=indpb, config=config)

    return new_individual1


# Mutation
# Todo: to set these parameters according to evodeep
def mutation(ind, config, indpb=0.5, prob_add_remove_layer=0.5):

    new_individual = complete_mutation(ind1=ind, indpb=indpb, prob_add_remove_layer=prob_add_remove_layer, config=config)
    return new_individual


# Coral replacement
def larvae_settlement(reef, population, max_attempts=2):
    """

    :param max_attempts:
    :param population: set of chromosomes
    :param fitness: fitness of each individual
    :param nPobl: population size
    :param poolPopulation: solutions of a pool
    :param poolFitness: fitness of the pool population
    :param Natt: max attempts to replacement
    :return: newPopulation: selected population
    :return: newFitness: fitness of the new population
    """

    settled = 0

    for ind in population:

        #print "++" + str(ind)
        attempts = 0


        while attempts < max_attempts:

            new_random_position = random.randrange(0, len(reef))
            #print str(new_random_position)
            #print str(ind)
            if reef[new_random_position] is None:
                reef[new_random_position] = ind
                settled += 1
            elif reef[new_random_position].fitness["accuracy_validation"] < ind.fitness["accuracy_validation"]:
                reef[new_random_position] = ind
                settled += 1

            attempts += 1

    return reef, settled

# Depredation
def depredation(reef):
    """

    :param population: population to be depredated
    :param fitness1: fitness of the population
    :param Fd: percentage of the population to be depredated
    :param pDep: probability of depredation
    :return: newPopulation: depredated population
    :return: newFitness1: updated fitness
    """

    # Calculating fitness mean and std deviation
    fitness_mean, fitness_std, fitness_max, fitness_min = fitness_mean_std(reef)

    range_min = 0
    range_max = (fitness_mean - (2*fitness_std))

    not_none_population1 = len(filter(lambda w: w is not None, reef))

    # Deleting corals according to formula
    reef = [None if ind is not None and range_min <= ind.fitness["accuracy_validation"] <= range_max else ind for ind in reef]

    not_none_population2 = len(filter(lambda w: w is not None, reef))

    # print "Individuals depredated: " + str(not_none_population1 - not_none_population2)

    return reef, not_none_population1 - not_none_population2


def eval_population(reef, ke):
    count = 0
    for ind in reef:
        if ind is not None and ind.fitness is None:
            # TODO check if correct
            # If the fitness is not none, the individual did not change, so it keeps the same fitness
            ind.fitness = eval_keras(ind, ke)
            # ind.fitness = dummy_eval(ind)
            count += 1

    # print "New individuals evaluated: " + str(count)
            # ind.fitness = dummy_eval(ind)
    return reef, count


runSCRO()
