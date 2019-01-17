import numpy as np
import json
import urllib
import os.path
import os
os.environ['KERAS_BACKEND']='tensorflow'
from random import shuffle
from KerasExecutor import KerasExecutor
from operator import attrgetter

from scipy.io import loadmat

from OperatorsDeep import complete_crossover, complete_mutation

from Individual import *
metrics = ["accuracy"]
early_stopping_patience = 100
loss = "categorical_crossentropy"


"""
IMPORTANT NOTE

REEF IS A LIST WITH INDIVIDUALS AND NONE POSITIONS.
POPULATION IS A SET OF INDIVIDUALS WHERE THE ORDER IN THE LIST DOES NOT MATTER WHICH CAN CONTAIN NONE VALUES

"""

class Configuration(object):
    def __init__(self, j):
        self.__dict__ = json.load(j)



def runSCRO(numIt, nPobl, numSeg, pCross, pMut, seed, sizeChromosome, polyDegree, percentage_hybridation):

    return None

def runTest():

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

    test_size = 0.2

    print "Starting keras executor"
    ke = KerasExecutor(dataset, test_size, metrics, early_stopping_patience, loss)


    config_data = open('parametersGenetic.json')

    configuration = Configuration(config_data)

    reef = initialisation(Rsize=3, rate_free_corals=0, config=configuration, n_global_in=deepcopy(ke.n_in), n_global_out=ke.n_out, ke=ke)
    # Population is already evaluated in the initialisation function

    # loop

    pool = []

    # 1 Asexual reproduction

    asexual_new_individual = asexual_reproduction(reef, configuration)
    pool = pool + [asexual_new_individual]

    # 2 Sexual reproduction

    sexual_new_individuals = sexual_reproduction(reef, configuration)
    pool = pool + sexual_new_individuals

    # 3 Larvae settlement

    eval_population(pool, ke)

    reef, settled = larvae_settlement(reef, pool)


    # 4 Evaluation

    # TODO: check if population is updated


    # 5 Depredation

    reef = depredation(reef)

    # stop criteria check


#################################################
#################################################
######              SCRO                #########
#################################################
#################################################



#  OPERATORS

# TODO: check: reef is a list.
def initialisation(Rsize, rate_free_corals, config, n_global_in, n_global_out, ke):
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

    eval_population(reef, ke)
    for ind in reef:
        print str(ind.fitness)

    # Calculating fitness mean and std deviation
    fitness_mean, fitness_std = fitness_mean_std(reef)

    # Deleting corals according to formula
    # It is not the same that the depredation one
    # new_population = [[ind if initial_deletion_check(ind.fitness, fitness_mean, fitness_std) else None for ind in line ] for line in population]
    new_reef = [
        ind if initial_deletion_check(ind.fitness["accuracy_validation"], fitness_mean, fitness_std) else None for
        ind in reef]

    print "Population reduced to: " + str(len(filter(lambda w: w is not None, new_reef))) + " solutions"

    for ind in filter(lambda w: w is not None, new_reef):
        print str(ind.fitness)

    return new_reef

def initial_deletion_check(fitness, fitness_mean, fitness_std):

    return (fitness_mean - fitness_std) < fitness <= 1


def fitness_mean_std(reef):

    # fitnesses_reef = np.array([[eval_keras(x, ke) if x is not None else None for x in line ] for line in population])

    # Todo: originally individuals will not be evaluated? : to check
    # Todo: create fitness in the individual
    fitnesses_reef = np.array([ind.fitness["accuracy_validation"] for ind in reef if ind is not None])  # None are removed

    fitness_mean = fitnesses_reef.mean()
    fitness_std = fitnesses_reef.std()

    return fitness_mean, fitness_std

# Asexual reproduction

def asexual_reproduction(reef, config):

    selected_individual = asexual_selection(reef)
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

    fitness_mean, fitness_std = fitness_mean_std(population)
    range_min = (fitness_mean + fitness_std)
    range_max = 1
    fragmentation = filter(lambda ind: range_min < ind.fitness["accuracy_validation"] <= range_max, population)


    # sorted_population = sorted(population, key=lambda coral: coral.fitness if coral is not None else -1, reverse=True)
    # max_value = round(fa * len(filter(lambda x: x is not None, population)))
    idx = random.randrange(0, len(fragmentation))
    # TODO: What if there is nothing but holes in the population? -> Is it that possible?

    aLarvae = deepcopy(fragmentation[idx])
    return aLarvae

# Sexual reproduction
def sexual_reproduction(reef, config):
    # population.sort(key=lambda x: x.fitness, reverse=True)

    new_population = []

    # A random fraction Fb of the individuals is selected uniformly

    not_none_population = filter(lambda w: w is not None, reef)

    fitness_mean, fitness_std = fitness_mean_std(not_none_population)

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

    print "NEW POL" + str(new_population)
    return new_population


# Crossover
# Todo: set this parameter according to evodeep
def crossover(ind1, ind2, config, indpb=0.2):

    # Fix crossover to get 1 individual
    new_individual1, new_individual2 = complete_crossover(ind1=ind1, ind2=ind2, indpb=indpb, config=config)

    return new_individual1


# Mutation
# Todo: to set these parameters according to evodeep
def mutation(ind, config, indpb=0.2, prob_add_remove_layer=0.2):

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

    for ind in population:
        attempts = 0
        settled = 0

        while attempts < max_attempts:

            new_random_position = random.randrange(0, len(reef))

            print "-" + str(reef[new_random_position])
            print "+" + str(ind)
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
    fitness_mean, fitness_std = fitness_mean_std(reef)

    range_min = 0
    range_max = (fitness_mean - (2*fitness_std))

    # Deleting corals according to formula
    reef = [None if ind is not None and range_min <= ind.fitness["accuracy_validation"] <= range_max else ind for ind in reef]

    return reef


def eval_population(reef, ke):

    for ind in reef:
        if ind is not None and ind.fitness is None:
            # TODO check if correct
            # If the fitness is not none, the individual did not change, so it keeps the same fitness
            ind.fitness = eval_keras(ind, ke)
            # ind.fitness = dummy_eval(ind)
    return reef


#runTest()
