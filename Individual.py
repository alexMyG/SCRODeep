
import itertools
import random
from copy import deepcopy
from greenery.fsm import fsm
import sys
#################################################
#################################################
######   INDIVIDUALS AND FITNESS        #########
#################################################
#################################################





class GlobalAttributes:
    """
    @DynamicAttrs
    """

    def __init__(self, config):
        for global_parameter_name in config.global_parameters.keys():
            setattr(self, global_parameter_name, generate_random_global_parameter(global_parameter_name, config))


#################################################
# Individual class, from EvoDeep
#################################################
class Individual(object):
    def __init__(self, config, n_global_in, n_global_out):

        n_layers_start = 5
        num_min = 3
        self.configuration = config

        self.global_attributes = GlobalAttributes(self.configuration)

        self.net_struct = []

        self.fitness=None

        state_machine = fsm(alphabet=set(config.fsm['alphabet']),
                            states=set(config.fsm['states']),
                            initial="inicial",
                            finals={"Dense"},
                            map=config.fsm['map'])

        candidates = list(itertools.takewhile(lambda c: len(c) <= n_layers_start,
                                              itertools.dropwhile(lambda l: len(l) < num_min,
                                                                  state_machine.strings())))

        first_layers = list(set([b[0] for b in candidates]))
        candidates = [random.choice([z for z in candidates if z[0] == first_layers[l]]) for l in
                      range(len(first_layers))]

        sizes = list(set(map(len, candidates)))
        random_size = random.choice(sizes)
        candidates = filter(lambda c: len(c) == random_size, candidates)

        candidate = random.choice(candidates)
        candidate = map(lambda lt: Layer([lt], config), candidate)
        self.net_struct = candidate
        self.net_struct[0].parameters['input_shape'] = (n_global_in,)
        self.net_struct[-1].parameters['units'] = n_global_out
        self.global_attributes.number_layers = len(self.net_struct)

    def toString(self):

        output = ""
        global_attributes_dictionary = self.global_attributes.__dict__
        for item in sorted(global_attributes_dictionary.keys()):
            output += "Global attribute " + str(item) + ": " + str(global_attributes_dictionary[item]) + "\n"

        output += "Net structure: \n"

        for index, layer in enumerate(self.net_struct):
            output += "\t Layer " + str(index) + "\n"

            output += "\t\t Layer type: " + layer.type + "\n"

            for p in sorted(layer.parameters.keys()):
                output += "\t\t " + p + ": " + str(layer.parameters[p]) + "\n"

        return output

    def __repr__(self):
        return "I: L" + str(self.global_attributes.number_layers) + ",".join(map(str, self.net_struct))
#################################################

#################################################
# Layer class
#################################################
class Layer:
    """
    Class representing each layer of the Keras workflow
    """

    def __init__(self, possible_layers, config, layer_position=None, n_input_outputs=None):
        """
        Fixed arguments of each layers (those not represented in the individual) such as in or out,
        are direct attributes
        Parameters are under the self.parameters

        :param possible_layers: name of possible next layers
        :param config: configuration object
        :param layer_position: position of the layer to be added

        """

        self.type = random.choice(possible_layers)
        self.parameters = {}

        for param in config.layers[self.type].keys():

            if param != "parameters":
                setattr(self, param, config.layers[self.type][param])
            else:
                for p in config.layers[self.type][param]:
                    self.parameters[p] = generate_random_layer_parameter(p, self.type, config)

        # Deal with number of neurons in first and last layer
        if layer_position == 'first':
            # self.type = 'Dense'
            self.parameters['input_shape'] = (n_input_outputs,)
        if layer_position == 'last':
            # Last layer is forced to be dense
            self.type = 'Dense'
            self.parameters = dict()
            for param in config.layers[self.type].keys():

                if param != "parameters":
                    setattr(self, param, config.layers[self.type][param])
                else:
                    for p in config.layers[self.type][param]:
                        self.parameters[p] = generate_random_layer_parameter(p, self.type, config)
            self.parameters['units'] = n_input_outputs

    def __repr__(self):
        return "[" + self.type[:2] + "(" + "|".join(
            map(lambda (k, v): k[:4] + ":" + str(v), self.parameters.items())) + ")]"
#################################################

def dummy_eval(individual):

    evaluation = {
        "accuracy_validation": random.random(),
        "number_layers": random.randrange(2, 10),
        "accuracy_training": random.random(),
        "accuracy_test": random.random()
    }

    # The number of layers attribute in the individual is not updated when using a dummy eval.

    return evaluation


def eval_keras(individual, ke):
    sys.stdout.write(".")
    sys.stdout.flush()

    my_ke = deepcopy(ke)

    metrics_names, scores_training, scores_validation, scores_test, model = my_ke.execute(individual)
    accuracy_training = scores_training[metrics_names.index("acc")]
    accuracy_validation = scores_validation[metrics_names.index("acc")]
    accuracy_test = scores_test[metrics_names.index("acc")]

    number_layers = individual.global_attributes.number_layers

    evaluation = {
        "accuracy_validation": accuracy_validation,
        "number_layers": number_layers,
        "accuracy_training": accuracy_training,
        "accuracy_test": accuracy_test
    }
    return evaluation


def create_random_valid_layer(config, last_layer_output_type, n_input_outputs=None, layer_position=None):
    """
    Generates a new valid randomly generated layer coherent with the previous existent layer
    :param n_input_outputs:
    :param config: configuration object
    :param layer_position: position of the layer to be added
    :param last_layer_output_type: output type of the previous existent layer
    :return:
    """
    possible_layers = []

    for layer_name, layer_config in config.layers.items():
        if layer_config['in'] == last_layer_output_type:
            possible_layers.append(layer_name)
    layer = Layer(possible_layers, config, layer_position, n_input_outputs)

    return layer

def parser_parameter_types(parameter_config, parameter):
    if parameter == "categorical":
        return parameter_config["values"][random.randrange(0, len(parameter_config["values"]))]

    elif parameter == "range":
        return random.randrange(*parameter_config["values"])

    elif parameter == "rangeDouble":
        return round(random.uniform(*parameter_config["values"]), 1)

    elif parameter == "matrixRatio":
        # return gen_matrix_ratio_tuple(parameter_config["aspect_ratio"], n_neurons_prev_layer)
        return parameter_config["aspect_ratio"]

    elif parameter == "categoricalNumeric":
        val = parameter_config["values"][random.randrange(0, len(parameter_config["values"]))]

        if val:
            return val, val
        else:
            return None

    elif parameter == "2Drange":
        return [random.randrange(*parameter_config["values"]) for _ in range(parameter_config["size"])]

    elif parameter == "boolean":
        return bool(random.getrandbits(1))

    else:
        print "PARAMETER " + parameter + " NOT DEFINED"

def generate_random_global_parameter(parameter_name, configuration):
    """
    This method generates a new random value based on
    :param parameter_name: the parameter for which a new value is given
    -parameter_name- can take. This param contains the whole configuration dictionary
    :param configuration:
    :return:
    """
    parameter_type = configuration.global_parameters[parameter_name]["type"]
    parameter_config = configuration.global_parameters[parameter_name]

    return parser_parameter_types(parameter_config, parameter_type)

def generate_random_layer_parameter(parameter_name, layer_type, configuration):
    """
    This method generates a new random value based on
    :param configuration:
    :param layer_type:
    :param parameter_name: the parameter for which a new value is given
    -parameter_name- can take. This param contains the whole configuration dictionary
    :return:
    """
    if "parameters" not in configuration.layers[layer_type]:
        return None

    parameter_type = configuration.layers[layer_type]["parameters"][parameter_name]["type"]
    parameter_config = configuration.layers[layer_type]["parameters"][parameter_name]

    return parser_parameter_types(parameter_config, parameter_type)

