
from os import listdir

import pandas as pd
reef_evolution_files = filter(lambda x: x.endswith("REEF_EVOLUTION.csv"), listdir("."))





#def plot_evolution_fitness_one_execution(file_name):






#plot_evolution_fitness_one_execution(reef_evolution_files[0])


file_name = reef_evolution_files[1]


data = pd.read_csv(file_name)


data.groupby('generation', as_index=False).agg({"accuracy_validation": ["mean", "std"],
                                                "accuracy_training": ["mean", "std"],
                                                "accuracy_test": ["mean", "std"]})


