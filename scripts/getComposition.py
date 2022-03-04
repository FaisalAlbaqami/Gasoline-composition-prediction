from GeneticAlgorithm import GeneticAlgorithm
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def prediction_output(fuel, ron):
    model_file_name = ".../ron/ron_post_publish.h5"
    generations = 30
    population_size = 50
    min_accuracy = 0.1
    genetic_algorithm = GeneticAlgorithm(model_file_name, fuel, generations, population_size, min_accuracy,
                                         ron)
    genetic_algorithm.fit()
    obtained_blend.append(genetic_algorithm.best_solution)
    found_ron.append(genetic_algorithm.best_result)
    ind_prop.append(genetic_algorithm.individual_properties)
    return obtained_blend, found_ron, ind_prop


def results_output(obtained_blend, found_ron, ind_prop):
    df_ind_prop = pd.DataFrame(ind_prop)
    df_blend = pd.DataFrame(obtained_blend)
    df_found_ron = pd.DataFrame(found_ron)
    df_found_ron.rename(columns={0: "found_ron"}, inplace=True)
    df_desired_ron = pd.DataFrame(desired_Ron)
    df_desired_ron.rename(columns={0: "desired_ron"}, inplace=True)
    df_results = pd.concat([df_blend, df_ind_prop, df_desired_ron, df_found_ron], axis=1).shift()[1:]
    df_results.index.name = "test_number"
    return df_results


