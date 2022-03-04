from GeneticAlgorithm import GeneticAlgorithm
import pandas as pd
import os
import random
import cProfile

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
random.seed(1)


def read_file(valid_set, fuel_props):
    df_valid_set = pd.read_csv(valid_set, index_col=0)
    df_fuel_prop = pd.read_csv(fuel_props, index_col=0)

    return df_valid_set, df_fuel_prop


def predict_composition(fuel_data, model_on, on, fuel_props):
    global desired_on
    list_predicted_on = []
    list_predicted_blend = []
    for i in range(len(fuel_data)):
        row = fuel_data.iloc[i, :-2].dropna()
        fuel_list = row.values.tolist()
        if on == 'RON':
            desired_on = fuel_data.iloc[i, 16:17]
        if on == 'MON':
            desired_on = fuel_data.iloc[i, 17:18]
        desired_on = float(desired_on)
        obtained_blend_on, predicted_on = prediction_output(fuel_list, desired_on, model_on, fuel_props)
        list_predicted_blend.append(obtained_blend_on)
        list_predicted_on.append(predicted_on)
    predicted_blend = pd.DataFrame(list_predicted_blend, index=fuel_data.index.copy()).fillna(value="")
    predicted_blend["Predicted"] = list_predicted_on
    predicted_blend["Predicted"] = predicted_blend["Predicted"].round(2)

    return predicted_blend


def prediction_output(fuel_data, on, model, fuel_p):
    generations = 300
    population_size = 200
    min_accuracy = 0.01
    genetic_algorithm = GeneticAlgorithm(model, fuel_data, generations, population_size, min_accuracy,
                                         on, fuel_p)
    genetic_algorithm.fit()
    predicted_blend = genetic_algorithm.best_solution
    predicted_on = genetic_algorithm.best_result

    return predicted_blend, predicted_on


def csv_creation(data, output_file_name):
    data.to_csv(output_file_name)


"""
Manually select the set
"""

dataset = 'set_I'
# dataset = 'set_II'
# dataset = 'set_III'
# dataset = 'set_IIII'

set_I = "../set_I.csv "
set_II = "../set_II.csv "
set_III = "../set_III.csv "
set_IIII = "../set_IIII.csv "
model_file_ron = "../models/ron.h5"
model_file_mon = "../models/mon.h5"
fuel_properties = "../fuel_properties.csv"

if dataset == 'set_I':
    dataset = set_I
    predicted_output_file = "../results_I.csv"
    fuel_input, fuel_prop = read_file(dataset, fuel_properties)
    predicted_fuel_composition = predict_composition(fuel_input, model_file_ron, model_file_mon, fuel_prop)
    csv_creation(predicted_fuel_composition, predicted_output_file)

if dataset == 'set_II':
    dataset = set_II
    predicted_output_file = "../results_II.csv"
    fuel_input, fuel_prop = read_file(dataset, fuel_properties)
    predicted_fuel_composition = predict_composition(fuel_input, model_file_ron, model_file_mon, fuel_prop)
    csv_creation(predicted_fuel_composition, predicted_output_file)

if dataset == 'set_III':
    dataset = set_III
    predicted_output_file = "../results_III.csv"
    fuel_input, fuel_prop = read_file(dataset, fuel_properties)
    predicted_fuel_composition = predict_composition(fuel_input, model_file_ron, model_file_mon, fuel_prop)
    csv_creation(predicted_fuel_composition, predicted_output_file)

if dataset == 'set_IIII':
    dataset = set_IIII
    predicted_output_file = "../results_IIII.csv"
    fuel_input, fuel_prop = read_file(dataset, fuel_properties)
    predicted_fuel_composition = predict_composition(fuel_input, model_file_ron, model_file_mon, fuel_prop)
    csv_creation(predicted_fuel_composition, predicted_output_file)

