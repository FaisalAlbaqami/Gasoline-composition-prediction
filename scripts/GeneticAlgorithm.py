import sys
from operator import itemgetter
from Model import Model
import random
import math
import pandas as pd


def find_initial_score(n):
    i = 0
    sums = 0
    scores = []
    scoring_sum = 0
    if n > 5:
        for k in range(2):
            remain = n - k
            sums = remain + sums
    else:
        for k in range(n):
            remain = n - k
            sums = remain + sums
    while i < n:
        score = ((100 - scoring_sum) / (n ** 2)) * sums
        scoring_sum = scoring_sum + score
        scores.append(score)
        i = i + 1
    while sum(scores) < 99:
        residual = 100 - sum(scores)
        for x in range(len(scores)):
            scores[x] = scores[x] + (scores[x]/100)*residual
    return scores


def find_score(parents, score, best_fitness, desired_output):
    reached_limit_list = []
    compositions = len(list(parents[0]["individual"]))
    for k in range(compositions):
        i = 0
        comp_list = []
        while i < len(parents):
            n = list(parents[i]["individual"].values())
            comp_list.append(n[k])
            i = i + 1
        reached_limit = check_proximity(comp_list, score[k])
        reached_limit_list.append(reached_limit)
    new_score = find_new_score(score, best_fitness, desired_output, reached_limit_list)
    score = new_score
    return score


def find_new_score(old_score, fitness, on_desired, limit_reached):
    global new_score_diff_percent
    score_value = old_score
    i = 0
    count_no_change = 0
    while i < len(score_value):
        if limit_reached[i] == "upper":
            score_value[i] = ((1 + ((fitness/2) / on_desired)) * old_score[i])  # changed from /2 to *2
            if score_value[i] > 75:
                score_value[i] = float(75)
        if limit_reached[i] == "lower":
            score_value[i] = ((1 - ((fitness/2) / on_desired)) * old_score[i])
            if score_value[i] < 5:
                score_value[i] = float(5)
        if score_value[i] < 5:
            score_value[i] = float(5)
        if limit_reached[i] == "no change":
            score_value[i] = old_score[i]
            count_no_change = count_no_change + 1
        i = i + 1
    sums = sum(score_value)
    diff = 100 - sums
    s = 0
    k = 0
    if count_no_change > 0:
        new_score_diff_percent = find_initial_score(count_no_change)
        min_value = min(score_value)
        min_index = score_value.index(min_value)
        min_diff_percent = min(new_score_diff_percent)
        min_diff_percent_index = new_score_diff_percent.index(min_diff_percent)

        if score_value[min_index] < 4 and limit_reached[min_index] == 'no change' and count_no_change > 1:
            count_no_change = count_no_change - 1
            limit_reached[min_index] = "skip"
            new_score_diff_percent = find_initial_score(count_no_change)

        elif (score_value[min_index] + diff * new_score_diff_percent[min_diff_percent_index] / 100) < 1 \
                and limit_reached[min_index] == 'no change' and count_no_change > 1:
            count_no_change = count_no_change - 1
            limit_reached[min_index] = "skip"
            new_score_diff_percent = find_initial_score(count_no_change)

    while k < len(score_value):
        if limit_reached[k] == "no change":
            score_value[k] = score_value[k] + diff * new_score_diff_percent[s] / 100
            s = s + 1
        k = k + 1
    return score_value


def find_range(scores):
    global lower, upper
    margin_1 = 0.25
    margin_2 = 0.35
    margin_3 = 0.50
    ranges = []
    for i in range(len(scores)):
        if scores[i] > 50:
            lower = scores[i] * (1 - margin_1)
            upper = scores[i] * (1 + margin_1)
            if upper > 91:
                upper = 91
        if 20 <= scores[i] <= 50:
            lower = scores[i] * (1 - margin_2)
            upper = scores[i] * (1 + margin_2)
        if scores[i] < 20:
            lower = scores[i] * (1 - margin_3)
            upper = scores[i] * (1 + margin_3)
            if lower < 0:
                lower = 0
        rang = (lower, upper)
        ranges.extend([rang])
    return ranges


def check_proximity(comp, score):
    count_up = 0
    count_low = 0
    comp_margin = 0.10
    margin = 0.25
    number_comp_exceed = 0.50 * len(comp)
    lower = score * (1 - margin)
    upper = score * (1 + margin)
    for i in range(len(comp)):
        if comp[i] > ((1 - comp_margin) * upper):
            count_up = count_up + 1
        if comp[i] < ((1 + comp_margin) * lower):
            count_low = count_low + 1
    if count_up > number_comp_exceed:
        limit = "upper"
    elif count_low > number_comp_exceed:
        limit = "lower"
    else:
        limit = "no change"

    return limit


def generate_n_percentages(n, score, total=100):
    """
    This function is based on trial and error and will perform very slowly for large numbers of n.
    :param score:
    :param n: the number of numbers needed
    :param total: the required total
    :return: an array of numbers in between lower and upper bounds summing to total
    """
    ranges = find_range(score)
    hit = False
    nums = None
    while not hit:
        summed, count = 0, 0
        nums = []
        while summed < total and count < n:
            lower, upper = ranges[count]
            lower = int(lower)
            upper = int(upper)
            num = random.randint(lower, upper)
            summed += num
            count += 1
            nums.append(num)
        if summed == total and count == n:
            hit = True
    return nums


def create_individual_crossover(parents):
    """
    Will create a new individual by averaging the values of two
    random parents from parents.
    :param parents: the parents to use to create new individuals
    :return: a new individual made from two distinct parents
    """
    male_index = random.randint(0, len(parents) - 1)
    female_index = random.randint(0, len(parents) - 1)
    while female_index == male_index:
        female_index = random.randint(0, len(parents) - 1)
    male = parents[male_index]["individual"]
    female = parents[female_index]["individual"]
    individual = {}
    for key in male.keys():
        individual[key] = (male[key] + female[key]) / 2
    return individual


def get_properties(individual, df):
    """
    Mol wt & BI are calculated based on mole contribution. The rest is based on weight contribution.
    :param df:
    :param individual: dictionary of fuel name and respective volume contribution as key values.
    :return: an array representing the resultant functional groups, mole weight and BI.
    """
    individual_properties = {}
    total_weight = 0
    total_moles = 0
    for name, volume in individual.items():
        fuel = df.loc[name].to_dict()
        # Get density and drop from dict, if there is no density take it as zero
        density = fuel.pop("density", 0)
        weight = density * volume
        moles = weight / fuel["mol_wt"]
        contribution = {}
        for k, v in fuel.items():
            if k == "mol_wt" or k == "bi":
                contribution[k] = moles * v
            else:
                contribution[k] = weight * v * 100
        total_moles += moles
        total_weight += weight
        # Add contribution to the individual properties
        for k, v in contribution.items():
            # Initialize keys
            if k not in individual_properties:
                individual_properties[k] = 0
            individual_properties[k] += v
    # Divide weight based properties by weight and the others by moles
    for k, v in individual_properties.items():
        if k == "mol_wt" or k == "bi":
            individual_properties[k] = v / total_moles
        else:
            individual_properties[k] = v / total_weight
    return individual_properties


class GeneticAlgorithm:
    """
    Encapsulates the genetic algorithm together with the model to be used and stopping criteria.
    An individual is represented through.
    """

    def __init__(self, ron_model, mon_model, fuels, generations, population_size, min_accuracy, desired_ron_output, desired_mon_output, fuel_prop):
        self.model_ron = Model(ron_model)
        self.model_mon = Model(mon_model)
        self.fuels = fuels
        self.generations = generations
        self.population_size = population_size
        self.min_accuracy = min_accuracy
        self.desired_ron_output = desired_ron_output
        self.desired_mon_output = desired_mon_output
        self.best_solution = None
        self.best_result_ron = sys.maxsize
        self.best_result_mon = sys.maxsize
        self.best_aki = sys.maxsize
        self.individual_properties = None
        self.fuel_prop = fuel_prop

    def create_individual(self, scores):
        """
        Will generate random volume percentages for each fuel, making sure the sum is 100.
        :return: an individual represented by a dictionary
        """
        fuel_count = len(self.fuels)
        values = generate_n_percentages(fuel_count, scores)
        return dict(zip(self.fuels, values))

    def create_initial_population(self):
        individuals = []
        scores = find_initial_score(len(self.fuels))
        while len(individuals) < self.population_size:
            individuals.append(self.create_individual(scores))
        return individuals, scores

    def fit(self):
        """
        This is the main function that handles population generation, and evolution.
        It will stop when generations or the min_accuracy has been reached.
        """
        individuals, score = self.create_initial_population()
        completed_generations = 0
        count = 0
        prev_fitness = 0
        # Stopping conditions
        while completed_generations < self.generations and \
                abs(self.best_result_ron - self.desired_ron_output) > self.min_accuracy:
            # print("Completed generations: {}".format(completed_generations))
            # Get fitness for each
            individuals_and_fitness = [dict([("individual", individual),
                                             ("fitness", self.individual_fitness(individual))])
                                       for individual in individuals]
            # Sort on fitness
            individuals_and_fitness.sort(key=itemgetter("fitness"))
            # Take top third of individuals as parents
            parents = individuals_and_fitness[0: self.population_size // 3]
            children = []
            best_fitness = list(parents[0].values())[1]
            if prev_fitness == 0:
                prev_fitness = best_fitness
            if best_fitness == prev_fitness:
                count = count + 1
            else:
                prev_fitness = best_fitness
                count = 0
            if count > 20 and best_fitness < 1:
                break
            elif count > 20:
                break
            if best_fitness > 1:
                # use the AKI to determine the scoring
                score = find_score(parents, score, best_fitness, (self.desired_ron_output + self.desired_ron_output)/2)
                child = self.create_individual(score)
                children.append(child)
                while self.population_size - len(parents) - len(children) > 0:
                    child = create_individual_crossover(parents)
                    # Form of mutation and avoids getting stuck in this loop
                    if child in children:
                        child = self.create_individual(score)
                    children.append(child)
            individuals = [parent["individual"] for parent in parents]
            individuals.extend(children)
            completed_generations += 1

    def individual_fitness(self, individual):
        """
        Takes an individual and returns it fitness. This function will also update the best_solution and best_result
        if this individual is the closest yet to the desired_output. 0 is best.
        :param individual: the individual to be evaluated
        :return: the difference between the individual's result and the desired_output
        """
        individual_properties = get_properties(individual, self.fuel_prop)
        result_ron = self.model_ron.predict(individual_properties)
        result_mon = self.model_mon.predict(individual_properties)
        aki_desired = (self.desired_ron_output + self.desired_mon_output) / 2
        result_aki = (result_mon + result_ron) / 2

        if abs(aki_desired - result_aki) < abs(aki_desired - self.best_aki):
            print("Best solution so far: {}".format(individual))
            print("With RON result: {}".format(result_ron))
            print("With MON result: {}".format(result_mon))
            self.best_solution = individual
            self.best_result_ron = result_ron
            self.best_result_mon = result_mon
            self.individual_properties = individual_properties
            self.best_aki = result_aki

        return (abs(self.desired_ron_output - result_ron)+abs(self.desired_mon_output - result_mon))/2
