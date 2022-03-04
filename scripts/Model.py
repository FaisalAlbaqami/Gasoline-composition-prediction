import numpy as np
from keras.engine.saving import load_model


class Model:

    parameters = ["ch3_wt%", "ch2_wt%", "ch_wt%", "olef_wt%", "naph_wt%", "aroma_wt%", "oh_wt%", "mol_wt", "bi"]
    parameter_count = len(parameters)

    def __init__(self, file_name):
        self.file_name = file_name
        self.model = load_model(file_name)

    def predict(self, key_values):
        """
        :param key_values: dictionary containing key names and respective values.
        :return: the RON predicted by the model given the information in key_values.
        """
        # Create the input_vector from the key_values and order of parameters.
        input_vector = np.zeros((1, self.parameter_count))
        for i in range(0, self.parameter_count):
            input_vector[0, i] = key_values[self.parameters[i]]

        # Returns it as an array of arrays thus the indexing.
        return self.model.predict(input_vector)[0][0]
