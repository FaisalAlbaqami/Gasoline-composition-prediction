# Gasoline-composition-prediction


This is the official repository for the "A Methodology for Designing Octane Number of Fuels Using Genetic Algorithms and Artificial Neural Networks" paper. The Gasoline Composition is predicted using an integerated system composed of Genetic Algorithms (GA) and Artificial Neural Networks (ANN) to achieve certain fuel charatarstics (i.e. RON and MON). The ANN model uses functional groups, branching index, and molecular weight to predict the gasoline charactarstics.

The fuel compositions and the desired charactarstics are fed to the integerated system to present the recipes that meet the desired specifications. Furthumore, the polygonal method, utilizes a scoring and adabtive learning method to select the best recipe that meets predetermined critirea. 

Please refer to the paper for a much more detailed description of the methodology.


## Overview of files

data/
set.csv: complete dataset used for the paper
fuel_properties: fuel properties used to convert the inputted fuel composition to its chemical characteristics

models/: ron & mon models developed using ANN.

results/: csv files containing the results.

scripts/
GeneticAlgorithm.py: contains the script for both GA and the polygonal algorithms. The script creates multiple recipes and select the optimum fuel that meets the desired criteria
getComposition.py: is used to present the compositions results.
model.py: is used to read the model .h5 files
MON_RON.py: it is used to read the input .csv files and operate as the center of the integrated structure. 


## Authorship

All code was written by Vincent C.O. van Oudenhoven and Faisal Albaqami. 
Faisal Albaqami and Abdul Gani Abdul Jameel were responsible for the used data. Please refer to the publication itself for the full data source.


## Acknowledgement

The authors would like to acknowledge the support received from the Interdisciplinary Center for Refining & Advanced Chemicals (CRAC), King Fahd University of Petroleum and Minerals (KFUPM), Saudi Arabia.


## BibTex

@article{article,
  author  = {Faisal Alboqami, Vincent C.O. van Oudenhoven, Usama Ahmed, Umer Zahid, Abdul-Hamid Emwas, S.Mani Sarathy, Abdul Gani Abdul Jameel}, 
  title   = { A Methodology for Designing Octane Number of Fuels Using Genetic Algorithms and Artificial Neural Networks},
  year    = 2022
}

