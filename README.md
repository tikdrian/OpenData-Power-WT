# OpenData-Power-WT
# Energy Hackdays 2020, August 28 - 29, 2020, Hightech Zentrum in Brugg
# http://hack.opendata.ch/project/471
# Challenge: 08 ML Wind Power-Prediction
## Goal:
## - Development of machine learning algorithms for improved site-specific performance prediction of wind turbines.
## - Development of alternative algorithms e.g. Artificial Neural Networks
## # Dataset of 8'000 simulated powers of the NREL 5MW reference wind turbine from the simulation tool ASHES at a range of different atmospheric conditions.

## Tested Solution:
## - Dataset split in 80% training set and 20% test set
## - Neural Nework algorithm with up to 5 layers (3 hidden layers)
### - Tested --> Keras model, fully connected Dense clas, with different number of nodes in the layer and rectified linear unit activation function.
