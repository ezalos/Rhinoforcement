# Rhinoforcement

## Abstract

The **AlphaZero** project by Google aroused my interest both as a developper and a chess player. In this project we shall attempt to use the main ideas behind the AlphaZero project and adapt them to our needs and to our small computing power. For those unfamiliar with AlphaZero it combines **Monte Carlo Tree Search** with **deep learning** to achieve unprecedented results in learning the game of chess. We shall apply this method to the game of Connect 4 for faster feedback and learning whilst keeping the program architecture flexible to facilitate a possible change of game.

Our objectives are:   

* Creating a program capable of learning any classical game.
* Exploring different neural net architectures.
* Exploring dataset management (duplicate deletion, depth compensation etc ..).
* Understanding Hyperparameters and their impact.
* Quantifying our results to make a comprehensive choice.
* Enable others to understand these choices and better adapt this to their projects.

## Monte Carlo Tree Search

MCTS DESCRIPTION    
needs :

* return a state
* return a set of all possible moves from a state
* identify terminal states
* finito

## Inserting a Neural Net

Architecture   
switching from UCB1 to PUCT.     
training pipeline.    
VS mode or continuous learning ?     
adapting our dataset to remove bias for early performance ?    
improve speed : dictionaries vs tree nodes and making a C library if needed.    
Making an infrastructure to record and display the performance impact of different choices.