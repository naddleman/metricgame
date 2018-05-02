Finds Nash equilibria of spatial games
======================================

A spatial game features a population of agents, a metric space, and a symmetric
2-by-2 game. These programs place agents uniformly at random on the space,
initializing them with random strategies. An agent is picked at random to revise
its strategy until the process reaches equilibrium

Revision
--------

The payoff to agent *i* is the weighted sum of payoffs from playing against
every other agent in the population. The weight for an interaction between *i*
and *j* is given by a (decreasing) function of the distance, *d(i,j)*
