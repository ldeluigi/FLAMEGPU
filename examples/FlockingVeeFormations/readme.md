# FLAME GPU: Flocking Vee Formations

This project is an implementation of the homonym model, where agents behave like birds, following simple rules that produce the formation of flocks shaped like 'V's. These type of flock are observed with some species of volatiles, like migratory birds, because of the efficiency improvement for long distance flights, thanks to areodynamics.

## The Model

The model is coded after the FlockingVeeFormations model of the NetLogo model library, which can be found at https://ccl.northwestern.edu/netlogo/models/FlockingVeeFormations.

The NetLogo model itself is coded after the Flocking NetLogo model idea, but differs completely in the implementation. One of the most important differences is that this model makes use of Random Number Generators, which make the model non-deterministic (theoretically speaking).

## Results

After some iterations, depending on the parameters, different 'V' flock formations can be observed. As the simulation keeps going, they tend to be bigger and fewer, but, depending on the parameters, they can create enough disturbs to each other so that the time required to create a single, giant V flock with every bird is extremely large.

## Visualization

The visualization code is written with OpenGL and it's based on the default dynamically generated visualization code of the Flame GPU framework, but was heavily modified to allow the visualization of cones that point in the direction of the movement. A different control system and camera movement was added to provide a better visualization of the flocks. This code entirely is shared with the Flocking Flame GPU model.

You can use gnuplot and plot.gp to draw a graph with the output of the program if compiled with FLOCKING_PLOT defined.

## Performances

The entire C++/CUDA code was thought and written with performances in mind, knowing the capability of the Flame GPU acrhitecture and the CUDA features. As a result, the simulation can easily support thousands of agents without a single drop in fps, depending on the hardware that is run on.