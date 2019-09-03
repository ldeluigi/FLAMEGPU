# FLAME GPU: Flocking

This project is an implementation of the homonym model, where agents behave like birds, following simple rules that produce the formation of flocks.

## The Model

The model is coded after the Flocking model of the NetLogo model library, which can be found at https://ccl.northwestern.edu/netlogo/models/Flocking.

Nevertheless, the model was slightly modified and expanded with the implementation of a limited field of view (expressed in degrees and distance) for each bird, allowing the study of the effects of such variable in the overall behavior.

The NetLogo model itself is coded after another Flocking model called Boids, documented at https://www.red3d.com/cwr/boids/, but presents heavy differences in the implementation from the original.

## Results

After some iterations, depending on the parameters, different flock formations can be observed. As the simulation keeps going, they tend to be bigger and fewer, up to the point where pretty much every single bird goes in the same direction as the others.

## Visualization

The visualization code is written with OpenGL and it's based on the default dynamically generated visualization code of the Flame GPU framework, but was heavily modified to allow the visualization of cones that point in the direction of the movement. A different control system and camera movement was added to provide a better visualization of the flocks. This code entirely is shared with the Flocking Vee Formations Flame GPU model.

In addition to that, the colour of each bird represents its state (alone, in cohesion/alignment or in separation). More information can be found in the documentation of the model.

## Performances

The entire C++/CUDA code was thought and written with performances in mind, knowing the capability of the Flame GPU acrhitecture and the CUDA features. As a result, the simulation can easily support thousands of agents without a single drop in fps, depending on the hardware that is run on.