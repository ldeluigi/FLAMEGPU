#ifndef __VISUALISATION_H
#define __VISUALISATION_H

#include <GL/glew.h>


#define SIMULATION_DELAY 1

// constants
const unsigned int WINDOW_WIDTH = 1280;
const unsigned int WINDOW_HEIGHT = 720;

//frustrum
const float NEAR_CLIP = 0.1f;
const float FAR_CLIP = 300;

//Circle model fidelity
const int CONE_SLICES = 8;
const float CONE_HEIGHT = 1.0f;
const float CONE_RADIUS = 0.25f;

//Viewing Distance
const float VIEW_DISTANCE = 16;

//light position
GLfloat LIGHT_POSITION[] = { 10.0f, 10.0f, 10.0f, 1.0f };

#endif //__VISUALISATION_H