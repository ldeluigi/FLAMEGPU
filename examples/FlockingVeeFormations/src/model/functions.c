
/*
 * FLAME GPU v 1.5.X for CUDA 9
 * Copyright University of Sheffield.
 * Original Author: Dr Paul Richmond (user contributions tracked on https://github.com/FLAMEGPU/FLAMEGPU)
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence
 * on www.flamegpu.com website.
 *
 */


#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>

#define PI_F 3.141592654f
#define PI2_F (PI_F * 2)
#define degreesToRadians(angleDegrees) ((angleDegrees) * PI_F / 180.0)

/**
 * Constraints the angle in input to a value in the interval [0, 2 * PI_F] where PI_F is pi in float.
 */
__FLAME_GPU_FUNC__ float constraintRadians(float r)
{
	float a = fmodf(r, PI2_F);
	return a < 0 ? a + PI2_F : a;
}

/**
 * Calculate the heading represented by the dx and dy distances towards the target,
 * with and angle in the interval [0, 2 * PI_F] where PI_F is pi in float.
 */
__FLAME_GPU_FUNC__ float calculateHeading(float dx, float dy)
{
	return constraintRadians(atan2f(dy, dx));
}

/**
 * Rotates the current heading clockwise.
 */
__FLAME_GPU_FUNC__ float right_turn(float current_heading, float turn)
{
	return constraintRadians(current_heading - turn);
}

/**
 * Rotates the current heading counterclockwise.
 */
__FLAME_GPU_FUNC__ float left_turn(float current_heading, float turn)
{
	return constraintRadians(current_heading + turn);
}

/**
 * Returns the minimum difference between two given angles in radians,
 * with a value inside the interval [-PI_F/2, PI_F/2], where PI_F is pi in float.
 */
__FLAME_GPU_FUNC__ float subtract_headings(float h1, float h2)
{
	float a = h1 - h2;
	return a + ((a > PI_F) ? -PI2_F : (a < -PI_F) ? PI2_F : 0);
}

/**
 * Returns a new heading which has performed a rotation of at most max_turn from the current_heading, counterclockwise.
 */
__FLAME_GPU_FUNC__ float turn_at_most(float turn, float current_heading, float max_turn)
{
	if (fabs(turn) > max_turn)
	{
		if (turn > 0)
		{
			return left_turn(current_heading, max_turn);
		}
		else
		{
			return right_turn(current_heading, max_turn);
		}
	}
	else
	{
		return left_turn(current_heading, turn);
	}
}

/**
 * Returns a new heading which has performed a rotation away from the target_heading.
 */
__FLAME_GPU_FUNC__ float turn_away(float target_heading, float current_heading, float max_turn)
{
	return turn_at_most(subtract_headings(current_heading, target_heading), current_heading, max_turn);
}

/**
 * Returns a new heading which has performed a rotation towards the target_heading.
 */
__FLAME_GPU_FUNC__ float turn_towards(float target_heading, float current_heading, float max_turn)
{
	return turn_at_most(subtract_headings(target_heading, current_heading), current_heading, max_turn);
}


/**
 * Setup FLAMEGPU Init function.
 * Operations performed:
 *	1. Creation of agents in order to meet the population value
 *	2. Reinitialization of the max_turn costants in radians from their previous values in degrees
 */
__FLAME_GPU_INIT_FUNC__ void setup()
{
	/* Random number seed initialization */
	srand(0);

	/* Missing agents population */
	int p = *get_population() - get_agent_turtle_default_count();
	float default_bounds = *get_bounds();
	if (p > 0)
	{
		printf("Creating %d additional agents...\n", p);

		/* Agents creation with random heading and position */
		xmachine_memory_turtle** turtle_AoS = h_allocate_agent_turtle_array(p);
		for (int i = 0; i < p; i++)
		{
			turtle_AoS[i]->heading = (float)rand() / (float)(RAND_MAX / PI2_F);
			turtle_AoS[i]->x = fmodf(rand(), default_bounds * 10) / 10.0;
			turtle_AoS[i]->y = fmodf(rand(), default_bounds * 10) / 10.0;
		}
		h_add_agents_turtle_default(turtle_AoS, p);
		h_free_agent_turtle_array(&turtle_AoS, p);
	}

	/* Degrees constants conversion to radians */
	float m_align, m_cohere, m_sep, m_fov;
	m_align = degreesToRadians(*get_max_align_turn());
	m_cohere = degreesToRadians(*get_max_cohere_turn());
	m_sep = degreesToRadians(*get_max_separate_turn());
	m_fov = degreesToRadians(*get_FOV());
	set_max_align_turn(&m_align);
	set_max_cohere_turn(&m_cohere);
	set_max_separate_turn(&m_sep);
	set_FOV(&m_fov);
#ifdef FLOCKING_VERBOSE
	printf("Conversion to Radians:\nAlign turn: %.4f rad\nCohere turn: %.4f rad\nSeparate turn: %.4f rad\nFOV: %.4f rad\n", m_align, m_cohere, m_sep, m_fov);
#endif // FLOCKING_VERBOSE
}

/**
 * Maps a coordinate to its respective value inside bounds, with offset as defined in the macro.
 */
__FLAME_GPU_FUNC__ float move_map(float coordinate)
{
	float temp = fmodf(coordinate, bounds);
	return temp < 0 ? temp + bounds : temp;
}

/**
 * Step function that moves the agent towards its heading direction and outputs the new position.
 */
__FLAME_GPU_FUNC__ int move(xmachine_memory_turtle* agent, xmachine_message_position_list* position_messages)
{
	// Calculate dx and dy
	float dx = cosf(agent->heading) * speed;
	float dy = sinf(agent->heading) * speed;
	// Move
	agent->x = move_map(dx + agent->x);
	agent->y = move_map(dy + agent->y);
	// Output position message
	add_position_message(position_messages, agent->x, agent->y, 0, dx, dy, agent->heading);
	return 0;
}

/**
 * Calculates the difference between two coordinates inside a toroidal space.
 */
__FLAME_GPU_FUNC__ float toroidalDifference(float a, float b)
{
	float d = fabs(a - b);
	if (d > bounds / 2.0)
	{
		d = bounds - d;
	}
	return a - b > 0 ? d : -d;
}

/**
 * Implementation of the squared distance between two points in a two dimensional torus.
 */
__FLAME_GPU_FUNC__ float toroidalDistance(float x1, float y1, float x2, float y2)
{
	float dx = toroidalDifference(x2, x1);
	float dy = toroidalDifference(y2, y1);
	return sqrt(dx * dx + dy * dy);
}

__FLAME_GPU_FUNC__ bool agent_equals(xmachine_memory_turtle* agent, xmachine_message_position* position)
{
	// TODO: should set and id for this
	return agent->x == position->x &&  agent->y == position->y;
}

/**
 * Returns true if the towards_angle is in the range [-FOV/2, FOV/2] centered at the current heading.
 */
__FLAME_GPU_FUNC__ bool in_FOV(float towards_angle, float current_heading)
{
	return fabs(subtract_headings(towards_angle, current_heading)) <= FOV / 2.0f;
}

/**
 * Flock FLAMEGPU Agent Function
 * Implements the behaviour of a turtle in the system.
 * 
 * @param agent Pointer to an agent structure of type xmachine_memory_turtle. This represents a single agent instance and can be modified directly.
 * @param position_messages  position_messages Pointer to input message list of type xmachine_message__list.
 *			Must be passed as an argument to the get_first_position_message and get_next_position_message functions.
 * @param partition_matrix Pointer to the partition matrix of type xmachine_message_position_PBM. 
 *			Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int flock(xmachine_memory_turtle* agent, xmachine_message_position_list* position_messages, xmachine_message_position_PBM* partition_matrix)
{
	/* 1. Finding the nearest neighbor heading and distance */
	float nearest_squared_distance = FLT_MAX;
	float nearest_heading = 0;
	/* 2. Average neighborhood heading evaluation */
	float avg_heading = agent->heading;
	float dx_sum = 0, dy_sum = 0;
	/* 3. Average heading to neighborhood evaluation */
	float sin_var = 0, cos_var = 0;
	/* 4. Neighbor count */
	int count = 0;

    xmachine_message_position* current_message = get_first_position_message(position_messages, partition_matrix, agent->x, agent->y, 0);
    while (current_message)
    {
		float d = toroidalDistance(agent->x, agent->y, current_message->x, current_message->y);
		float towards = calculateHeading(toroidalDifference(current_message->x, agent->x), toroidalDifference(current_message->y, agent->y));
		if (!agent_equals(agent, current_message) && d <= vision && in_FOV(towards, agent->heading))
		{
			/* 1. */
			if (d < nearest_squared_distance)
			{
				nearest_squared_distance = d;
				nearest_heading = towards;
			}

			/* 2. */
			dx_sum += current_message->dx;
			dy_sum += current_message->dy;

			/* 3. */
			sin_var += sinf(towards);
			cos_var += cosf(towards);

			/* 4. */
			count++;
		}
        current_message = get_next_position_message(current_message, position_messages, partition_matrix);
    }

	/* Minimum distance check */
	if (nearest_squared_distance < minimum_separation)
	{
		// Separation
#ifdef FLOCKING_VERBOSE
		printf("Separating from %f to %f to avoid %f, calculated from nearest neighbor\n", agent->heading, turn_away(nearest_heading, agent->heading, max_separate_turn), nearest_heading);
#endif // FLOCKING_VERBOSE
		agent->heading = turn_away(nearest_heading, agent->heading, max_separate_turn);
		agent->colour = FLAME_GPU_VISUALISATION_COLOUR_BLUE;
	}
	/* Check if at least one neighbor is present */
	else if (count > 0)
	{
		// Align
		agent->colour = FLAME_GPU_VISUALISATION_COLOUR_YELLOW;
		avg_heading = calculateHeading(dx_sum, dy_sum);
#ifdef FLOCKING_VERBOSE
		printf("Align from %f to %f to match %f, calculated from %d neighbors (dys: %f, dxs: %f)\n", agent->heading, turn_towards(avg_heading, agent->heading, max_align_turn), avg_heading, count, dy_sum, dx_sum);
#endif // FLOCKING_VERBOSE
		agent->heading = turn_towards(avg_heading, agent->heading, max_align_turn);
		
		// Cohere
		/* Average direction towards every neighbor evaluation */
		sin_var /= count;
		cos_var /= count;
		avg_heading = calculateHeading(cos_var, sin_var);
#ifdef FLOCKING_VERBOSE
		printf("Cohere from %f to %f to match %f, calculated from %d neighbors (sin: %f, cos: %f)\n", agent->heading, turn_towards(avg_heading, agent->heading, max_cohere_turn), avg_heading, count, sin_var, cos_var);
#endif // FLOCKING_VERBOSE
		agent->heading = turn_towards(avg_heading, agent->heading, max_cohere_turn);
	}
	else
	{
		/* Agent is alone */
		agent->colour = FLAME_GPU_VISUALISATION_COLOUR_BLACK;
	}
    return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
