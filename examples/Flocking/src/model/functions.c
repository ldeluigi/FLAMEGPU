
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
#define degreesToRadians(angleDegrees) ((angleDegrees) * PI_F / 180.0)

/**
 * Returns a new heading which has performed a rotation of at most max_turn from the current_heading.
 */
__FLAME_GPU_FUNC__ float turn_at_most(float turn, float current_heading, float max_turn)
{
	float temp = fmodf(abs(turn) > max_turn ? (turn > 0 ? current_heading + max_turn : current_heading - max_turn) : current_heading + turn, 2 * PI_F);
	if (temp < 0) temp += 2 * PI_F;
	return temp;
}

/**
 * Returns a new heading which has performed a rotation away from the target_heading.
 */
__FLAME_GPU_FUNC__ float turn_away(float target_heading, float current_heading, float max_turn)
{
	return turn_at_most(current_heading - target_heading, current_heading, max_turn);
}

/**
 * Returns a new heading which has performed a rotation towards the target_heading.
 */
__FLAME_GPU_FUNC__ float turn_towards(float target_heading, float current_heading, float max_turn)
{
	return turn_at_most(target_heading - current_heading, current_heading, max_turn);
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
			turtle_AoS[i]->heading = (float)rand() / (float)(RAND_MAX / PI_F);
			turtle_AoS[i]->x = (fmodf(rand(), default_bounds * 10)) / 10.0;
			turtle_AoS[i]->y = (fmodf(rand(), default_bounds * 10)) / 10.0;
		}
		h_add_agents_turtle_default(turtle_AoS, p);
		h_free_agent_turtle_array(&turtle_AoS, p);
	}

	/* Degrees constants conversion to radians */
	float m_align, m_cohere, m_sep;
	m_align = degreesToRadians(*get_max_align_turn());
	m_cohere = degreesToRadians(*get_max_cohere_turn());
	m_sep = degreesToRadians(*get_max_separate_turn());
	set_max_align_turn(&m_align);
	set_max_cohere_turn(&m_cohere);
	set_max_separate_turn(&m_sep);
}

/**
 * Maps a coordinate to its respective value inside bounds, with offset as defined in the macro.
 */
__FLAME_GPU_FUNC__ float move_map(float coordinate)
{
	float temp = fmodf(coordinate, bounds);
	if (temp < 0) temp += bounds;
	return temp;
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
 * Implementation of the squared distance between two points in a two dimensional torus.
 */
__FLAME_GPU_FUNC__ float toroidalSquaredDistance(float x1, float y1, float x2, float y2)
{
	float dx = fabs(x2 - x1);
	float dy = fabs(y2 - y1);
	float halfbounds = bounds / 2.0;
	if (dx > halfbounds)
	{
		dx = bounds - dx;
	}
	if (dy > halfbounds)
	{
		dy = bounds - dy;
	}
	return dx * dx + dy * dy;
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
	/* Finding the nearest neighbor heading and distance */
	float nearest_squared_distance = FLT_MAX;
	float nearest_heading = 0;

    xmachine_message_position* current_message = get_first_position_message(position_messages, partition_matrix, agent->x, agent->y, 0);
    while (current_message)
    {
		float d = toroidalSquaredDistance(current_message->x, current_message->y, agent->x, agent->y);
		if (d < nearest_squared_distance && d > 0 && d <= vision)
		{
			nearest_squared_distance = d;
			nearest_heading = current_message->heading;
		}
        current_message = get_next_position_message(current_message, position_messages, partition_matrix);
    }

	/* Minimal distance check */
	if (nearest_squared_distance < minimum_separation)
	{
		// Separation
		agent->heading = turn_away(nearest_heading, agent->heading, max_separate_turn);
		agent->colour = FLAME_GPU_VISUALISATION_COLOUR_BLUE;
	}
	/* Check if at least a neighbor is present */
	else if (nearest_squared_distance < FLT_MAX)
	{
		// Align
		agent->colour = FLAME_GPU_VISUALISATION_COLOUR_GREEN;
		/* Average neighborhood heading evaluation */
		float avg_heading = agent->heading;
		float dx_sum = 0, dy_sum = 0;
		float sin_var = 0, cos_var = 0;
		int count = 0;
		xmachine_message_position* current_message = get_first_position_message(position_messages, partition_matrix, agent->x, agent->y, 0);
		while (current_message)
		{
			dx_sum += current_message->dx;
			dy_sum += current_message->dy;
			float towards_heading = atan2f(current_message->y - agent->y, current_message->x - agent->x);
			sin_var += sinf(towards_heading);
			cos_var += cosf(towards_heading);
			count++;
			current_message = get_next_position_message(current_message, position_messages, partition_matrix);
		}
		if (count != 0)
		{
			avg_heading = atan2f(dy_sum, dx_sum);
		}
		agent->heading = turn_towards(avg_heading, agent->heading, max_align_turn);


		// Cohere
		if (count != 0)
		{
			/* Average direction towards every neighbor evaluation */
			sin_var /= count;
			cos_var /= count;
			avg_heading = atan2f(sin_var, cos_var);
		}
		else
		{
			avg_heading = agent->heading;
		}
		agent->heading = turn_towards(avg_heading, agent->heading, max_cohere_turn);
	}
	else
	{
		agent->colour = FLAME_GPU_VISUALISATION_COLOUR_BLACK;
	}
    return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
