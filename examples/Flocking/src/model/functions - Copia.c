
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
#define BOUND 20
#define OFFSET -10
#define degreesToRadians(angleDegrees) ((angleDegrees) * PI_F / 180.0)

__FLAME_GPU_FUNC__ float square(float x)
{
	return x * x;
}

__FLAME_GPU_FUNC__ float turn_at_most(float turn, float current_heading, float max_turn)
{
	return abs(turn) > max_turn ? (turn > 0 ? current_heading + max_turn : current_heading - max_turn) : current_heading + turn;
}

__FLAME_GPU_FUNC__ float turn_away(float target_heading, float current_heading, float max_turn)
{
	return turn_at_most(current_heading - target_heading, current_heading, max_turn);
}

__FLAME_GPU_FUNC__ float turn_towards(float target_heading, float current_heading, float max_turn)
{
	return turn_at_most(target_heading - current_heading, current_heading, max_turn);
}


/**
 * setup FLAMEGPU Init function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_INIT_FUNC__ void setup()
{

}

__FLAME_GPU_FUNC__ int output_position(xmachine_memory_turtle* agent, xmachine_message_position_list* position_messages)
{
	return 0;
}

/**
 * move FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_turtle. This represents a single agent instance and can be modified directly.
 * @param position_messages  position_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_position_message and get_next_position_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_position_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int move(xmachine_memory_turtle* agent, xmachine_message_position_list* position_messages)
{
	float nearest_squared_distance = FLT_MAX;
	float nearest_heading = 0;
    
    // Find nearest neighbor
    xmachine_message_position* current_message = get_first_position_message(position_messages);
    while (current_message)
    {
		float d = square(current_message->x - agent->x) + square(current_message->y - agent->y);
		// Non è toroidale così
		if (d < nearest_squared_distance)
		{
			nearest_squared_distance = d;
			nearest_heading = current_message->heading;
		}
        current_message = get_next_position_message(current_message, position_messages);
    }

	if (nearest_squared_distance < minimum_separation)
	{
		// Separation
		agent->heading = turn_away(nearest_heading, agent->heading, degreesToRadians(max_separate_turn));
		agent->colour = FLAME_GPU_VISUALISATION_COLOUR_BLUE;
	}
	else if (nearest_squared_distance < FLT_MAX)
	{
		// Align
		agent->colour = FLAME_GPU_VISUALISATION_COLOUR_GREEN;
		float avg_heading = agent->heading;
		float dx_sum = 0, dy_sum = 0;
		float sin_var = 0, cos_var = 0;
		int count = 0;
		xmachine_message_position* current_message = get_first_position_message(position_messages);
		while (current_message)
		{
			dx_sum += current_message->dx;
			dy_sum += current_message->dy;
			float towards_heading = atan2f(current_message->y - agent->y, current_message->x - agent->x);
			sin_var += sinf(towards_heading);
			cos_var += cosf(towards_heading);
			count++;
			current_message = get_next_position_message(current_message, position_messages);
		}
		if (count != 0)
		{
			avg_heading = atan2f(dy_sum, dx_sum);
		}
		agent->heading = turn_towards(avg_heading, agent->heading, degreesToRadians(max_align_turn));


		// Cohere
		if (count != 0)
		{
			sin_var /= count;
			cos_var /= count;
			avg_heading = atan2f(sin_var, cos_var);
		}
		else
		{
			avg_heading = agent->heading;
		}
		agent->heading = turn_towards(avg_heading, agent->heading, degreesToRadians(max_cohere_turn));
	}
	else
	{
		agent->colour = FLAME_GPU_VISUALISATION_COLOUR_YELLOW;
	}

	// Calculate dx and dy
	agent->dx = cosf(agent->heading) * speed;
	agent->dy = sinf(agent->heading) * speed;
	// Move
	agent->x = fmodf(fmodf(agent->dx + agent->x, BOUND) + BOUND, BOUND) + OFFSET;
	agent->y = fmodf(fmodf(agent->dy + agent->y, BOUND) + BOUND, BOUND) + OFFSET;
    return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
