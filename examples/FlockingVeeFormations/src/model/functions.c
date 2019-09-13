
/*
 * Flocking Vee Formation implemented with Flame GPU, based on the NetLogo library model with the same name.
 * Link at the original FlockingVeeFormation: https://ccl.northwestern.edu/netlogo/models/FlockingVeeFormations
 * 
 */


#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>

#ifdef FLOCKING_PLOT
#include <iostream>
#include <fstream>
std::ofstream outputFile;
#endif // FLOCKING_PLOT

#define PI_F 3.141592654f
#define PI2_F (PI_F * 2)
#define degreesToRadians(angleDegrees) ((angleDegrees) * PI_F / 180.0)

/**
 * Constraints the angle in input to a value in the interval [0, 2 * PI_F] where PI_F is pi in float.
 */
__FLAME_GPU_FUNC__ float constraintRadians(float r)
{
	const float a = fmodf(r, PI2_F);
	return a < 0 ? a + PI2_F : a;
}

/**
 * Calculate the heading represented by the dx and dy distances towards the target,
 * with and angle in the interval [0, 2 * PI_F] where PI_F is pi in float.
 */
__FLAME_GPU_FUNC__ inline float calculateHeading(float dx, float dy)
{
	return constraintRadians(atan2f(dy, dx));
}

/**
 * Rotates the current heading clockwise.
 */
__FLAME_GPU_FUNC__ inline float right_turn(float current_heading, float turn)
{
	return constraintRadians(current_heading - turn);
}

/**
 * Rotates the current heading counterclockwise.
 */
__FLAME_GPU_FUNC__ inline float left_turn(float current_heading, float turn)
{
	return constraintRadians(current_heading + turn);
}

/**
 * Returns the minimum difference between two given angles in radians,
 * with a value inside the interval [-PI_F/2, PI_F/2], where PI_F is pi in float.
 */
__FLAME_GPU_FUNC__ float subtract_headings(float h1, float h2)
{
	const float a = h1 - h2;
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
__FLAME_GPU_FUNC__ inline float turn_away(float target_heading, float current_heading, float max_turn)
{
	return turn_at_most(subtract_headings(current_heading, target_heading), current_heading, max_turn);
}

/**
 * Returns a new heading which has performed a rotation towards the target_heading.
 */
__FLAME_GPU_FUNC__ inline float turn_towards(float target_heading, float current_heading, float max_turn)
{
	return turn_at_most(subtract_headings(target_heading, current_heading), current_heading, max_turn);
}

__FLAME_GPU_STEP_FUNC__ void step()
{
#ifdef FLOCKING_PLOT
	int accelerate, avoid, separate, align, alone;
	accelerate = count_turtle_default_colour_variable(FLAME_GPU_VISUALISATION_COLOUR_CYAN);
	avoid = count_turtle_default_colour_variable(FLAME_GPU_VISUALISATION_COLOUR_RED);
	separate = count_turtle_default_colour_variable(FLAME_GPU_VISUALISATION_COLOUR_GREEN);
	align = count_turtle_default_colour_variable(FLAME_GPU_VISUALISATION_COLOUR_YELLOW);
	alone = count_turtle_default_colour_variable(FLAME_GPU_VISUALISATION_COLOUR_BLACK);

	outputFile << getIterationNumber() << " " << accelerate << " " << avoid << " " << separate << " " << align << " " << alone << std::endl;
#endif // FLOCKING_PLOT
}

__FLAME_GPU_EXIT_FUNC__ void finish()
{
#ifdef FLOCKING_PLOT
	outputFile.close();
#endif // FLOCKING_PLOT;
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

	const float speed = *get_speed();

	/* Missing agents population */
	const int p = *get_population() - get_agent_turtle_default_count();
	const float default_bounds = *get_bounds();
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
			turtle_AoS[i]->speed = speed;
		}
		h_add_agents_turtle_default(turtle_AoS, p);
		h_free_agent_turtle_array(&turtle_AoS, p);
	}

	/* Degrees constants conversion to radians */
	float m_turn, m_fov, m_obs;
	m_turn = degreesToRadians(*get_max_turn());
	m_fov = degreesToRadians(*get_FOV());
	m_obs = degreesToRadians(*get_obstruction_angle());
	set_max_turn(&m_turn);
	set_FOV(&m_fov);
	set_obstruction_angle(&m_obs);
#ifdef FLOCKING_VERBOSE
	printf("Conversion to Radians:\nMax turn: %.4f rad\nFOV: %.4f rad\n", m_turn, m_fov);
#endif // FLOCKING_VERBOSE
#ifdef FLOCKING_PLOT
	outputFile.open("out");
	if (!outputFile.is_open()) set_exit_early();
	else outputFile << "# itno accelerate avoid separate align alone" << std::endl;
#endif // FLOCKING_PLOT
}

/**
 * Maps a coordinate to its respective value inside bounds, with offset as defined in the macro.
 */
__FLAME_GPU_FUNC__ float move_map(float coordinate)
{
	const float temp = fmodf(coordinate, bounds);
	return temp < 0 ? temp + bounds : temp;
}

/**
 * Step function that moves the agent towards its heading direction and outputs the new position.
 */
__FLAME_GPU_FUNC__ int move(xmachine_memory_turtle* agent, xmachine_message_position_list* position_messages)
{
	// Calculate dx and dy
	const float dx = cosf(agent->heading) * agent->speed;
	const float dy = sinf(agent->heading) * agent->speed;
	// Move
	agent->x = move_map(dx + agent->x);
	agent->y = move_map(dy + agent->y);
	// Output position message
	add_position_message(position_messages, agent->x, agent->y, 0, dx, dy, agent->heading, agent->speed);
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
	const float dx = toroidalDifference(x2, x1);
	const float dy = toroidalDifference(y2, y1);
	return sqrt(dx * dx + dy * dy);
}

__FLAME_GPU_FUNC__ inline bool agent_equals(xmachine_memory_turtle* agent, xmachine_message_position* position)
{
	// TODO: should set and id for this
	return agent->x == position->x &&  agent->y == position->y;
}

/**
 * Returns true if the towards_angle is in the range [-FOV/2, FOV/2] centered at the current heading.
 */
__FLAME_GPU_FUNC__ inline bool in_cone(float towards_angle, float current_heading, float fov)
{
	return fabs(subtract_headings(towards_angle, current_heading)) <= fov / 2.0f;
}

/**
 * Returns true if the towards_angle is in the range [-FOV/2, FOV/2] centered at the current heading.
 */
__FLAME_GPU_FUNC__ inline bool in_FOV(float towards_angle, float current_heading)
{
	return in_cone(towards_angle, current_heading, FOV);
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
__FLAME_GPU_FUNC__ int flock(xmachine_memory_turtle* agent, xmachine_message_position_list* position_messages, xmachine_message_position_PBM* partition_matrix, RNG_rand48* rand48)
{
	// Reset base speed
	agent->speed = speed;

	/* 1. Finding the nearest neighbor heading and distance */
	float nearest_distance = FLT_MAX;
	float nearest_heading = 0;
	float nearest_towards = 0;
	float nearest_speed = 0;

	/* 2. Neighbor count */
	int count = 0;

	/* 3. Obstruction check */
	bool obstructed = false;

    xmachine_message_position* current_message = get_first_position_message(position_messages, partition_matrix, agent->x, agent->y, 0);
    while (current_message)
    {
		float d = toroidalDistance(agent->x, agent->y, current_message->x, current_message->y);
		float towards = calculateHeading(toroidalDifference(current_message->x, agent->x), toroidalDifference(current_message->y, agent->y));
		if (!agent_equals(agent, current_message) && d <= vision && in_FOV(towards, agent->heading))
		{
			/* 1. */
			if (d < nearest_distance)
			{
				nearest_distance = d;
				nearest_heading = current_message->heading;
				nearest_speed = current_message->speed;
				nearest_towards = towards;
			}

			/* 2. */
			count++;

			/* 3. */
			if (!obstructed && in_cone(towards, agent->heading, obstruction_angle))
			{
				obstructed = true;
			}
		}
        current_message = get_next_position_message(current_message, position_messages, partition_matrix);
    }

	/* Check if at least one neighbor is present */
	if (count > 0)
	{
		if (nearest_distance > updraft_distance)
		{
			agent->heading = turn_towards(nearest_towards, agent->heading, max_turn);
			agent->speed = speed * (1 + speed_change);
			agent->colour = FLAME_GPU_VISUALISATION_COLOUR_CYAN;
#ifdef FLOCKING_VERBOSE
			printf("[%.2f, %.2f] Turning towards %f and accelerating to gain updraft advantage\n", agent->x, agent->y, nearest_towards);
#endif // FLOCKING_VERBOSE
		}
		else if (obstructed)
		{
			const float random = rnd<CONTINUOUS>(rand48);
			agent->heading = turn_at_most(random * max_turn * 2 - max_turn, agent->heading, max_turn);
			agent->speed = speed * (1 + speed_change);
			agent->colour = FLAME_GPU_VISUALISATION_COLOUR_RED;
#ifdef FLOCKING_VERBOSE
			printf("[%.2f, %.2f] Moving to avoid obstruction\n", agent->x, agent->y);
#endif // FLOCKING_VERBOSE
		}
		else if (nearest_distance < minimum_separation)
		{
			agent->speed = speed * (1 - speed_change);
			agent->colour = FLAME_GPU_VISUALISATION_COLOUR_GREEN;
#ifdef FLOCKING_VERBOSE
			printf("[%.2f, %.2f] Slowing down to mantain minimum separation\n", agent->x, agent->y);
#endif // FLOCKING_VERBOSE
		}
		else
		{
			agent->speed = nearest_speed;
			agent->heading = turn_towards(nearest_heading, agent->heading, max_turn);
			agent->colour = FLAME_GPU_VISUALISATION_COLOUR_YELLOW;
#ifdef FLOCKING_VERBOSE
			printf("[%.2f, %.2f] Imitating neighbor speed and heading\n", agent->x, agent->y);
#endif // FLOCKING_VERBOSE
		}
	}
	else
	{
		/* Agent is alone */
		agent->colour = FLAME_GPU_VISUALISATION_COLOUR_BLACK;
	}
    return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
