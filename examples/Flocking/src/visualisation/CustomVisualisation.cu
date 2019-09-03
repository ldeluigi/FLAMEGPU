#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include "header.h"
#include "CustomVisualisation.h"

// cuda keywords
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#define FOVY 45.0

// Buffer Objects ids
GLuint coneIndices;
GLuint coneVerts;
GLuint coneNormals;


// Simulation buffers/textures
cudaGraphicsResource_t turtle_default_cgr;
GLuint turtle_default_tbo;
GLuint turtle_default_displacementTex;


// Mouse Controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -VIEW_DISTANCE;

// Keyboard Controls
unsigned int simulation_speed = 1;
const float camera_speed = 0.2f;
float translate_y = 0, translate_x = 0;
glm::fvec3 pos(0, 0, 0);
enum Direction { UP, DOWN, LEFT, RIGHT};
#if defined(PAUSE_ON_START)
bool paused = true;
#else
bool paused = false;
#endif

// Vertex Shader
GLuint vertexShader;
GLuint fragmentShader;
GLuint shaderProgram;
GLuint vs_displacementMap;
GLuint vs_mapIndex;



// Timer
cudaEvent_t start, stop;
const int display_rate = 50;
int frame_count;
float frame_time = 0.0;

#ifdef SIMULATION_DELAY
	int delay_count = 0;
#endif

/**
 * Prototypes
 */
int initGL();
void initShader();
void createVBO(GLuint* vbo, GLuint size);
void deleteVBO(GLuint* vbo);
void createTBO(cudaGraphicsResource_t* cudaResource, GLuint* tbo, GLuint* tex, GLuint size);
void deleteTBO(cudaGraphicsResource_t* cudaResource, GLuint* tbo);
void setVertexBufferData();
void reshape(int width, int height);
void display();
void close();
void keyboard(unsigned char key, int x, int y);
void special(int key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void runCuda();
void checkGLError();
void createVIBO(GLuint* vbo, GLuint size);
void moveCamera(Direction d);


/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort = true)
{
	gpuAssert(cudaPeekAtLastError(), file, line);
#ifdef _DEBUG
	gpuAssert(cudaDeviceSynchronize(), file, line);
#endif

}

/**
 * Shaders sources.
 */
const char vertexShaderSource[] = R"(
	#extension GL_EXT_gpu_shader4 : enable
	uniform samplerBuffer displacementMap;
	attribute in float mapIndex;
	varying vec3 normal, lightDir;
	varying vec4 colour;
	void main()
	{
		vec4 position = gl_Vertex;
		vec4 lookup = texelFetchBuffer(displacementMap, (int)mapIndex);
		if (lookup.w > 7.5)
			colour = vec4(0.518, 0.353, 0.02, 0.0);
		else if (lookup.w > 6.5)
			colour = vec4(1.0, 1.0, 1.0, 0.0);
		else if (lookup.w > 5.5)
			colour = vec4(1.0, 0.0, 1.0, 0.0);
		else if (lookup.w > 4.5)
			colour = vec4(0.0, 1.0, 1.0, 0.0);
		else if (lookup.w > 3.5)
			colour = vec4(1.0, 1.0, 0.0, 0.0);
		else if (lookup.w > 2.5)
			colour = vec4(0.0, 0.0, 1.0, 0.0);
		else if (lookup.w > 1.5)
			colour = vec4(0.0, 1.0, 0.0, 0.0);
		else if (lookup.w > 0.5)
			colour = vec4(1.0, 0.0, 0.0, 0.0);
		else
			colour = vec4(0.0, 0.0, 0.0, 0.0);
		
		lookup.w = 1.0;
	    float xtemp = position.x * cos(lookup.z) - position.y * sin(lookup.z);
        position.y = position.x * sin(lookup.z) + position.y * cos(lookup.z);
	    position.x = xtemp;
	    lookup.z = 0;
	 	position += lookup;
	    gl_Position = gl_ModelViewProjectionMatrix * position;
	 	
	 	vec3 mvVertex = vec3(gl_ModelViewMatrix * position);
	 	lightDir = vec3(gl_LightSource[0].position.xyz - mvVertex);
	 	normal = gl_NormalMatrix * gl_Normal;
	 })"
;

const char fragmentShaderSource[] = R"(

	varying vec3 normal, lightDir;
	varying vec4 colour;
	void main (void)
	{
		// Defining The Material Colors
		vec4 AmbientColor = vec4(0.25, 0.0, 0.0, 1.0);
		vec4 DiffuseColor = colour;
		
		// Scaling The Input Vector To Length 1
		vec3 n_normal = normalize(normal);
		vec3 n_lightDir = normalize(lightDir);
		
		// Calculating The Diffuse Term And Clamping It To [0;1]
		float DiffuseTerm = clamp(dot(n_normal, n_lightDir), 0.0, 1.0);
		
		// Calculating The Final Color
		gl_FragColor = AmbientColor + DiffuseColor * DiffuseTerm;
		
	})"
;

// GPU Kernel
__global__ void output_turtle_agent_to_VBO(xmachine_memory_turtle_list* agents, glm::vec4* vbo, glm::vec3 centralise) {

	//global thread index
	const int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	vbo[index].x = agents->x[index] - centralise.x;
	vbo[index].y = agents->y[index] - centralise.y;
	vbo[index].z = agents->heading[index];
	vbo[index].w = agents->colour[index];
}

/**
 * Hook function to the simulation.
 */
void initVisualisation()
{
	// Create GL context
	int   argc = 1;
	char glutString[] = "GLUT application";
	char *argv[] = { glutString, NULL };
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow("Flocking - FLAME GPU Visualiser");

	// GL Initialization
	if (!initGL()) {
		return;
	}
	initShader();

	// Callbacks registration
	glutReshapeFunc(reshape);
	glutDisplayFunc(display);
	glutCloseFunc(close);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	// Set the closing behaviour 
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);


	// Vertex Buffers Object creation
	createVBO(&coneVerts, (CONE_SLICES + 2) * sizeof(glm::vec3));
	createVBO(&coneNormals, (CONE_SLICES + 2) * sizeof(glm::vec3));
	createVIBO(&coneIndices, (3 * 2 * CONE_SLICES) * sizeof(unsigned int));

	setVertexBufferData();

	// Texture Buffer Object creation
	createTBO(&turtle_default_cgr, &turtle_default_tbo, &turtle_default_displacementTex, xmachine_memory_turtle_MAX * sizeof(glm::vec4));


	//set shader uniforms
	glUseProgram(shaderProgram);

	//create a events for timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	puts("\nHotkeys:\n\tEsc, Q: quit.\n\tSpace: pause/play.\n\t.: perform a single iteration.\n\t"
		"1...9: set simulation speed.\n\tArrows: change observer position.\n\t"
		"Left mouse button + mouse movement: change observer perspective.\n\t"
		"Right mouse button + mouse movement: change observer height.\n");
}

/**
 * Hook function to the simulation.
 */
void runVisualisation() {
	// Flush outputs prior to simulation loop.
	fflush(stdout);
	fflush(stderr);
	// start rendering mainloop
	glutMainLoop();
}

/**
 * Run the Cuda part of the computation
 */
void runCuda()
{
	if (!paused) {
#ifdef SIMULATION_DELAY
		delay_count++;
		if (delay_count == SIMULATION_DELAY) {
			delay_count = 0;
			for (unsigned int i = 0; i < simulation_speed; i++)
			{
				singleIteration();
			}
		}
#else
		for (unsigned int i = 0; i < simulation_speed; i++)
		{
			singleIteration();
		}
#endif
	}

	// Kernel params
	const int threads_per_tile = 256;
	int tile_size;
	dim3 grid;
	dim3 threads;
	glm::vec3 centralise;

	// Pointer
	glm::vec4 *dptr;

	if (get_agent_turtle_default_count() > 0)
	{
		// Mapping OpenGL buffer object for writing from CUDA
		size_t accessibleBufferSize = 0;
		gpuErrchk(cudaGraphicsMapResources(1, &turtle_default_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &accessibleBufferSize, turtle_default_cgr));

		// Cuda block size
		tile_size = (int)ceil((float)get_agent_turtle_default_count() / threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);

		// Variables for a continuous environment
		centralise = getMaximumBounds() + getMinimumBounds();
		centralise /= 2;

		// Kernel call
		output_turtle_agent_to_VBO << < grid, threads >> > (get_device_turtle_default_agents(), dptr, centralise);
		gpuErrchkLaunch();

		// Buffer Object Unmap
		gpuErrchk(cudaGraphicsUnmapResources(1, &turtle_default_cgr));
	}

}

/**
 * GL Initialization.
 */
int initGL()
{
	// Necessary OpenGL extensions initialization
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 "
		"GL_ARB_pixel_buffer_object")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.\n");
		fflush(stderr);
		return 1;
	}

	// Default initialization
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glEnable(GL_DEPTH_TEST);

	// Window reshape
	reshape(WINDOW_WIDTH, WINDOW_HEIGHT);
	checkGLError();

	// Lighting
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	return 1;
}

/**
 * GL Vertex Shader compilation and initialization.
 */
void initShader()
{
	const char* v = vertexShaderSource;
	const char* f = fragmentShaderSource;

	// Vertex Shader
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &v, 0);
	glCompileShader(vertexShader);

	// Fragment Shader
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &f, 0);
	glCompileShader(fragmentShader);

	// Program
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	// Error check
	GLint status;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		printf("ERROR: Shader Compilation Error\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(vertexShader, 262144, &len, data);
		printf("%s", data);
	}
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		printf("ERROR: Shader Compilation Error\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(fragmentShader, 262144, &len, data);
		printf("%s", data);
	}
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		printf("ERROR: Shader Program Link Error\n");
	}

	// Get shader variables
	vs_displacementMap = glGetUniformLocation(shaderProgram, "displacementMap");
	vs_mapIndex = glGetAttribLocation(shaderProgram, "mapIndex");
}


/**
 * Creates an Index Buffer for Vertices.
 */
void createVIBO(GLuint* vbo, GLuint size)
{
	// Index Buffer creation
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *vbo);

	// Index Buffer initialization
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, NULL, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	checkGLError();
}


/**
 * Creates a Vertex Buffer Object.
 */
void createVBO(GLuint* vbo, GLuint size)
{
	// Vertex Buffer creation
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// Vertex Buffer initialization
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	checkGLError();
}

/**
 * Deletes a Vertex or an Index Buffer.
 */
void deleteVBO(GLuint* vbo)
{
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

/**
 * Creates a Texture Buffer object.
 */
void createTBO(cudaGraphicsResource_t* cudaResource, GLuint* tbo, GLuint* tex, GLuint size)
{
	// Texture Buffer Creation
	glGenBuffers(1, tbo);
	glBindBuffer(GL_TEXTURE_BUFFER_EXT, *tbo);

	// Texture Buffer Initialization
	glBufferData(GL_TEXTURE_BUFFER_EXT, size, 0, GL_DYNAMIC_DRAW);

	// Textures
	glGenTextures(1, tex);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, *tex);
	glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA32F_ARB, *tbo);
	glBindBuffer(GL_TEXTURE_BUFFER_EXT, 0);

	// Buffer object registration with CUDA
	gpuErrchk(cudaGraphicsGLRegisterBuffer(cudaResource, *tbo, cudaGraphicsMapFlagsWriteDiscard));

	checkGLError();
}

/**
 * Deletes a Texture Buffer object.
 */
void deleteTBO(cudaGraphicsResource_t* cudaResource, GLuint* tbo)
{
	gpuErrchk(cudaGraphicsUnregisterResource(*cudaResource));
	*cudaResource = 0;

	glBindBuffer(1, *tbo);
	glDeleteBuffers(1, tbo);

	*tbo = 0;
}


/**
 * Sets the cone vertices.
 */
static void setConeVertex(glm::vec3* data, int slice) {
	if (slice == CONE_SLICES + 1)
	{
		data->x = CONE_HEIGHT;
		data->y = 0;
		data->z = 0;
	}
	else if (slice == 0)
	{
		data->x = 0;
		data->y = 0;
		data->z = 0;
	}
	else
	{
		const float s = slice - 1.0f;
		const float PI = 3.14159265358f;

		const float theta = 2 * PI*s / CONE_SLICES;

		data->x = 0;
		data->y = sin(theta) * CONE_RADIUS;
		data->z = cos(theta) * CONE_RADIUS;
	}
}

/**
 * Sets the cone normals.
 */
static void setConeNormal(glm::vec3* data, int slice) {
	if (slice == CONE_SLICES + 1)
	{
		data->x = 1;
		data->y = 0;
		data->z = 0;
	}
	else if (slice == 0)
	{
		data->x = -1;
		data->y = 0;
		data->z = 0;
	}
	else
	{
		const float angle = atanf(CONE_RADIUS / CONE_HEIGHT);
		const float s = slice - 1.0f;
		const float PI = 3.14159265358f;

		const float theta = 2 * PI * s / CONE_SLICES;

		data->x = sin(angle);
		data->y = cos(angle) * sin(theta);
		data->z = cos(angle) * cos(theta);
	}
}

/**
 * Sets the cone vertex indices for the triangles draw primitive.
 */
static void setConeVertexIndex(unsigned int* data, int count) {
	const int n = count / 6;
	const int i = count % 6;
	if (i == 0) *data = 0;
	else if (i == 5) *data = CONE_SLICES + 1;
	else
	{
		if (i < 3)
		{
			const unsigned int index = n + i;
			*data = (index == CONE_SLICES + 1 ? 1 : index);
		}
		else
		{
			const unsigned int index = n + i - 2;
			*data = (index == CONE_SLICES + 1 ? 1 : index);
		}
	}
}





/**
 * Sets the VBO data.
 */
void setVertexBufferData()
{

	int slice;
	int i;

	// vertex points data upload
	glBindBuffer(GL_ARRAY_BUFFER, coneVerts);
	glm::vec3* verts = (glm::vec3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	i = 0;
	for (slice = 0; slice <= CONE_SLICES + 1; slice++) {
		setConeVertex(&verts[i++], slice);
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);

	// vertex normal data upload
	glBindBuffer(GL_ARRAY_BUFFER, coneNormals);
	glm::vec3* normals = (glm::vec3*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	i = 0;
	for (slice = 0; slice <= CONE_SLICES + 1; slice++) {
		setConeNormal(&normals[i++], slice);
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);

	// vertex points data upload
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coneIndices);
	unsigned int* indices = (unsigned int*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);
	for (i = 0; i < CONE_SLICES * 2 * 3; i++) {
		setConeVertexIndex(&indices[i], i);
	}
	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
}




/**
 * Window reshape function.
 */
void reshape(int width, int height) {
	// viewport
	glViewport(0, 0, width, height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(FOVY, (GLfloat)width / (GLfloat)height, NEAR_CLIP, FAR_CLIP);

	checkGLError();
}


/**
 * Displays the model.
 */
void display()
{
	float millis;

	// CUDA start Timing
	cudaEventRecord(start);

	// run CUDA kernel to generate vertex positions
	runCuda();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();


	// Camera movement
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 0.0, 1.0);
	// move
	glTranslatef(translate_x, translate_y, translate_z);
	pos = glm::fvec3(translate_x, translate_y, translate_z);



	// Set light position
	glLightfv(GL_LIGHT0, GL_POSITION, LIGHT_POSITION);


	// Draw turtle Agents in default state
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, turtle_default_displacementTex);
	// loop
	for (int i = 0; i < get_agent_turtle_default_count(); i++) {
		glVertexAttrib1f(vs_mapIndex, (float)i);

		// draw using vertex and attribute data on the gpu (fast)
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);
		glEnableClientState(GL_INDEX_ARRAY);
		glBindBuffer(GL_ARRAY_BUFFER, coneVerts);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, coneNormals);
		glNormalPointer(GL_FLOAT, 0, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coneIndices);
		glDrawElements(GL_TRIANGLES, 3 * 2 * CONE_SLICES, GL_UNSIGNED_INT, NULL);


		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_INDEX_ARRAY);
	}


	// CUDA stop timing
	cudaEventRecord(stop);
	glFlush();
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millis, start, stop);
	frame_time += millis;

	if (frame_count == display_rate) {
		char title[100];
		sprintf(title, "Flocking - Execution & Rendering Total: %f (FPS), %f milliseconds per frame", display_rate / (frame_time / 1000.0f), frame_time / display_rate);
		glutSetWindowTitle(title);

		// reset
		frame_count = 0;
		frame_time = 0.0;
	}
	else {
		frame_count++;
	}


	glutSwapBuffers();
	glutPostRedisplay();

	// If an early exit has been requested, close the visualisation by leaving the main loop.
	if (get_exit_early()) {
		glutLeaveMainLoop();
	}
}

/**
 * Closes the window.
 */
void close()
{
	// Cleanup visualisation memory

	deleteVBO(&coneVerts);
	deleteVBO(&coneNormals);
	deleteVBO(&coneIndices);

	deleteTBO(&turtle_default_cgr, &turtle_default_tbo);

	// Destroy cuda events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	// Call exit functions and clean up simulation memory
	cleanup();
}


/**
 * Keyboard event handler.
 */
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key) {
		// Space == 32
	case (32):
		paused = !paused;
		break;
	case '.':
		singleIteration();
		fflush(stdout);
		break;
	case '1':
	case '2':
	case '3':
	case '4':
	case '5':
	case '6':
	case '7':
	case '8':
	case '9':
		simulation_speed = key - '0';
		break;
		// Esc == 27
	case 27:
	case 'q':
		// Set the flag indicating we wish to exit the simulation.
		set_exit_early();
	}
}

/**
 * Keyboard event handler.
 */
void special(int key, int x, int y) {
	switch (key)
	{
	case(GLUT_KEY_RIGHT):
		moveCamera(Direction::RIGHT);
		break;
	case(GLUT_KEY_UP):
		moveCamera(Direction::UP);
		break;
	case(GLUT_KEY_LEFT):
		moveCamera(Direction::LEFT);
		break;
	case(GLUT_KEY_DOWN):
		moveCamera(Direction::DOWN);
		break;
	}
}

void moveCamera(Direction d)
{
	switch (d)
	{
	case Direction::DOWN:
		translate_y += camera_speed;
		break;
	case Direction::UP:
		translate_y -= camera_speed;
		break;
	case Direction::RIGHT:
		translate_x -= camera_speed;
		break;
	case Direction::LEFT:
		translate_x += camera_speed;
		break;
	}
}


/**
 * Mouse event handler.
 */
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) {
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP) {
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
	glutPostRedisplay();
}

void motion(int x, int y)
{
	float dx = (float)x - mouse_old_x;
	float dy = (float)y - mouse_old_y;

	if (mouse_buttons & 1) {
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4) {
		translate_z += dy * VIEW_DISTANCE * 0.001f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

/**
 * OpenGL error check.
 */
void checkGLError() {
	int Error;
	if ((Error = glGetError()) != GL_NO_ERROR)
	{
		const char* Message = (const char*)gluErrorString(Error);
		fprintf(stderr, "OpenGL Error : %s\n", Message);
	}
}
