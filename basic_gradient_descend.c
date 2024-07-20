#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#define NUM_PARAMS 1

#define MAX_ITER 1000
#define ALPHA 0.1
#define BETA 0.1
#define LIMIT_STEP 1

typedef struct params
{
    float array[NUM_PARAMS];
} Params;

typedef struct data
{   // the data needs to be dynamically allocated
    float *data;
} Data;

void plot_gradients(float* gradients)
{
    char * commandsForGnuplot[] = {"set title \"Gradient plot through iterations\"", "plot 'data.temp'"};
    FILE * temp = fopen("data.temp", "w");
    
    FILE * gnuplotPipe = popen ("gnuplot -persistent", "w");

    int i;
    for (i = 0; i < MAX_ITER; i++)
    {
        fprintf(temp, "%d %f \n", i, gradients[i]); 
    }

    for (i = 0; i < MAX_ITER; i++)
    {
        fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i]); 
    }
}

float clamp(float x, float min, float max)
{
    if (x > max)
        return max;
    if (x < min)
        return min;
    return x;
}

Params* copy_params(Params* original)
{
    Params* copy = (Params*)malloc(sizeof(Params));
    for (int i = 0; i < NUM_PARAMS; i++)
    {
        copy->array[i] = original->array[i];
    }
    return copy;
}

Data* import_data(char* filename)
{
    Data* data = (Data*)malloc(sizeof(Data));
    // importing the data on which the params are being optimized for
    return data;
}

float loss_function(Params* curr_params, Data* data)
{   // the function that is optimized
    return 1;
}

float loss_function_gradient(Params* curr_params, Data* data)
{   // the gradient of the function that needs to be optimized
    return 1;
}

Params* generate_random_params()
{   // generating a random state of parameters, this is needed for the initial state of the optimization
    Params* nasumicna = (Params*)malloc(sizeof(Params));
    return nasumicna; 
}

void update_params(Params* curr_params, Data* data)
{
    for (int i = 0; i < NUM_PARAMS; i++)
    {
        curr_params->array[i] -= clamp(ALPHA * loss_function_gradient(curr_params, data), -LIMIT_STEP, LIMIT_STEP);
    }
}

Params* gradient_descent()
{   // main function
    float gradients[MAX_ITER];
    Params* params[MAX_ITER];

    Params* curr_params = generate_random_params();
    Data* data = import_data("filename");    // change the file name to import data

    for (int iter = 0; iter < MAX_ITER; iter++)
    {
        gradients[iter] = loss_function(curr_params, data);
        params[iter] = copy_params(curr_params);

        update_params(curr_params, data);
    }

    plot_gradients(gradients);

    return curr_params;
}

int main()
{
    // Params** params = gradient_descent();
    return 0;
}