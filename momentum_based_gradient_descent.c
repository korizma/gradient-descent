#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#define NUM_PARAMS 2

#define MAX_ITER 50000
#define ALPHA 0.01
#define BETA 0.1
#define LIMIT_STEP 5
#define BATCH_SIZE 100
#define TERMINATION_CRITERIUM 0.01

#define BUFFER_INPUT 5
#define REPEATED_INDICES 1000

typedef struct params
{
    float array[NUM_PARAMS];
} Params;

typedef struct data
{   // the data needs to be dynamically allocated
    float *x;
    float *y;
    int len;
} Data;

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
    // importing the data on which the params are being optimized for
    Data* data = (Data*)malloc(sizeof(Data));

    FILE* file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Error opening file\n");
        exit(1);
    }

    data->x = (float*)malloc(sizeof(float)*BUFFER_INPUT);
    data->y = (float*)malloc(sizeof(float)*BUFFER_INPUT);

    int curr_line = 0;
    int curr_cap = BUFFER_INPUT;
    int nista;
    while (fscanf(file, "%d,%f,%f\n", &nista, &data->x[curr_line], &data->y[curr_line]) != EOF)
    {
        curr_line++;
        if (curr_cap == curr_line)
        {
            curr_cap += BUFFER_INPUT;
            data->x = (float*)realloc(data->x, curr_cap*sizeof(float));
            data->y = (float*)realloc(data->y, curr_cap*sizeof(float));
        }
    }

    fclose(file);

    printf("Data imported:\nx y\n");
    for (int i = 0; i < curr_line; i++)
    {
        printf("%f %f\n", data->x[i], data->y[i]);
    }

    data->len = curr_line;
    return data;
}

int* random_indices(int len)
{
    int* indices = (int*)malloc(sizeof(int)*BATCH_SIZE);
    int* taken = (int*)calloc(len, sizeof(int));
    int repeated = 0;

    for (int i = 0; i < BATCH_SIZE; i++)
    {
        int index = rand() % len;
        while (taken[index] == 1)
        {
            index = rand() % len;

            repeated++;
            if (repeated > REPEATED_INDICES)
            {
                printf("Error: too many repeated indices\n");
                exit(1);
            }
        }

        taken[index] = 1;
        indices[i] = index;
    }
    

    free(taken);

    return indices;
}

float loss_function(Params* curr_params, Data* data)
{   // the function that is optimized
    float loss = 0;
    int *indices = random_indices(data->len);

    for (int j = 0; j < BATCH_SIZE; j++)
    {
        loss += pow(curr_params->array[0] * data->x[indices[j]] + curr_params->array[1]* (data->x[indices[j]] * data->x[indices[j]]) - data->y[indices[j]], 2 );
    }

    return loss;
}

float* loss_function_gradient(Params* curr_params, Data* data)
{   // the gradient of the function that needs to be optimized
    float* gradients = (float*)malloc(sizeof(float)*NUM_PARAMS);

    float dx = 10e-3;

    for (int i = 0; i < NUM_PARAMS; i++)
    {
        Params* params_plus = copy_params(curr_params);

        params_plus->array[i] += dx;

        gradients[i] = (loss_function(params_plus, data) - loss_function(curr_params, data)) / dx;

        free(params_plus);
    }
    return gradients;
}

Params* generate_random_params()
{   // generating a random state of parameters, this is needed for the initial state of the optimization
    Params* nasumicna = (Params*)malloc(sizeof(Params));

    for (int i = 0; i < NUM_PARAMS; i++)
        nasumicna->array[i] = rand() % 21 - 10;
    
    return nasumicna; 
}

void update_params(Params* curr_params, Data* data, float *velocities)
{
    float* gradients = loss_function_gradient(curr_params, data);
    float* new_velocities = (float*)malloc(sizeof(float)*NUM_PARAMS);

    for (int i = 0; i < NUM_PARAMS; i++)
    {
        new_velocities[i] = BETA * velocities[i] - ALPHA * clamp(gradients[i], -LIMIT_STEP, LIMIT_STEP);
        curr_params->array[i] += new_velocities[i];

        velocities[i] = new_velocities[i];
    }

    free(gradients);
    free(new_velocities);
}

void print_history(float* loss, float* gradients, Params** params, int final_iter)
{
    printf("Loss, Gradients, Params:\n");
    for (int i = 0; i < final_iter; i++)
    {
        printf("%f, %f, %f, %f\n", loss[i], gradients[i], params[i]->array[0], params[i]->array[1]);
    }
}


Params* gradient_descent()
{   // main function
    float gradients[MAX_ITER], loss[MAX_ITER];
    Params* params[MAX_ITER];
    float velocities[NUM_PARAMS] = {0};

    Params* curr_params = generate_random_params();
    Data* data = import_data("podaci_za_fit.csv");    // change the file name to import data

    int iter;
    for (iter = 0; iter < MAX_ITER; iter++)
    {
        gradients[iter] = loss_function_gradient(curr_params, data)[0];
        params[iter] = copy_params(curr_params);
        loss[iter] = loss_function(curr_params, data);

        update_params(curr_params, data, velocities);
        if (loss[iter] < TERMINATION_CRITERIUM)
            break;
    }
    // print_history(loss, gradients, params, iter);

    return curr_params;
}

int main()
{
    srand(time(NULL));

    clock_t start = clock();

    Params* optimized_params = gradient_descent();

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execute time: %f\n", time_spent);

    printf("Optimized parameters:\n");
    for (int i = 0; i < NUM_PARAMS; i++)
    {
        printf("%f  ", optimized_params->array[i]);
    }
    printf("\n");

    return 0;
}