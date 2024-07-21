#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#define NUM_PARAMS 1

#define MAX_ITER 1000
#define ALPHA 0.01
#define BETA 0.1
#define LIMIT_STEP 1

#define BUFFER_INPUT 5

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

    while (fscanf(file, "%f %f\n", &data->x[curr_line], &data->y[curr_line]) != EOF)
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

float loss_function(Params* curr_params, Data* data)
{   // the function that is optimized
    float loss = 0;

    for (int i = 0; i < NUM_PARAMS; i++)
    {
        for (int j = 0; j < data->len; j++)
        {
            loss += pow( pow(curr_params->array[i], data->x[j]) - data->y[j], 2 );
        }
    }

    return loss;
}

float* loss_function_gradient(Params* curr_params, Data* data)
{   // the gradient of the function that needs to be optimized
    float* gradients = (float*)malloc(sizeof(float)*NUM_PARAMS);

    float dx = 10e-4;

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

void update_params(Params* curr_params, Data* data)
{
    float* gradients = loss_function_gradient(curr_params, data);
    for (int i = 0; i < NUM_PARAMS; i++)
    {
        curr_params->array[i] -= ALPHA * clamp(gradients[i], -LIMIT_STEP, LIMIT_STEP);
    }
}

void print_history(float* gradients, Params** params)
{
    printf("Gradients, Params:\n");
    for (int i = 0; i < MAX_ITER; i++)
    {
        printf("%f, %f\n", gradients[i], params[i]->array[0]);
    }
}


Params* gradient_descent()
{   // main function
    float gradients[MAX_ITER];
    Params* params[MAX_ITER];

    Params* curr_params = generate_random_params();
    Data* data = import_data("data.txt");    // change the file name to import data

    for (int iter = 0; iter < MAX_ITER; iter++)
    {
        gradients[iter] = loss_function_gradient(curr_params, data)[0];
        params[iter] = copy_params(curr_params);

        update_params(curr_params, data);
    }

    print_history(gradients, params);

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