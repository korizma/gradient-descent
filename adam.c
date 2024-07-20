#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#define MAX_ITER 1000
#define ALPHA 0.001
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 10e-8

typedef struct dataPoint
{
    int nesto;
} DataPoint;

float loss_function(DataPoint curr_point)
{   // ovo je funkcija koja se optimizuje
    return 1;
}

float loss_function_gradient(DataPoint curr_point)
{   // ovo je gradient loss funkcij, tj. izvod funkcije koje optimizuju
    return 1;
}

float agregate_next(float m_t, DataPoint curr_point)
{
    float gradient_at_point = loss_function_gradient(curr_point);
    return BETA1 * m_t +  (1 - BETA1) * (gradient_at_point * gradient_at_point);
}


