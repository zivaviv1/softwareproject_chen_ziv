#include <stdio.h>
#include <stdlib.h>

struct cord
{
    double value;
    struct cord *next;
};

struct vector
{
    struct vector *next;
    struct cord *cords;
};

/* Prototypes */
double euclidean_distance(struct cord *p, double *q, int d);
double custom_sqrt(double n);

int main(int argc, char *argv[]) 
{
    /* Declarations */
    int k, iter, N, d, i, j, dim, iteration, closest_k;
    double n, min_dist, dist, max_delta, sum_sq, diff, delta, epsilon;
    char c;
    struct vector *head_vec, *curr_vec, *v, *temp_v;
    struct cord *head_cord, *curr_cord, *temp, *curr_c, *temp_c;
    double **centroids, **new_centroids;
    double *cluster_counts;

    /* Validation of input args */
    if (argc < 2 || argc > 3) {
        printf("An Error Has Occurred\n");
        return 1;
    }

    {
        char *endptr;
        long k_long = strtol(argv[1], &endptr, 10);
        if (argv[1][0] == '\0' || *endptr != '\0' || k_long <= 0 || k_long > 2147483647L) {
            k = -1;
        } else {
            k = (int)k_long;
        }
    }

    if (argc == 3) {
        char *endptr;
        long iter_long = strtol(argv[2], &endptr, 10);
        if (argv[2][0] == '\0' || *endptr != '\0' || iter_long <= 0 || iter_long > 2147483647L) {
            iter = -1;
        } else {
            iter = (int)iter_long;
        }
    } else {
        iter = 400;
    }
    
    if (iter >= 800 || iter <= 1) {
        printf("Incorrect maximum iteration!\n");
        return 1;
    }
    
    N = 0; 
    d = 0; 
    centroids = NULL; 

    head_cord = malloc(sizeof(struct cord));
    curr_cord = head_cord;
    curr_cord->next = NULL;

    head_vec = malloc(sizeof(struct vector));
    curr_vec = head_vec;
    curr_vec->next = NULL;
    curr_vec->cords = NULL;

    /* Read the file */
    while (scanf("%lf%c", &n, &c) == 2)
    {
        if (N == 0) {
            d++; 
        }

        if (c == '\n')
        {
            curr_cord->value = n;
            curr_vec->cords = head_cord;
            
            if (N == 0) {
                centroids = malloc(k * sizeof(double *));
            }

            if (N < k) {
                centroids[N] = malloc(d * sizeof(double));
                temp = head_cord;
                for (dim = 0; dim < d; dim++) {
                    centroids[N][dim] = temp->value;
                    temp = temp->next;
                }
            }

            N++;

            curr_vec->next = malloc(sizeof(struct vector));
            curr_vec = curr_vec->next;
            curr_vec->next = NULL;
            curr_vec->cords = NULL; 

            head_cord = malloc(sizeof(struct cord));
            curr_cord = head_cord;
            curr_cord->next = NULL;
            continue;
        }

        curr_cord->value = n;
        curr_cord->next = malloc(sizeof(struct cord));
        curr_cord = curr_cord->next;
        curr_cord->next = NULL;
    }
    
    if (k >= N || N == 0 || k <= 1) {
        printf("Incorrect number of clusters!\n");
        return 1;
    }

    /* Run the algorithm */
    epsilon = 0.001;
    new_centroids = malloc(k * sizeof(double *));
    cluster_counts = malloc(k * sizeof(double)); 
    
    for (i = 0; i < k; i++) {
        new_centroids[i] = malloc(d * sizeof(double));
    }

    for (iteration = 0; iteration < iter; iteration++) 
    {
        /* Reset accumulators */
        for (i = 0; i < k; i++) {
            cluster_counts[i] = 0.0;
            for (dim = 0; dim < d; dim++) {
                new_centroids[i][dim] = 0.0;
            }
        }

        /* Assign vectors to closest centroid */
        v = head_vec;
        for (i = 0; i < N; i++) {
            if (v->cords == NULL) break;

            min_dist = 1e300; /* Acts as INFINITY */
            closest_k = 0;

            for (j = 0; j < k; j++) {
                dist = euclidean_distance(v->cords, centroids[j], d);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_k = j;
                }
            }

            cluster_counts[closest_k] += 1.0;
            
            curr_c = v->cords;
            for (dim = 0; dim < d; dim++) {
                new_centroids[closest_k][dim] += curr_c->value;
                curr_c = curr_c->next;
            }
            v = v->next;
        }

        /* Update the centroids */
        max_delta = 0.0;
        for (i = 0; i < k; i++) {
            if (cluster_counts[i] == 0.0) {
                for (dim = 0; dim < d; dim++) {
                    new_centroids[i][dim] = centroids[i][dim];
                }
            } else {
                for (dim = 0; dim < d; dim++) {
                    new_centroids[i][dim] /= cluster_counts[i];
                }
            }

            sum_sq = 0.0;
            for (dim = 0; dim < d; dim++) {
                diff = new_centroids[i][dim] - centroids[i][dim];
                sum_sq += diff * diff;
            }
            
            delta = custom_sqrt(sum_sq);

            if (delta > max_delta) {
                max_delta = delta;
            }

            for (dim = 0; dim < d; dim++) {
                centroids[i][dim] = new_centroids[i][dim];
            }
        }

        /* Check for convergence */
        if (max_delta < epsilon) {
            break;
        }
    }

    /* Print output */
    for (i = 0; i < k; i++) {
        for (dim = 0; dim < d; dim++) {
            printf("%.4f", centroids[i][dim]);
            if (dim < d - 1) printf(",");
        }
        printf("\n");
    }

    /* Free the memory */
    for (i = 0; i < k; i++) {
        free(centroids[i]);
        free(new_centroids[i]);
    }
    free(centroids);
    free(new_centroids);
    free(cluster_counts);

    v = head_vec;
    while (v != NULL) {
        curr_c = v->cords;
        while (curr_c != NULL) {
            temp_c = curr_c;
            curr_c = curr_c->next;
            free(temp_c);
        }
        temp_v = v;
        v = v->next;
        free(temp_v);
    }
    free(head_cord);

    return 0;
}

/* Distance calculation */
double euclidean_distance(struct cord *p, double *q, int d) {
    double sum_sq = 0.0;
    double diff;
    int i;
    struct cord *curr = p;
    for (i = 0; i < d; i++) {
        diff = curr->value - q[i];
        sum_sq += diff * diff;
        curr = curr->next;
    }
    return custom_sqrt(sum_sq);
}

/* Newton's method for calculating square root */
double custom_sqrt(double n) {
    double x, y, precision;
    
    if (n <= 0.0) return 0.0;
    
    x = n;
    y = 1.0;
    precision = 0.0000001; 
    
    while (x - y > precision) {
        x = (x + y) / 2.0;
        y = n / x;
    }
    
    return x;
}