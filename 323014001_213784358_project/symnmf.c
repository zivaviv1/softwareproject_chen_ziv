#include "symnmf.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE_SIZE 8192 /* max length of an input line */
#define BETA 0.5 /* beta in the H update rule */

static int is_blank(const char *line);
static int parse_values(char *line, double *values, int limit, int *count);
static int read_shape(FILE *file, int *rows, int *cols);
static int load_values(FILE *file, Matrix *matrix);
static double squared_distance(const Matrix *points, int first, int second);
static double *degrees_for(const Matrix *matrix);
static Matrix *matrix_product(const Matrix *left, const Matrix *right);
static Matrix *matrix_transpose(const Matrix *matrix);
static Matrix *update_h(const Matrix *w, const Matrix *h);
static double difference_squared(const Matrix *first, const Matrix *second);
static int valid_goal(const char *goal);
static Matrix *result_for_goal(const char *goal, const Matrix *points);

/* Creates a matrix filled with zeros */
Matrix *matrix_create(int rows, int cols)
{
    Matrix *matrix;
    size_t count;

    if (rows <= 0 || cols <= 0) {
        return NULL;
    }
    count = (size_t)rows * (size_t)cols;
    /* guard against size overflow */
    if (count > (size_t)-1 / sizeof(double)) {
        return NULL;
    }
    matrix = (Matrix *)malloc(sizeof(Matrix));
    if (matrix == NULL) {
        return NULL;
    }
    matrix->data = (double *)calloc(count, sizeof(double));
    if (matrix->data == NULL) {
        free(matrix);
        return NULL;
    }
    matrix->rows = rows;
    matrix->cols = cols;
    return matrix;
}

/* Returns a new copy of the given matrix */
Matrix *matrix_copy(const Matrix *source)
{
    Matrix *copy;
    size_t count;

    if (source == NULL) {
        return NULL;
    }
    copy = matrix_create(source->rows, source->cols);
    if (copy == NULL) {
        return NULL;
    }
    count = (size_t)source->rows * (size_t)source->cols;
    memcpy(copy->data, source->data, count * sizeof(double));
    return copy;
}

/* Frees a matrix and its data */
void matrix_free(Matrix *matrix)
{
    if (matrix != NULL) {
        free(matrix->data);
        free(matrix);
    }
}

/* Check if a line is only whitesapce */
static int is_blank(const char *line)
{
    while (isspace((unsigned char)*line)) {
        line++;
    }
    return *line == '\0';
}

/* Parses a  line of numbers into values and count it*/
static int parse_values(char *line, double *values, int limit, int *count)
{
    char *cursor;
    char *end;
    int column;
    double value;
    cursor = line;
    column = 0;
    while (1) {
        while (isspace((unsigned char)*cursor)) {
            cursor++;
        }
        if (*cursor == '\0') break;
        value = strtod(cursor, &end);
        if (end == cursor || (limit > 0 && column >= limit)) {
            return 0;
        }
        if (values != NULL) {
            values[column] = value;
        }
        column++;
        cursor = end;
        while (isspace((unsigned char)*cursor)) {
            cursor++;
        }
        if (*cursor == '\0') break;
        if (*cursor != ',') {
            return 0;
        }
        cursor++;
        while (isspace((unsigned char)*cursor)) {
            cursor++;
        }
        if (*cursor == '\0') return 0;
    }
    *count = column;
    return column > 0;
}

/* Count rows and check all rows have the same length in the file */
static int read_shape(FILE *file, int *rows, int *cols)
{
    char line[LINE_SIZE];
    int count;
    int columns;

    *rows = 0;
    columns = 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        if (is_blank(line)) {
            continue;
        }
        if (!parse_values(line, NULL, 0, &count)) {
            return 0;
        }
        if (columns == 0) {
            columns = count;
        } else if (columns != count) {
            return 0;
        }
        (*rows)++;
    }
    if (ferror(file) || *rows == 0) {
        return 0;
    }
    *cols = columns;
    return 1;
}

/* Fill the matrix with the values */
static int load_values(FILE *file, Matrix *matrix)
{
    char line[LINE_SIZE];
    int count;
    int row;

    row = 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        if (is_blank(line)) {
            continue;
        }
        if (row >= matrix->rows || !parse_values(line,
                matrix->data + row * matrix->cols, matrix->cols, &count)) {
            return 0;
        }
        if (count != matrix->cols) {
            return 0;
        }
        row++;
    }
    return !ferror(file) && row == matrix->rows;
}

/* Reads the data points file into a matrix */
Matrix *read_points(const char *path)
{
    FILE *file;
    Matrix *points;
    int rows;
    int cols;

    file = fopen(path, "r");
    if (file == NULL) {
        return NULL;
    }
    if (!read_shape(file, &rows, &cols)) {
        fclose(file);
        return NULL;
    }
    /* go back to the start for the second pass */
    rewind(file);
    points = matrix_create(rows, cols);
    if (points == NULL || !load_values(file, points)) {
        matrix_free(points);
        fclose(file);
        return NULL;
    }
    fclose(file);
    return points;
}

/* Squared Euclidean distance between two rows of points */
static double squared_distance(const Matrix *points, int first, int second)
{
    int column;
    double difference;
    double distance;

    distance = 0.0;
    for (column = 0; column < points->cols; column++) {
        difference = points->data[first * points->cols + column]
            - points->data[second * points->cols + column];
        distance += difference * difference;
    }
    return distance;
}

/* Get similarity matrix */
Matrix *calculate_sym(const Matrix *points)
{
    Matrix *similarity;
    int first;
    int second;
    double value;

    if (points == NULL) {
        return NULL;
    }
    similarity = matrix_create(points->rows, points->rows);
    if (similarity == NULL) {
        return NULL;
    }
    /* A is symmetric, compute the upper triangle and mirror it.
       The diagonal stays 0 from calloc */
    for (first = 0; first < points->rows; first++) {
        for (second = first + 1; second < points->rows; second++) {
            value = exp(-squared_distance(points, first, second) / 2.0);
            similarity->data[first * points->rows + second] = value;
            similarity->data[second * points->rows + first] = value;
        }
    }
    return similarity;
}

/* Get degree of each row */
static double *degrees_for(const Matrix *matrix)
{
    double *degrees;
    int row;
    int column;

    degrees = (double *)calloc((size_t)matrix->rows, sizeof(double));
    if (degrees == NULL) {
        return NULL;
    }
    for (row = 0; row < matrix->rows; row++) {
        for (column = 0; column < matrix->cols; column++) {
            degrees[row] += matrix->data[row * matrix->cols + column];
        }
    }
    return degrees;
}

/* Calculate diagonal degree matrix D */
Matrix *calculate_ddg(const Matrix *points)
{
    Matrix *similarity;
    Matrix *degree;
    double *degrees;
    int row;

    similarity = calculate_sym(points);
    if (similarity == NULL) {
        return NULL;
    }
    degrees = degrees_for(similarity);
    degree = matrix_create(similarity->rows, similarity->cols);
    if (degrees == NULL || degree == NULL) {
        free(degrees);
        matrix_free(similarity);
        matrix_free(degree);
        return NULL;
    }
    for (row = 0; row < degree->rows; row++) {
        degree->data[row * degree->cols + row] = degrees[row];
    }
    free(degrees);
    matrix_free(similarity);
    return degree;
}

/* calculate normalized similarity */
Matrix *calculate_norm(const Matrix *points)
{
    Matrix *similarity;
    Matrix *normalized;
    double *degrees;
    int row;
    int column;

    similarity = calculate_sym(points);
    if (similarity == NULL) {
        return NULL;
    }
    degrees = degrees_for(similarity);
    normalized = matrix_create(similarity->rows, similarity->cols);
    if (degrees == NULL || normalized == NULL) {
        free(degrees);
        matrix_free(similarity);
        matrix_free(normalized);
        return NULL;
    }
    for (row = 0; row < normalized->rows; row++) {
        for (column = 0; column < normalized->cols; column++) {
            /* avoid division by zero */
            if (degrees[row] > 0.0 && degrees[column] > 0.0) {
                normalized->data[row * normalized->cols + column] =
                    similarity->data[row * similarity->cols + column]
                    / sqrt(degrees[row] * degrees[column]);
            }
        }
    }
    free(degrees);
    matrix_free(similarity);
    return normalized;
}

/* Caculate left * right */
static Matrix *matrix_product(const Matrix *left, const Matrix *right)
{
    Matrix *result;
    int row;
    int column;
    int index;

    if (left == NULL || right == NULL || left->cols != right->rows) {
        return NULL;
    }
    result = matrix_create(left->rows, right->cols);
    if (result == NULL) {
        return NULL;
    }
    for (row = 0; row < left->rows; row++) {
        for (column = 0; column < right->cols; column++) {
            for (index = 0; index < left->cols; index++) {
                result->data[row * result->cols + column] +=
                    left->data[row * left->cols + index]
                    * right->data[index * right->cols + column];
            }
        }
    }
    return result;
}

/* Get transpose of the matrix */
static Matrix *matrix_transpose(const Matrix *matrix)
{
    Matrix *transpose;
    int row;
    int column;

    transpose = matrix_create(matrix->cols, matrix->rows);
    if (transpose == NULL) {
        return NULL;
    }
    for (row = 0; row < matrix->rows; row++) {
        for (column = 0; column < matrix->cols; column++) {
            transpose->data[column * transpose->cols + row] =
                matrix->data[row * matrix->cols + column];
        }
    }
    return transpose;
}

/* One update step:
   H_new = H * (1 - beta + beta * (W H) / (H H^T H)), element-wise */
static Matrix *update_h(const Matrix *w, const Matrix *h)
{
    Matrix *wh;
    Matrix *ht;
    Matrix *gram;
    Matrix *denominator;
    Matrix *next;
    int row;
    int column;

    /* numerator W H and denominator H (H^T H) */
    wh = matrix_product(w, h);
    ht = matrix_transpose(h);
    gram = matrix_product(ht, h);
    denominator = matrix_product(h, gram);
    next = matrix_create(h->rows, h->cols);
    if (wh == NULL || ht == NULL || gram == NULL || denominator == NULL
            || next == NULL) {
        matrix_free(wh);
        matrix_free(ht);
        matrix_free(gram);
        matrix_free(denominator);
        matrix_free(next);
        return NULL;
    }
    for (row = 0; row < h->rows; row++) {
        for (column = 0; column < h->cols; column++) {
            next->data[row * h->cols + column] = h->data[row * h->cols + column];
            if (denominator->data[row * h->cols + column] > 0.0) {
                next->data[row * h->cols + column] *= 1.0 - BETA + BETA
                    * wh->data[row * h->cols + column]
                    / denominator->data[row * h->cols + column];
            }
        }
    }
    matrix_free(wh);
    matrix_free(ht);
    matrix_free(gram);
    matrix_free(denominator);
    return next;
}

/* Squared norm of (first - second) */
static double difference_squared(const Matrix *first, const Matrix *second)
{
    double difference;
    double total;
    int index;

    total = 0.0;
    for (index = 0; index < first->rows * first->cols; index++) {
        difference = first->data[index] - second->data[index];
        total += difference * difference;
    }
    return total;
}

/* Runs the SymNMF updates from initial_h until ||H_new - H||_F^2 < epsilon
   or max_iter iterations, returns the final H */
Matrix *symnmf(const Matrix *initial_h, const Matrix *w, int max_iter,
               double epsilon)
{
    Matrix *current;
    Matrix *next;
    int iteration;
    double delta;

    if (initial_h == NULL || w == NULL || w->rows != w->cols
            || initial_h->rows != w->rows || max_iter <= 0 || epsilon < 0.0) {
        return NULL;
    }
    current = matrix_copy(initial_h);
    if (current == NULL) {
        return NULL;
    }
    for (iteration = 0; iteration < max_iter; iteration++) {
        next = update_h(w, current);
        if (next == NULL) {
            matrix_free(current);
            return NULL;
        }
        delta = difference_squared(next, current);
        matrix_free(current);
        current = next;
        if (delta < epsilon) {
            break;
        }
    }
    return current;
}

/* Prints the matrix */
void print_matrix(const Matrix *matrix)
{
    int row;
    int column;

    for (row = 0; row < matrix->rows; row++) {
        for (column = 0; column < matrix->cols; column++) {
            printf("%.4f", matrix->data[row * matrix->cols + column]);
            if (column + 1 < matrix->cols) {
                printf(",");
            }
        }
        printf("\n");
    }
}

/* The C program supports only sym, ddg and norm */
static int valid_goal(const char *goal)
{
    return strcmp(goal, "sym") == 0 || strcmp(goal, "ddg") == 0
        || strcmp(goal, "norm") == 0;
}

/* Calculates the matrix that matches the goal */
static Matrix *result_for_goal(const char *goal, const Matrix *points)
{
    if (strcmp(goal, "sym") == 0) {
        return calculate_sym(points);
    }
    if (strcmp(goal, "ddg") == 0) {
        return calculate_ddg(points);
    }
    return calculate_norm(points);
}

int main(int argc, char **argv)
{
    Matrix *points;
    Matrix *result;

    if (argc != 3 || !valid_goal(argv[1])) {
        printf("An Error Has Occurred\n");
        return 1;
    }
    points = read_points(argv[2]);
    if (points == NULL) {
        printf("An Error Has Occurred\n");
        return 1;
    }
    result = result_for_goal(argv[1], points);
    matrix_free(points);
    if (result == NULL) {
        printf("An Error Has Occurred\n");
        return 1;
    }
    print_matrix(result);
    matrix_free(result);
    return 0;
}