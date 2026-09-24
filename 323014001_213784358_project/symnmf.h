#ifndef SYMNMF_H
#define SYMNMF_H

typedef struct {
    int rows;
    int cols;
    double *data;
} Matrix;

Matrix *matrix_create(int rows, int cols);
Matrix *matrix_copy(const Matrix *source);
void matrix_free(Matrix *matrix);
Matrix *read_points(const char *path);
Matrix *calculate_sym(const Matrix *points);
Matrix *calculate_ddg(const Matrix *points);
Matrix *calculate_norm(const Matrix *points);
Matrix *symnmf(const Matrix *initial_h, const Matrix *w, int max_iter,
               double epsilon);
void print_matrix(const Matrix *matrix);

#endif