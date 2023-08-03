#ifndef BRAIN_H
#define BRAIN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} Matrix;

Matrix matrix_init(size_t rows, size_t cols);
void matrix_randomize(Matrix M, float min, float max);
void matrix_fill(Matrix M, float val);
void matrix_dot(Matrix M, Matrix A, Matrix B);
void matrix_add(Matrix M, Matrix A, Matrix B);
void matrix_print(Matrix M);

#endif

#ifndef BRAIN_IMPL
#define BRAIN_IMPL
Matrix matrix_init(size_t rows, size_t cols) {
    Matrix M;
    M.rows = rows;
    M.cols = cols;
    M.data = (float *)malloc(rows * cols * sizeof(*M.data));
    assert(M.data != NULL);
    return M;
}

void matrix_dot(Matrix M, Matrix A, Matrix B) {
    if (A.cols != B.rows) {
        printf("Error: matrix dimensions do not match\n");
        exit(1);
    }
    for (size_t i = 0; i < M.rows; i++) {
        for (size_t j = 0; j < M.cols; j++) {
            float sum = 0;
            for (size_t k = 0; k < A.cols; k++) {
                sum += A.data[i * A.cols + k] * B.data[k * B.cols + j];
            }
            M.data[i * M.cols + j] = sum;
        }
    }
}

void matrix_add(Matrix M, Matrix A, Matrix B) {
    if (A.rows != B.rows || A.cols != B.cols) {
        printf("Error: matrix dimensions do not match\n");
        exit(1);
    }
    for (size_t i = 0; i < M.rows; i++) {
        for (size_t j = 0; j < M.cols; j++) {
            M.data[i * M.cols + j] = A.data[i * A.cols + j] + B.data[i * B.cols + j];
        }
    }
}

void matrix_print(Matrix M) {
    for (size_t i = 0; i < M.rows; i++) {
        printf("| ");
        for (size_t j = 0; j < M.cols; j++) {
            printf("%f ", M.data[i * M.cols + j]);
        }
        printf("|\n");
    }
    printf("\n");
}

void matrix_fill(Matrix M, float val) {
    for (size_t i = 0; i < M.rows; i++) {
        for (size_t j = 0; j < M.cols; j++) {
            M.data[i * M.cols + j] = val;
        }
    }
}

void matrix_randomize(Matrix M, float min, float max) {
    for (size_t i = 0; i < M.rows; i++) {
        for (size_t j = 0; j < M.cols; j++) {
            M.data[i * M.cols + j] = ((float)rand() / RAND_MAX) * (max - min) + min;
        }
    }
}

#endif