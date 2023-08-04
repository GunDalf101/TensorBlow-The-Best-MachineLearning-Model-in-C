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
    size_t stride;
    size_t rows;
    size_t cols;
    float *data;
} Matrix;

Matrix matrix_init(size_t rows, size_t cols);
void matrix_randomize(Matrix M, float min, float max);
void matrix_fill(Matrix M, float val);
void matrix_copy(Matrix M, Matrix A);
void matrix_row(Matrix M, size_t row, Matrix A);
void matrix_col(Matrix M, size_t col, Matrix A);
void matrix_dot(Matrix M, Matrix A, Matrix B);
void matrix_add(Matrix M, Matrix A, Matrix B);
void matrix_print(Matrix M, const char *name);
void sigmoid_activation(Matrix M);
#define MATRIX_AT(M, i, j) M.data[i * M.cols + j]
#define MATRIX_PRINT(M) matrix_print(M, #M)
#define sigmoid(x) (1.0f / (1.0f + exp(-x)))
#define sigmoid_derivative(x) (x * (1.0f - x))
#define relu(x) (x > 0.0f ? x : 0.0f)
#define relu_derivative(x) (x > 0.0f ? 1.0f : 0.0f)
#define tanh(x) (2 / (1.0f + exp(-2 * x)) - 1.0f)
#define tanh_derivative(x) (1.0f - x * x)
#define leaky_relu(x) (x > 0.0f ? x : 0.01f * x)
#define leaky_relu_derivative(x) (x > 0.0f ? 1.0f : 0.01f)
#define identity(x) (x)
#define identity_derivative(x) (1f)
#define softmax(x) (exp(x) / exp(x).sum())
#define softmax_derivative(x) (x * (1f - x))
#define cross_entropy(x, y) (-log(x[y]))
#define cross_entropy_derivative(x, y) (x - y)

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

void matrix_copy(Matrix M, Matrix A) {
    if (M.rows != A.rows || M.cols != A.cols) {
        printf("Error: matrix dimensions do not match\n");
        exit(1);
    }
    memcpy(M.data, A.data, M.rows * M.cols * sizeof(*M.data));
}

void matrix_row(Matrix M, size_t row, Matrix A) {
    if (M.cols != A.cols) {
        printf("Error: matrix dimensions do not match\n");
        exit(1);
    }
    for (size_t i = 0; i < M.cols; i++) {
        M.data[i] = A.data[row * A.cols + i];
    }
}

void matrix_col(Matrix M, size_t col, Matrix A) {
    if (M.rows != A.rows) {
        printf("Error: matrix dimensions do not match\n");
        exit(1);
    }
    for (size_t i = 0; i < M.rows; i++) {
        M.data[i * M.cols + col] = A.data[i * A.cols];
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

void matrix_print(Matrix M, const char *name) {
    printf("%s:\n", name);
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

void sigmoid_activation(Matrix M) {
    for (size_t i = 0; i < M.rows; i++) {
        for (size_t j = 0; j < M.cols; j++) {
            M.data[i * M.cols + j] = sigmoid(M.data[i * M.cols + j]);
        }
    }
}

#endif