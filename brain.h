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
void matrix_print(Matrix M, const char *name, size_t padd);
void sigmoid_activation(Matrix M);
#define MATRIX_AT(M, i, j) M.data[i * M.cols + j]
#define MATRIX_PRINT(M) matrix_print(M, #M, 0)
#define NEURALNET_PRINT(nn) neuralNetPrint(nn, #nn)
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

typedef struct {
    size_t count;
    Matrix *a; // count + 1
    Matrix *w; // count
    Matrix *b; // count
} neuralNet;

#define NEURALNET_INPUT(nn) nn.a[0]
#define NEURALNET_OUTPUT(nn) nn.a[nn.count]
#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])


neuralNet neuralNetInit(size_t *ark, size_t arkount);
void neuralNetPrint(neuralNet nn, const char *name);
void neuralNetRandomize(neuralNet nn, float min, float max);
void neuralNetForward(neuralNet nn);
void neuralNetFiniteDifference(neuralNet M, neuralNet g, float epsilon, Matrix train_input, Matrix train_output);
float neuralNetCost(neuralNet nn, Matrix train_input, Matrix train_output);
void neuralNetLearn(neuralNet nn, neuralNet g, float rate);

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

void matrix_print(Matrix M, const char *name, size_t padd) {
    printf("%*s%s: [\n", (int)padd, "", name);
    for (size_t i = 0; i < M.rows; i++) {
        printf("%*s", (int)padd, "");
        for (size_t j = 0; j < M.cols; j++) {
            printf("    %f ", M.data[i * M.cols + j]);
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padd, "");
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

neuralNet neuralNetInit(size_t *ark, size_t arkount){
    assert(arkount > 0);
    neuralNet nn;
    nn.count = arkount - 1;

    nn.w = (Matrix *)malloc(nn.count * sizeof(*nn.w));
    assert(nn.w != NULL);
    nn.b = (Matrix *)malloc(nn.count * sizeof(*nn.b));
    assert(nn.b != NULL);
    nn.a = (Matrix *)malloc((nn.count + 1) * sizeof(*nn.a));
    assert(nn.a != NULL);

    nn.a[0] = matrix_init(1, ark[0]);
    for (size_t i = 0; i < nn.count; i++){
        nn.w[i] = matrix_init(nn.a[i].cols, ark[i + 1]);
        nn.b[i] = matrix_init(1, ark[i + 1]);
        nn.a[i + 1] = matrix_init(1, ark[i + 1]);
    }

    return nn;
}

void neuralNetPrint(neuralNet nn, const char *name){
    char buf[256];
    printf("%s: [\n", name);
    for (size_t i = 0; i < nn.count; i++){
        snprintf(buf, sizeof(buf), "weight %zu", i);
        matrix_print(nn.w[i], buf, i * 3);
        snprintf(buf, sizeof(buf), "bias %zu", i);
        matrix_print(nn.b[i], buf, i * 3);
    }
    printf("]\n");
}

void neuralNetRandomize(neuralNet nn, float min, float max){
    for (size_t i = 0; i < nn.count; i++){
        matrix_randomize(nn.w[i], min, max);
        matrix_randomize(nn.b[i], min, max);
    }
}

void neuralNetForward(neuralNet nn){
    for (size_t i = 0; i < nn.count; i++){
        matrix_dot(nn.a[i + 1], nn.a[i], nn.w[i]);
        matrix_add(nn.a[i + 1], nn.a[i + 1], nn.b[i]);
        sigmoid_activation(nn.a[i + 1]);
    }
}

float neuralNetCost(neuralNet nn, Matrix train_input, Matrix train_output){
    assert(train_input.rows == train_output.rows);
    assert(train_input.cols == NEURALNET_INPUT(nn).cols);
    size_t n = train_input.rows;
    float cost = 0;
    for (size_t i = 0; i < n; i++){
        Matrix x = matrix_init(1, train_input.cols);
        Matrix y = matrix_init(1, train_output.cols);
        matrix_row(x, i, train_input);
        matrix_row(y, i, train_output);
        matrix_copy(NEURALNET_INPUT(nn), x);
        neuralNetForward(nn);
        size_t m = train_output.cols;
        for (size_t j = 0; j < m; j++){
            float d = MATRIX_AT(NEURALNET_OUTPUT(nn), 0, j) - MATRIX_AT(y, 0, j);
            cost += pow(d, 2);
        }
    }
    return cost / n;
}

void neuralNetFiniteDifference(neuralNet M, neuralNet g, float epsilon, Matrix train_input, Matrix train_output){
    float saved;
    float cost = neuralNetCost(M, train_input, train_output);
    for (size_t i = 0; i < M.count; i++){
        for (size_t j = 0; j < M.w[i].rows; j++){
            for (size_t k = 0; k < M.w[i].cols; k++){
                saved = MATRIX_AT(M.w[i], j, k);
                MATRIX_AT(M.w[i], j, k) += epsilon;
                MATRIX_AT(g.w[i], j, k) = (neuralNetCost(M, train_input, train_output) - cost) / epsilon;
                MATRIX_AT(M.w[i], j, k) = saved;
            }
        }
        for (size_t j = 0; j < M.b[i].rows; j++){
            for (size_t k = 0; k < M.b[i].cols; k++){
                saved = MATRIX_AT(M.b[i], j, k);
                MATRIX_AT(M.b[i], j, k) += epsilon;
                MATRIX_AT(g.b[i], j, k) = (neuralNetCost(M, train_input, train_output) - cost) / epsilon;
                MATRIX_AT(M.b[i], j, k) = saved;
            }
        }
    }
}

void neuralNetLearn(neuralNet nn, neuralNet g, float rate){
    for (size_t i = 0; i < nn.count; i++){
        for (size_t j = 0; j < nn.w[i].rows; j++){
            for (size_t k = 0; k < nn.w[i].cols; k++){
                MATRIX_AT(nn.w[i], j, k) -= rate * MATRIX_AT(g.w[i], j, k);
            }
        }
        for (size_t j = 0; j < nn.b[i].rows; j++){
            for (size_t k = 0; k < nn.b[i].cols; k++){
                MATRIX_AT(nn.b[i], j, k) -= rate * MATRIX_AT(g.b[i], j, k);
            }
        }
    }
}

#endif