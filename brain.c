#include "brain.h"

typedef struct {
    Matrix a0, a1, a2;
    Matrix w1, b1;
    Matrix w2, b2;
} neuralNet;

neuralNet XorInit(size_t input_size, size_t hidden_size, size_t output_size){
    neuralNet Xor;
    Xor.a0 = matrix_init(1, input_size);
    Xor.w1 = matrix_init(input_size, hidden_size);
    Xor.b1 = matrix_init(1, hidden_size);
    Xor.a1 = matrix_init(1, hidden_size);
    Xor.w2 = matrix_init(hidden_size, output_size);
    Xor.b2 = matrix_init(1, output_size);
    Xor.a2 = matrix_init(1, output_size);
    return Xor;
}

float XorForward(neuralNet Xor) {
    matrix_dot(Xor.a1, Xor.a0, Xor.w1);
    matrix_add(Xor.a1, Xor.a1, Xor.b1);
    sigmoid_activation(Xor.a1);
    matrix_dot(Xor.a2, Xor.a1, Xor.w2);
    matrix_add(Xor.a2, Xor.a2, Xor.b2);
    sigmoid_activation(Xor.a2);
    return MATRIX_AT(Xor.a2, 0, 0);
}

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0
};

float model_cost (neuralNet M, Matrix train_input, Matrix train_output){
    assert(train_input.rows == train_output.rows);
    assert(train_output.cols == M.a2.cols);
    size_t n = train_input.rows;
    float cost = 0;

    for (size_t i = 0; i < n; i++){
        Matrix x = matrix_init(1, train_input.cols);
        Matrix y = matrix_init(1, train_output.cols);
        matrix_row(x, i, train_input);
        matrix_row(y, i, train_output);
        matrix_copy(M.a0, x);
        XorForward(M);
        size_t m = train_input.cols;
        for (size_t j = 0; j < m; j++){
            float d = MATRIX_AT(M.a2, 0, j) - MATRIX_AT(y, 0, j);
            cost += pow(d, 2);
        }
    }
    return cost / n;
}

void finite_difference (neuralNet M, neuralNet g, float epsilon, Matrix train_input, Matrix train_output){
    float saved;
    float cost = model_cost(M, train_input, train_output);

    for (size_t i = 0; i < M.w1.rows; i++){
        for (size_t j = 0; j < M.w1.cols; j++){
            saved = MATRIX_AT(M.w1, i, j);
            MATRIX_AT(M.w1, i, j) += epsilon;
            MATRIX_AT(g.w1, i, j) = (model_cost(M, train_input, train_output) - cost) / epsilon;
            MATRIX_AT(M.w1, i, j) = saved;
        }
    }
    for (size_t i = 0; i < M.b1.rows; i++){
        for (size_t j = 0; j < M.b1.cols; j++){
            saved = MATRIX_AT(M.b1, i, j);
            MATRIX_AT(M.b1, i, j) += epsilon;
            MATRIX_AT(g.b1, i, j) = (model_cost(M, train_input, train_output) - cost) / epsilon;
            MATRIX_AT(M.b1, i, j) = saved;
        }
    }
    for (size_t i = 0; i < M.w2.rows; i++){
        for (size_t j = 0; j < M.w2.cols; j++){
            saved = MATRIX_AT(M.w2, i, j);
            MATRIX_AT(M.w2, i, j) += epsilon;
            MATRIX_AT(g.w2, i, j) = (model_cost(M, train_input, train_output) - cost) / epsilon;
            MATRIX_AT(M.w2, i, j) = saved;
        }
    }
    for (size_t i = 0; i < M.b2.rows; i++){
        for (size_t j = 0; j < M.b2.cols; j++){
            saved = MATRIX_AT(M.b2, i, j);
            MATRIX_AT(M.b2, i, j) += epsilon;
            MATRIX_AT(g.b2, i, j) = (model_cost(M, train_input, train_output) - cost) / epsilon;
            MATRIX_AT(M.b2, i, j) = saved;
        }
    }
}

void learninXor(neuralNet M, neuralNet g, float rate)
{
    for (size_t i = 0; i < M.w1.rows; i++){
        for (size_t j = 0; j < M.w1.cols; j++){
            MATRIX_AT(M.w1, i, j) -= rate * MATRIX_AT(g.w1, i, j);
        }
    }
    for (size_t i = 0; i < M.b1.rows; i++){
        for (size_t j = 0; j < M.b1.cols; j++){
            MATRIX_AT(M.b1, i, j) -= rate * MATRIX_AT(g.b1, i, j);
        }
    }
    for (size_t i = 0; i < M.w2.rows; i++){
        for (size_t j = 0; j < M.w2.cols; j++){
            MATRIX_AT(M.w2, i, j) -= rate * MATRIX_AT(g.w2, i, j);
        }
    }
    for (size_t i = 0; i < M.b2.rows; i++){
        for (size_t j = 0; j < M.b2.cols; j++){
            MATRIX_AT(M.b2, i, j) -= rate * MATRIX_AT(g.b2, i, j);
        }
    }
}

int main(){
    srand(time(NULL));

    size_t n = sizeof(td) / sizeof(td[0]) / 3;
    Matrix train_input = matrix_init(n, 2);
    Matrix train_output = matrix_init(n, 1);
    for (size_t i = 0; i < n; i++){
        MATRIX_AT(train_input, i, 0) = td[i * 3 + 0];
        MATRIX_AT(train_input, i, 1) = td[i * 3 + 1];
        MATRIX_AT(train_output, i, 0) = td[i * 3 + 2];
    }
    neuralNet g = XorInit(2, 2, 1);
    neuralNet M = XorInit(2, 2, 1);
    matrix_randomize(M.w1, 0, 1);
    matrix_randomize(M.b1, 0, 1);
    matrix_randomize(M.w2, 0, 1);
    matrix_randomize(M.b2, 0, 1);
    
    printf("cost: %f\n", model_cost(M, train_input, train_output));
    while (model_cost(M, train_input, train_output) > 0.0001){
        finite_difference(M, g, 0.1, train_input, train_output);
        learninXor(M, g, 0.1);
        printf("cost: %f\n", model_cost(M, train_input, train_output));
    }
    printf("cost: %f\n", model_cost(M, train_input, train_output));
    for (size_t i = 0; i < 2; i++){
        for (size_t j = 0; j < 2; j++){
            MATRIX_AT(M.a0, 0, 0) = i;
            MATRIX_AT(M.a0, 0, 1) = j;
            XorForward(M);
            float y_hat = MATRIX_AT(M.a2, 0, 0);
            printf("x1: %f, x2: %f, y_hat: %f\n", (float)i, (float)j, y_hat);
        }
    }
    return 0;
}