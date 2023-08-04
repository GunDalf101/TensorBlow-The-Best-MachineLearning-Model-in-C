#include "brain.h"

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0
};

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
    size_t ark[] = {2, 2, 1};
    neuralNet nn = neuralNetInit(ark, ARRAY_LEN(ark));
    neuralNet g = neuralNetInit(ark, ARRAY_LEN(ark));
    neuralNetRandomize(nn, 0, 1);
    printf("cost: %f\n", neuralNetCost(nn, train_input, train_output));
    neuralNetFiniteDifference(nn, g, 0.1, train_input, train_output);
    neuralNetLearn(nn, g, 0.1);
    while (neuralNetCost(nn, train_input, train_output) > 0.000013){
        neuralNetFiniteDifference(nn, g, 0.1, train_input, train_output);
        neuralNetLearn(nn, g, 0.1);
        printf("cost: %f\n", neuralNetCost(nn, train_input, train_output));
    }
    printf("cost: %f\n", neuralNetCost(nn, train_input, train_output));
    for (size_t i = 0; i < n; i++){
        Matrix x = matrix_init(1, train_input.cols);
        Matrix y = matrix_init(1, train_output.cols);
        matrix_row(x, i, train_input);
        matrix_row(y, i, train_output);
        matrix_copy(NEURALNET_INPUT(nn), x);
        neuralNetForward(nn);
        printf("%f %f %f\n", MATRIX_AT(x, 0, 0), MATRIX_AT(x, 0, 1), MATRIX_AT(NEURALNET_OUTPUT(nn), 0, 0));
    }
    return 0;
}