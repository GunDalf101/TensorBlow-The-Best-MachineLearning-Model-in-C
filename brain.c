#include "brain.h"

int main(){
    srand(time(NULL));
    Matrix w1 = matrix_init(2, 2);
    Matrix b1 = matrix_init(1, 2);
    Matrix w2 = matrix_init(2, 1);
    Matrix b2 = matrix_init(1, 1);

    matrix_randomize(w1, 0, 1);
    matrix_randomize(b1, 0, 1);
    matrix_randomize(w2, 0, 1);
    matrix_randomize(b2, 0, 1);

    matrix_print(w1);
    matrix_print(b1);
    matrix_print(w2);
    matrix_print(b2);
    return 0;
}