#include <stdio.h>
#include <math.h>

int is_perfect_square(int num) {
    if (num < 0) return 0;
    int s = sqrt(num);
    return s * s == num;
}

int calculate_weight(int num) {
    int weight = 0;
    if (is_perfect_square(num)) weight += 5;
    if (num % 4 == 0 && num % 6 == 0) weight += 4;
    if (num % 2 == 0) weight += 3;
    return weight;
}

int main() {
    int numbers[] = {10, 36, 54, 89, 12,72,12,81};
    int count = sizeof(numbers)/sizeof(numbers[0]);
    int weights[count];

    for(int i=0; i<count; i++) {
        weights[i] = calculate_weight(numbers[i]);
    }

    for(int i=0; i<count-1; i++) {
        for(int j=0; j<count-i-1; j++) {
            if(weights[j] > weights[j+1] ||
              (weights[j] == weights[j+1] && numbers[j] > numbers[j+1])) {
                int temp_num = numbers[j];
                int temp_weight = weights[j];
                numbers[j] = numbers[j+1];
                weights[j] = weights[j+1];
                numbers[j+1] = temp_num;
                weights[j+1] = temp_weight;
            }
        }
    }

    for(int i=0; i<count; i++) {
        printf("<%d,%d>", numbers[i], weights[i]);
        if(i < count-1) printf(", ");
    }

    return 0;
}
