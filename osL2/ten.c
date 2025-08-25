#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <math.h>

int *numbers;
int n;
double avg, stddev, max_val;

void* calculate_average(void* arg) {
    double sum = 0;
    for(int i = 0; i < n; i++) {
        sum += numbers[i];
    }
    avg = sum / n;
    return NULL;
}

void* calculate_stddev(void* arg) {
    double sum = 0;
    for(int i = 0; i < n; i++) {
        sum += (numbers[i] - avg) * (numbers[i] - avg);
    }
    stddev = sqrt(sum / n);
    return NULL;
}

void* find_maximum(void* arg) {
    max_val = numbers[0];
    for(int i = 1; i < n; i++) {
        if(numbers[i] > max_val) {
            max_val = numbers[i];
        }
    }
    return NULL;
}

int main(int argc, char* argv[]) {
    printf("Register No: 24BYB1095\n");
    n = argc - 1;
    numbers = malloc(n * sizeof(int));
    
    for(int i = 0; i < n; i++) {
        numbers[i] = atoi(argv[i + 1]);
    }
    
    pthread_t t1, t2, t3;
    
    pthread_create(&t1, NULL, calculate_average, NULL);
    pthread_join(t1, NULL);
    
    pthread_create(&t2, NULL, calculate_stddev, NULL);
    pthread_create(&t3, NULL, find_maximum, NULL);
    
    pthread_join(t2, NULL);
    pthread_join(t3, NULL);
    
    printf("Average: %.2f\n", avg);
    printf("Standard Deviation: %.2f\n", stddev);
    printf("Maximum: %.2f\n", max_val);
    
    free(numbers);
    return 0;
}
