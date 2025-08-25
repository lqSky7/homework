#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

int *numbers;
int n;
int sum = 0;

void* sort_descending(void* arg) {
    for(int i = 0; i < n - 1; i++) {
        for(int j = 0; j < n - i - 1; j++) {
            if(numbers[j] < numbers[j + 1]) {
                int temp = numbers[j];
                numbers[j] = numbers[j + 1];
                numbers[j + 1] = temp;
            }
        }
    }
    return NULL;
}

void* calculate_sum(void* arg) {
    for(int i = 0; i < n; i++) {
        sum += numbers[i];
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
    
    pthread_t t1, t2;
    
    pthread_create(&t1, NULL, sort_descending, NULL);
    pthread_create(&t2, NULL, calculate_sum, NULL);
    
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    
    printf("Sorted (descending): ");
    for(int i = 0; i < n; i++) {
        printf("%d ", numbers[i]);
    }
    printf("\nSum: %d\n", sum);
    
    free(numbers);
    return 0;
}
