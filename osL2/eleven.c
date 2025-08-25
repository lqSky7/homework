#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

int limit;
int *primes;
int prime_count = 0;

int is_prime(int n) {
    if(n < 2) return 0;
    for(int i = 2; i * i <= n; i++) {
        if(n % i == 0) return 0;
    }
    return 1;
}

void* find_primes(void* arg) {
    primes = malloc(limit * sizeof(int));
    for(int i = 2; i <= limit; i++) {
        if(is_prime(i)) {
            primes[prime_count++] = i;
        }
    }
    return NULL;
}

int main(int argc, char* argv[]) {
    printf("Register No: 24BYB1095\n");
    limit = atoi(argv[1]);
    
    pthread_t tid;
    pthread_create(&tid, NULL, find_primes, NULL);
    pthread_join(tid, NULL);
    
    printf("Prime numbers up to %d: ", limit);
    for(int i = 0; i < prime_count; i++) {
        printf("%d ", primes[i]);
    }
    printf("\n");
    
    free(primes);
    return 0;
}
