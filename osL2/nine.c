#include <stdio.h>
#include <pthread.h>

void* worker(void* arg) {
    printf("Worker Thread ID: %lu\n", pthread_self());
    return NULL;
}

int main() {
    printf("Register No: 24BYB1095\n");
    pthread_t tid;
    
    printf("Main Thread ID: %lu\n", pthread_self());
    pthread_create(&tid, NULL, worker, NULL);
    pthread_join(tid, NULL);
    return 0;
}
