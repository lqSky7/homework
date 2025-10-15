#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define BUFFER_SIZE 5
#define NUM_ITEMS 10

int buffer[BUFFER_SIZE];
int in = 0, out = 0;

sem_t empty, full;
pthread_mutex_t mutex;

void print_buffer() {
    printf("Buffer: [");
    for (int i = 0; i < BUFFER_SIZE; i++) {
        if ((i >= out && i < in) || (in < out && (i >= out || i < in))) {
            printf("%d ", buffer[i]);
        } else {
            printf("- ");
        }
    }
    printf("]\n");
}

void* producer(void* arg) {
    for (int i = 0; i < NUM_ITEMS; i++) {
        int item = rand() % 100;
        
        sem_wait(&empty);
        pthread_mutex_lock(&mutex);
        
        buffer[in] = item;
        printf("Producer: Added %d at position %d\n", item, in);
        in = (in + 1) % BUFFER_SIZE;
        print_buffer();
        
        pthread_mutex_unlock(&mutex);
        sem_post(&full);
        
        sleep(1);
    }
    printf("Producer: Done\n");
    return NULL;
}

void* consumer(void* arg) {
    for (int i = 0; i < NUM_ITEMS; i++) {
        sem_wait(&full);
        pthread_mutex_lock(&mutex);
        
        int item = buffer[out];
        printf("Consumer: Removed %d from position %d\n", item, out);
        out = (out + 1) % BUFFER_SIZE;
        print_buffer();
        
        pthread_mutex_unlock(&mutex);
        sem_post(&empty);
        
        sleep(2);
    }
    printf("Consumer: Done\n");
    return NULL;
}

int main() {
    pthread_t prod, cons;
    
    printf("=== Producer-Consumer with Semaphores ===\n");
    printf("Buffer Size: %d, Items: %d\n\n", BUFFER_SIZE, NUM_ITEMS);
    
    sem_init(&empty, 0, BUFFER_SIZE);
    sem_init(&full, 0, 0);
    pthread_mutex_init(&mutex, NULL);
    
    pthread_create(&prod, NULL, producer, NULL);
    pthread_create(&cons, NULL, consumer, NULL);
    
    pthread_join(prod, NULL);
    pthread_join(cons, NULL);
    
    sem_destroy(&empty);
    sem_destroy(&full);
    pthread_mutex_destroy(&mutex);
    
    printf("\nProgram finished.\n");
    return 0;
}

