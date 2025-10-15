#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define NUM_PHILOSOPHERS 5
#define EATING_TIMES 3

sem_t forks[NUM_PHILOSOPHERS];
pthread_mutex_t print_mutex;

void* philosopher(void* arg) {
    int id = *(int*)arg;
    int left = id;
    int right = (id + 1) % NUM_PHILOSOPHERS;
    
    for (int i = 0; i < EATING_TIMES; i++) {
        // Thinking
        pthread_mutex_lock(&print_mutex);
        printf("Philosopher %d: Thinking...\n", id);
        pthread_mutex_unlock(&print_mutex);
        sleep(rand() % 3 + 1);
        
        // Hungry - try to pick up forks
        pthread_mutex_lock(&print_mutex);
        printf("Philosopher %d: Hungry, trying to pick up forks %d and %d\n", id, left, right);
        pthread_mutex_unlock(&print_mutex);
        
        // Asymmetric solution: last philosopher picks up right fork first
        if (id == NUM_PHILOSOPHERS - 1) {
            sem_wait(&forks[right]);
            pthread_mutex_lock(&print_mutex);
            printf("Philosopher %d: Picked up right fork %d\n", id, right);
            pthread_mutex_unlock(&print_mutex);
            
            sem_wait(&forks[left]);
            pthread_mutex_lock(&print_mutex);
            printf("Philosopher %d: Picked up left fork %d\n", id, left);
            pthread_mutex_unlock(&print_mutex);
        } else {
            sem_wait(&forks[left]);
            pthread_mutex_lock(&print_mutex);
            printf("Philosopher %d: Picked up left fork %d\n", id, left);
            pthread_mutex_unlock(&print_mutex);
            
            sem_wait(&forks[right]);
            pthread_mutex_lock(&print_mutex);
            printf("Philosopher %d: Picked up right fork %d\n", id, right);
            pthread_mutex_unlock(&print_mutex);
        }
        
        // Eating
        pthread_mutex_lock(&print_mutex);
        printf("Philosopher %d: EATING (meal %d/%d)\n", id, i+1, EATING_TIMES);
        pthread_mutex_unlock(&print_mutex);
        sleep(rand() % 2 + 1);
        
        // Put down forks
        sem_post(&forks[left]);
        sem_post(&forks[right]);
        
        pthread_mutex_lock(&print_mutex);
        printf("Philosopher %d: Finished eating, put down forks\n", id);
        pthread_mutex_unlock(&print_mutex);
    }
    
    pthread_mutex_lock(&print_mutex);
    printf("Philosopher %d: Done eating all meals!\n", id);
    pthread_mutex_unlock(&print_mutex);
    
    return NULL;
}

int main() {
    pthread_t philosophers[NUM_PHILOSOPHERS];
    int ids[NUM_PHILOSOPHERS];
    
    printf("========================================\n");
    printf("  DINING PHILOSOPHERS PROBLEM\n");
    printf("========================================\n");
    printf("Number of Philosophers: %d\n", NUM_PHILOSOPHERS);
    printf("Meals per Philosopher: %d\n", EATING_TIMES);
    printf("Deadlock Prevention: Asymmetric solution\n");
    printf("========================================\n\n");
    
    // Initialize semaphores for forks
    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        sem_init(&forks[i], 0, 1);
    }
    pthread_mutex_init(&print_mutex, NULL);
    
    // Create philosopher threads
    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        ids[i] = i;
        pthread_create(&philosophers[i], NULL, philosopher, &ids[i]);
    }
    
    // Wait for all philosophers to finish
    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        pthread_join(philosophers[i], NULL);
    }
    
    // Cleanup
    for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
        sem_destroy(&forks[i]);
    }
    pthread_mutex_destroy(&print_mutex);
    
    printf("\n========================================\n");
    printf("All philosophers finished dining!\n");
    printf("========================================\n");
    
    return 0;
}

