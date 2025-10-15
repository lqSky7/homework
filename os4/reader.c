#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdlib.h>

struct shared_data {
    int n;
    int fib[100];
    int ready;
};

int main() {
    key_t key = ftok("shmfile", 65);
    int shmid = shmget(key, 1024, 0666 | IPC_CREAT);
    struct shared_data *data = (struct shared_data*) shmat(shmid, NULL, 0);
    
    // Initialize
    data->ready = 0;
    
    // Read input
    printf("Enter number of terms: ");
    scanf("%d", &data->n);
    
    // Write to shared memory and signal writer
    data->ready = 1;
    printf("Reader: Data written to shared memory\n");
    
    // Wait for writer to calculate Fibonacci
    while(data->ready != 2);
    
    // Print results
    printf("\nReader: Fibonacci series:\n");
    for(int i = 0; i < data->n; i++) {
        printf("%d ", data->fib[i]);
    }
    printf("\n");
    
    shmdt(data);
    shmctl(shmid, IPC_RMID, NULL);
    return 0;
}

