#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>
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
    
    // Wait for reader to write input
    while(data->ready != 1);
    
    printf("Writer: Received n = %d\n", data->n);
    
    // Calculate Fibonacci series
    if(data->n >= 1) data->fib[0] = 0;
    if(data->n >= 2) data->fib[1] = 1;
    
    for(int i = 2; i < data->n; i++) {
        data->fib[i] = data->fib[i-1] + data->fib[i-2];
    }
    
    printf("Writer: Fibonacci series calculated\n");
    
    // Signal reader that calculation is done
    data->ready = 2;
    
    shmdt(data);
    return 0;
}

