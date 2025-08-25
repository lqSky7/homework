#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    printf("Register No: 24BYB1095\n");
    int N = 3;
    int child_count = 0;
    
    for(int i = 0; i < N; i++) {
        if(fork() == 0) {
            printf("Child PID: %d\n", getpid());
            return 0;
        }
        child_count++;
    }
    
    for(int i = 0; i < N; i++) {
        wait(NULL);
    }
    printf("Number of child processes: %d\n", child_count);
    return 0;
}
