#include <stdio.h>
#include <unistd.h>

int main() {
    printf("Register No: 24BYB1095\n");
    pid_t pid = fork();
    
    if(pid == 0) {
        printf("Child PID: %d\n", getpid());
        printf("Parent PID: %d\n", getppid());
    } else {
        sleep(5);
        printf("Parent PID: %d\n", getpid());
    }
    return 0;
}
