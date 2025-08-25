#include <stdio.h>
#include <unistd.h>

int main() {
    printf("Register No: 24BYB1095\n");
    pid_t pid = fork();
    
    if(pid == 0) {
        sleep(2);
        printf("Child PID: %d\n", getpid());
        printf("Parent PID: %d\n", getppid());
        printf("New Parent PID: %d\n", getppid());
    } else {
        printf("Parent exiting...\n");
    }
    return 0;
}
