#include <stdio.h>
#include <unistd.h>

int main() {
    printf("Register No: 24BYB1095\n");
    pid_t pid = vfork();
    
    if(pid == 0) {
        printf("Child PID: %d\n", getpid());
        _exit(0);
    } else {
        printf("Parent PID: %d\n", getpid());
    }
    return 0;
}
