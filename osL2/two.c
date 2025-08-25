#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>

int main() {
    printf("Register No: 24BYB1095\n");
    pid_t pid = fork();
    
    if(pid == 0) {
        sleep(30);
    } else {
        sleep(2);
        kill(pid, SIGTERM);
        printf("Child process terminated\n");
        wait(NULL);
    }
    return 0;
}
