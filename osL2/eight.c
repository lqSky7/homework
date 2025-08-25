#include <stdio.h>
#include <unistd.h>
#include <string.h>

int main() {
    printf("Register No: 24BYB1095\n");
    int pipe1[2], pipe2[2];
    char msg[100], reply[100];
    
    pipe(pipe1);
    pipe(pipe2);
    
    if(fork() == 0) {
        read(pipe1[0], msg, sizeof(msg));
        printf("Child received: %s\n", msg);
        sprintf(reply, "%s PID: %d", msg, getpid());
        write(pipe2[1], reply, strlen(reply) + 1);
    } else {
        strcpy(msg, "Hello from parent");
        write(pipe1[1], msg, strlen(msg) + 1);
        read(pipe2[0], reply, sizeof(reply));
        printf("Parent received: %s\n", reply);
    }
    return 0;
}
