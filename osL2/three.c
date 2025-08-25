#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main(int argc, char *argv[]) {
    printf("Register No: 24BYB1095\n");
    
    if(argc != 3) {
        printf("Usage: %s <command> <argument>\n", argv[0]);
        return 1;
    }
    
    pid_t pid = fork();
    
    if(pid == 0) {
        execl(argv[1], argv[1], argv[2], NULL);
        printf("Error executing command\n");
        return 1;
    } else {
        wait(NULL);
        printf("Command executed\n");
    }
    return 0;
}
