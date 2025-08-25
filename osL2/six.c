#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <time.h>

int main() {
    printf("Register No: 24BYB1095\n");
    clock_t start, end;
    pid_t pid = fork();
    
    if(pid == 0) {
        start = clock();
        int n = 5, sum = 0;
        for(int i = 1; i <= n; i++) {
            sum += i;
        }
        end = clock();
        printf("Child Sum: %d, Time: %f\n", sum, ((double)(end-start))/CLOCKS_PER_SEC);
    } else {
        start = clock();
        int n = 5, fact = 1;
        for(int i = 1; i <= n; i++) {
            fact *= i;
        }
        end = clock();
        printf("Parent Factorial: %d, Time: %f\n", fact, ((double)(end-start))/CLOCKS_PER_SEC);
        wait(NULL);
    }
    return 0;
}
