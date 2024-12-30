#include <stdio.h>

int main() {
    int a;
    scanf("%d", &a);
    
    if (a % 10 == 5 || a % 10 == 7) {
        printf("Lucky Winner");
    } else {
        printf("Not a Lucky Winner");
    }
    return 0;
}