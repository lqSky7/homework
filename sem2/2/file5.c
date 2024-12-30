#include <stdio.h>
#include <stdlib.h>

int main() {
    int a, b, c;
    scanf("%d", &a);
    
    if(a < 10) {
        printf("Invalid Input");
        return 0;
    }
    
    b = a % 10;
    while(a >= 10) {
        a = a / 10;
    }
    c = a;
    
    printf("%d", abs(c - b));
    return 0;
}