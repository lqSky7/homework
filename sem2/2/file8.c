#include <stdio.h>

int main() {
    int a, b = 20;
    scanf("%d", &a);
    
    for(int i = 1; i <= a; i++) {
        printf("%d", b);
        if(i < a) printf(" ");
        b = b + (i * 40) + 4;
    }
    return 0;
}