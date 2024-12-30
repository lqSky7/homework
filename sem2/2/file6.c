#include <stdio.h>

int main() {
    int a, b = 0;
    scanf("%d", &a);
    
    for(int i = 1; i < a; i++) {
        if(a % i == 0) {
            b += i;
        }
    }
    
    if(b == a) {
        printf("Perfect number");
    } else {
        printf("Not a perfect number");
    }
    return 0;
}