#include <stdio.h>

int main() {
    int a, b, c = 0, d = 0;
    scanf("%d", &a);
    
    for(int i = 0; i < a; i++) {
        scanf("%d", &b);
        if(b % 2 == 0) {
            c++;
        } else {
            d++;
        }
    }
    printf("%d %d", c, d);
    return 0;
}