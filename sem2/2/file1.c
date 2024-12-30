#include <stdio.h>

int main() {
    int a, b, c, d, e;
    int f = 0;
    float g;
    scanf("%d %d %d %d %d", &a, &b, &c, &d, &e);
    
    if (a < 2 || b < 2 || c < 2 || d < 2 || e < 2) {
        printf("No");
        return 0;
    }
    
    if (a == 5 || b == 5 || c == 5 || d == 5 || e == 5) {
        f = 1;
    }
    
    g = (a + b + c + d + e) / 5.0;
    
    if (g >= 4.0 && f == 1) {
        printf("Yes");
    } else {
        printf("No");
    }
    return 0;
}