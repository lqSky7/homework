#include <stdio.h>

int main() {
    int a, b, c, d, e;
    float f;
    scanf("%d %d %d %d %d %f", &a, &b, &c, &d, &e, &f);
    
    printf("%.2f", (f * 6) - (a + b + c + d + e));
    return 0;
}