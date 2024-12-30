#include <stdio.h>

int main() {
    int a;
    double b;
    scanf("%d", &a);
    
    if (a < 15000) {
        b = a + (0.15 * a) + (0.90 * a);
    } else {
        b = a + 5000 + (0.98 * a);
    }
    
    printf("%.2lf", b);
    return 0;
}