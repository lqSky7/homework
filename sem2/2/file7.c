#include <stdio.h>
#include <string.h>

int main() {
    char a[10];
    int b = 0, c = 0;
    scanf("%s", a);
    
    if(strlen(a) != 5) {
        printf("Length Invalid");
        return 0;
    }
    
    for(int i = 0; i < strlen(a)-1; i++) {
        if(a[i] == a[i+1]) {
            b = 1;
            c = i + 1;
            break;
        }
    }
    
    printf("Validated PIN: ");
    for(int i = 0; i < strlen(a); i++) {
        if(i != c) printf("%c", a[i]);
    }
    printf("\n");
    
    if(b) {
        printf("Repeated occurrence");
    } else {
        printf("Valid PIN");
    }
    return 0;
}