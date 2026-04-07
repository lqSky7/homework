#include <stdio.h>
#include <string.h>

#define MAX 256

static int is_binary(const char *s) {
    if (!s || !*s) return 0;
    for (int i = 0; s[i]; i++) if (s[i] != '0' && s[i] != '1') return 0;
    return 1;
}

static int is_power_of_two(int x) { return x && !(x & (x - 1)); }

static void sender_mode(void) {
    char data[MAX];
    printf("Enter data bits: ");
    scanf("%255s", data);
    if (!is_binary(data)) {
        printf("Invalid binary data.\n");
        return;
    }

    int m = (int)strlen(data);
    int r = 0;
    while ((1 << r) < (m + r + 1)) r++;
    int n = m + r;

    int code[MAX] = {0};
    int j = 0;
    for (int i = 1; i <= n; i++) {
        if (!is_power_of_two(i)) code[i] = data[m - 1 - j++] - '0';
    }

    for (int p = 0; p < r; p++) {
        int pos = 1 << p;
        int parity = 0;
        for (int i = 1; i <= n; i++) if (i & pos) parity ^= code[i];
        code[pos] = parity;
    }

    printf("Encoded Hamming code: ");
    for (int i = n; i >= 1; i--) printf("%d", code[i]);
    printf("\nParity bits used: %d\n", r);
}

static void receiver_mode(void) {
    char recv[MAX];
    printf("Enter received Hamming code bits: ");
    scanf("%255s", recv);
    if (!is_binary(recv)) {
        printf("Invalid binary data.\n");
        return;
    }

    int n = (int)strlen(recv);
    int code[MAX] = {0};
    for (int i = 1; i <= n; i++) code[i] = recv[n - i] - '0';

    int r = 0;
    while ((1 << r) <= n) r++;
    int syndrome = 0;

    for (int p = 0; p < r; p++) {
        int pos = 1 << p;
        int parity = 0;
        for (int i = 1; i <= n; i++) if (i & pos) parity ^= code[i];
        if (parity) syndrome += pos;
    }

    if (syndrome == 0) {
        printf("No error detected.\n");
    } else {
        printf("Error detected at bit position: %d\n", syndrome);
        if (syndrome >= 1 && syndrome <= n) code[syndrome] ^= 1;
        printf("Corrected code: ");
        for (int i = n; i >= 1; i--) printf("%d", code[i]);
        printf("\n");
    }

    printf("Extracted data bits: ");
    for (int i = n; i >= 1; i--) if ((i & (i - 1)) != 0) printf("%d", code[i]);
    printf("\n");
}

int main(void) {
    int option;
    printf("Hamming Code\n1. Sender\n2. Receiver\nChoose option: ");
    if (scanf("%d", &option) != 1) return 1;

    if (option == 1) sender_mode();
    else if (option == 2) receiver_mode();
    else printf("Invalid option.\n");

    return 0;
}
