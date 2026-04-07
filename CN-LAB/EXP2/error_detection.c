#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_BITS 1024
#define MAX_WORDS 128

static int is_binary(const char *s) {
    if (!s || !*s) return 0;
    for (size_t i = 0; s[i]; i++) if (s[i] != '0' && s[i] != '1') return 0;
    return 1;
}

static int ones_count(const char *s) {
    int c = 0;
    for (size_t i = 0; s[i]; i++) if (s[i] == '1') c++;
    return c;
}

static int vrc_parity_bit(const char *data, int odd) {
    int ones = ones_count(data);
    if (odd) return (ones % 2 == 0) ? 1 : 0;
    return (ones % 2 == 0) ? 0 : 1;
}

static void run_vrc(void) {
    char mode[16], parity[16], data[MAX_BITS];
    int odd;

    printf("Do you want to send or receive the data? ");
    scanf("%15s", mode);
    printf("Which parity generator do you want to use odd/even ? ");
    scanf("%15s", parity);
    odd = (tolower((unsigned char)parity[0]) == 'o');

    if (tolower((unsigned char)mode[0]) == 's') {
        printf("Enter the original data: ");
        scanf("%1023s", data);
        if (!is_binary(data)) {
            printf("Invalid input. Enter only 0 and 1.\n");
            return;
        }
        int p = vrc_parity_bit(data, odd);
        printf("\nOutput/Result:\n");
        printf("Original Input: %s\n", data);
        printf("Used Parity Generator: %s\n", odd ? "odd" : "even");
        printf("Generated Redundant bit : %d\n", p);
        printf("Bits to be transmitted : %s%d\n", data, p);
    } else {
        printf("Enter the received data (including parity bit): ");
        scanf("%1023s", data);
        if (!is_binary(data) || strlen(data) < 2) {
            printf("Invalid input.\n");
            return;
        }
        int ok = odd ? (ones_count(data) % 2 == 1) : (ones_count(data) % 2 == 0);
        printf("\nOutput/Result:\n");
        printf("Received Data: %s\n", data);
        printf("Used Parity Generator: %s\n", odd ? "odd" : "even");
        printf("Status: %s\n", ok ? "No error detected" : "Error detected");
    }
}

static int read_words(char words[MAX_WORDS][MAX_BITS], int *n, int *len, const char *msg) {
    printf("%s", msg);
    scanf("%d", n);
    if (*n <= 0 || *n > MAX_WORDS) return 0;
    printf("Enter each binary word on a new line:\n");
    for (int i = 0; i < *n; i++) {
        scanf("%1023s", words[i]);
        if (!is_binary(words[i])) return 0;
        if (i == 0) *len = (int)strlen(words[i]);
        if ((int)strlen(words[i]) != *len) return 0;
    }
    return 1;
}

static void run_lrc(void) {
    char mode[16], parity[16];
    int odd;

    printf("Do you want to send or receive the data? ");
    scanf("%15s", mode);
    printf("Which parity generator do you want to use odd/even ? ");
    scanf("%15s", parity);
    odd = (tolower((unsigned char)parity[0]) == 'o');

    char words[MAX_WORDS][MAX_BITS];
    int n = 0, len = 0;

    if (tolower((unsigned char)mode[0]) == 's') {
        if (!read_words(words, &n, &len, "Enter number of data words: ")) {
            printf("Invalid words.\n");
            return;
        }
        char lrc[MAX_BITS];
        for (int j = 0; j < len; j++) {
            int ones = 0;
            for (int i = 0; i < n; i++) if (words[i][j] == '1') ones++;
            lrc[j] = odd ? ((ones % 2 == 0) ? '1' : '0') : ((ones % 2 == 0) ? '0' : '1');
        }
        lrc[len] = '\0';

        printf("\nOutput/Result:\n");
        for (int i = 0; i < n; i++) printf("Data word %d: %s\n", i + 1, words[i]);
        printf("Generated LRC: %s\n", lrc);
        printf("Bits to be transmitted:\n");
        for (int i = 0; i < n; i++) printf("%s\n", words[i]);
        printf("%s\n", lrc);
    } else {
        if (!read_words(words, &n, &len, "Enter number of received words (including LRC row): ")) {
            printf("Invalid words.\n");
            return;
        }
        int ok = 1;
        for (int j = 0; j < len; j++) {
            int ones = 0;
            for (int i = 0; i < n; i++) if (words[i][j] == '1') ones++;
            if (odd && ones % 2 != 1) ok = 0;
            if (!odd && ones % 2 != 0) ok = 0;
        }
        printf("\nOutput/Result:\n");
        printf("Status: %s\n", ok ? "No error detected" : "Error detected");
    }
}

static void xor_division(const char *data, const char *gen, char *remainder) {
    int n = (int)strlen(data), g = (int)strlen(gen);
    char temp[MAX_BITS * 2];
    strcpy(temp, data);

    for (int i = 0; i <= n - g; i++) {
        if (temp[i] == '1') {
            for (int j = 0; j < g; j++) temp[i + j] = (temp[i + j] == gen[j]) ? '0' : '1';
        }
    }
    strcpy(remainder, temp + n - (g - 1));
}

static void run_crc(void) {
    char mode[16], data[MAX_BITS], gen[MAX_BITS], appended[MAX_BITS * 2], rem[MAX_BITS];
    printf("Do you want to send or receive the data? ");
    scanf("%15s", mode);

    if (tolower((unsigned char)mode[0]) == 's') {
        printf("Enter the original data (binary): ");
        scanf("%1023s", data);
        printf("Enter generator polynomial (binary): ");
        scanf("%1023s", gen);
        if (!is_binary(data) || !is_binary(gen) || strlen(gen) < 2 || gen[0] != '1') {
            printf("Invalid input.\n");
            return;
        }
        strcpy(appended, data);
        for (size_t i = 0; i < strlen(gen) - 1; i++) strcat(appended, "0");
        xor_division(appended, gen, rem);

        printf("\nOutput/Result:\n");
        printf("Original Input: %s\n", data);
        printf("Generator: %s\n", gen);
        printf("Generated Redundant bits (CRC remainder): %s\n", rem);
        printf("Bits to be transmitted: %s%s\n", data, rem);
    } else {
        printf("Enter received data (data + CRC bits): ");
        scanf("%1023s", data);
        printf("Enter generator polynomial (binary): ");
        scanf("%1023s", gen);
        if (!is_binary(data) || !is_binary(gen) || strlen(gen) < 2 || gen[0] != '1') {
            printf("Invalid input.\n");
            return;
        }
        xor_division(data, gen, rem);
        int ok = 1;
        for (size_t i = 0; rem[i]; i++) if (rem[i] != '0') ok = 0;
        printf("\nOutput/Result:\n");
        printf("Remainder after division: %s\n", rem);
        printf("Status: %s\n", ok ? "No error detected" : "Error detected");
    }
}

static unsigned int checksum_ones_complement(unsigned int words[], int n, int bits) {
    unsigned int mask = (1U << bits) - 1U;
    unsigned int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += words[i];
        while (sum >> bits) sum = (sum & mask) + (sum >> bits);
    }
    return (~sum) & mask;
}

static void run_checksum(void) {
    char mode[16];
    int n, bits;
    unsigned int words[MAX_WORDS];

    printf("Do you want to send or receive the data? ");
    scanf("%15s", mode);
    printf("Enter word size in bits (e.g., 8 or 16): ");
    scanf("%d", &bits);
    if (bits <= 1 || bits >= 31) {
        printf("Invalid word size.\n");
        return;
    }

    if (tolower((unsigned char)mode[0]) == 's') {
        printf("Enter number of data words: ");
        scanf("%d", &n);
        if (n <= 0 || n > MAX_WORDS) return;
        printf("Enter words in decimal format:\n");
        for (int i = 0; i < n; i++) scanf("%u", &words[i]);
        unsigned int cs = checksum_ones_complement(words, n, bits);
        printf("\nOutput/Result:\n");
        printf("Generated checksum: %u\n", cs);
        printf("Transmitted words: ");
        for (int i = 0; i < n; i++) printf("%u ", words[i]);
        printf("%u\n", cs);
    } else {
        printf("Enter number of received words (including checksum word): ");
        scanf("%d", &n);
        if (n <= 1 || n > MAX_WORDS) return;
        printf("Enter words in decimal format:\n");
        for (int i = 0; i < n; i++) scanf("%u", &words[i]);
        unsigned int cs = checksum_ones_complement(words, n, bits);
        printf("\nOutput/Result:\n");
        printf("Verification result value: %u\n", cs);
        printf("Status: %s\n", cs == 0 ? "No error detected" : "Error detected");
    }
}

int main(void) {
    int choice;
    printf("Choose any one of the following\n1. VRC\n2. LRC\n3. CRC\n4. Checksum\n");
    printf("User Input: ");
    if (scanf("%d", &choice) != 1) {
        printf("Invalid choice.\n");
        return 1;
    }

    switch (choice) {
        case 1: run_vrc(); break;
        case 2: run_lrc(); break;
        case 3: run_crc(); break;
        case 4: run_checksum(); break;
        default: printf("Invalid choice.\n");
    }
    return 0;
}
