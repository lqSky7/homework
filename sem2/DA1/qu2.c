#include <stdio.h>
#include <stdbool.h>

bool areAnagrams(char *str1, char *str2) {
    int count[26] = {0};

    // Count character frequencies
    while (*str1) {
        count[*str1 - 'a']++;
        str1++;
    }

    while (*str2) {
        count[*str2 - 'a']--;
        str2++;
    }

    // Check if all frequencies are zero
    for (int i = 0; i < 26; i++) {
        if (count[i] != 0) {
            return false;
        }
    }

    return true;
}

// Function to group anagrams
void groupAnagrams(char words[][50], int n) {
    bool grouped[100] = {false};

    printf("Anagrams:\n");

    for (int i = 0; i < n; i++) {
        if (grouped[i]) continue;

        // Start a new group
        printf("{%s", words[i]);
        grouped[i] = true;

        // Find anagrams for this word
        for (int j = i + 1; j < n; j++) {
            if (!grouped[j] && areAnagrams(words[i], words[j])) {
                printf(",%s", words[j]);
                grouped[j] = true;
            }
        }

        printf("}\n");
    }

    // Print words with no anagrams
    printf("Others:\n{");
    bool first = true;
    for (int i = 0; i < n; i++) {
        if (!grouped[i]) {
            if (!first) printf(",");
            printf("%s", words[i]);
            first = false;
        }
    }
    printf("}\n");
}

int main() {
    char words[][50] = {"tar", "rat", "banana", "art", "nabana", "baby"};
    int n = sizeof(words) / sizeof(words[0]);

    groupAnagrams(words, n);

    return 0;
}
