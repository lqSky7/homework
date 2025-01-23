#include <stdio.h>

// Function to sort the original array based on the weight array
void sortBasedOnWeights(int originalArray[], int weightArray[], int n) {
    // Create an index array to store the original indices
    int indices[n];
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }

    // Sort the indices based on the weight array (using bubble sort)
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (weightArray[indices[j]] > weightArray[indices[j + 1]]) {
                // Swap indices
                int temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }

    // Create temporary arrays to store the sorted original and weight arrays
    int sortedOriginal[n];
    int sortedWeights[n];

    // Populate the sorted arrays using the sorted indices
    for (int i = 0; i < n; i++) {
        sortedOriginal[i] = originalArray[indices[i]];
        sortedWeights[i] = weightArray[indices[i]];
    }

    // Copy the sorted arrays back to the original arrays
    for (int i = 0; i < n; i++) {
        originalArray[i] = sortedOriginal[i];
        weightArray[i] = sortedWeights[i];
    }
}

int main() {
    // Original array
    int originalArray[] = {10, 36, 54, 89, 12};
    // Weight array
    int weightArray[] = {3, 12, 3, 0, 7};
    int n = sizeof(originalArray) / sizeof(originalArray[0]);

    // Sort the original array based on the weight array
    sortBasedOnWeights(originalArray, weightArray, n);

    // Print the sorted original array and weight array
    printf("Sorted Original Array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", originalArray[i]);
    }
    printf("\n");

    printf("Sorted Weight Array:   ");
    for (int i = 0; i < n; i++) {
        printf("%d ", weightArray[i]);
    }
    printf("\n");

    return 0;
}
