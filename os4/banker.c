#include <stdio.h>
#include <stdbool.h>

#define P 5  // Number of processes
#define R 4  // Number of resource types

// Function to calculate the Need matrix
void calculateNeed(int need[P][R], int max[P][R], int alloc[P][R]) {
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < R; j++) {
            need[i][j] = max[i][j] - alloc[i][j];
        }
    }
}

// Function to check if the system is in a safe state
bool isSafe(int processes[], int avail[], int max[P][R], int alloc[P][R]) {
    int need[P][R];
    calculateNeed(need, max, alloc);

    bool finish[P];
    for (int i = 0; i < P; i++) {
        finish[i] = false;
    }

    int safeSeq[P];
    int work[R];

    // Initialize work as available
    for (int i = 0; i < R; i++) {
        work[i] = avail[i];
    }

    int count = 0;
    while (count < P) {
        bool found = false;

        for (int p = 0; p < P; p++) {
            if (!finish[p]) {
                int j;

                // Check if all resources can be allocated
                for (j = 0; j < R; j++) {
                    if (need[p][j] > work[j]) {
                        break;
                    }
                }

                // If all resources can be allocated
                if (j == R) {
                    // Add allocated resources to available/work
                    for (int k = 0; k < R; k++) {
                        work[k] += alloc[p][k];
                    }

                    safeSeq[count++] = p;
                    finish[p] = true;
                    found = true;
                }
            }
        }

        if (!found) {
            printf("System is not in safe state (Deadlock possible)\n");
            return false;
        }
    }

    // If system is in safe state, print the safe sequence
    printf("System is in safe state.\n");
    printf("Safe sequence is: ");
    for (int i = 0; i < P; i++) {
        printf("P%d", safeSeq[i]);
        if (i != P - 1) {
            printf(" -> ");
        }
    }
    printf("\n");

    return true;
}

// Function to print matrices
void printMatrices(int alloc[P][R], int max[P][R], int avail[]) {
    int need[P][R];
    calculateNeed(need, max, alloc);

    printf("\n========== ALLOCATION MATRIX ==========\n");
    printf("Process   A   B   C   D\n");
    for (int i = 0; i < P; i++) {
        printf("  P%d      %d   %d   %d   %d\n", i, alloc[i][0], alloc[i][1], alloc[i][2], alloc[i][3]);
    }

    printf("\n========== MAXIMUM MATRIX ==========\n");
    printf("Process   A   B   C   D\n");
    for (int i = 0; i < P; i++) {
        printf("  P%d      %d   %d   %d   %d\n", i, max[i][0], max[i][1], max[i][2], max[i][3]);
    }

    printf("\n========== NEED MATRIX ==========\n");
    printf("Process   A   B   C   D\n");
    for (int i = 0; i < P; i++) {
        printf("  P%d      %d   %d   %d   %d\n", i, need[i][0], need[i][1], need[i][2], need[i][3]);
    }

    printf("\n========== AVAILABLE RESOURCES ==========\n");
    printf("A = %d, B = %d, C = %d, D = %d\n", avail[0], avail[1], avail[2], avail[3]);
}

// Function to request resources
bool requestResources(int process, int request[], int avail[], int max[P][R], int alloc[P][R]) {
    int need[P][R];
    calculateNeed(need, max, alloc);

    printf("\n--- Process P%d requesting resources ---\n", process);
    printf("Request: A=%d, B=%d, C=%d, D=%d\n", request[0], request[1], request[2], request[3]);

    // Check if request is less than or equal to need
    for (int i = 0; i < R; i++) {
        if (request[i] > need[process][i]) {
            printf("ERROR: Process has exceeded its maximum claim!\n");
            return false;
        }
    }

    // Check if request is less than or equal to available
    for (int i = 0; i < R; i++) {
        if (request[i] > avail[i]) {
            printf("ERROR: Resources not available. Process must wait.\n");
            return false;
        }
    }

    // Pretend to allocate requested resources
    for (int i = 0; i < R; i++) {
        avail[i] -= request[i];
        alloc[process][i] += request[i];
        need[process][i] -= request[i];
    }

    // Check if the new state is safe
    int processes[P];
    for (int i = 0; i < P; i++) {
        processes[i] = i;
    }

    if (isSafe(processes, avail, max, alloc)) {
        printf("Request GRANTED!\n");
        return true;
    } else {
        // Rollback the allocation
        for (int i = 0; i < R; i++) {
            avail[i] += request[i];
            alloc[process][i] -= request[i];
            need[process][i] += request[i];
        }
        printf("Request DENIED! (Would lead to unsafe state)\n");
        return false;
    }
}

int main() {
    int processes[P];
    for (int i = 0; i < P; i++) {
        processes[i] = i;
    }

    // Allocation Matrix
    int alloc[P][R] = {
        {0, 0, 1, 2},  // P0
        {1, 0, 0, 0},  // P1
        {1, 3, 5, 4},  // P2
        {0, 6, 3, 2},  // P3
        {0, 0, 1, 4}   // P4
    };

    // Maximum Matrix
    int max[P][R] = {
        {0, 0, 1, 2},  // P0
        {1, 7, 5, 0},  // P1
        {2, 3, 5, 6},  // P2
        {0, 6, 5, 2},  // P3
        {0, 6, 5, 6}   // P4
    };

    // Available Resources
    int avail[R] = {1, 5, 2, 0};

    printf("\n");
    printf("************************************************************\n");
    printf("*          BANKER'S ALGORITHM - DEADLOCK AVOIDANCE         *\n");
    printf("************************************************************\n");

    // Print initial state
    printf("\n========== INITIAL SYSTEM STATE ==========\n");
    printMatrices(alloc, max, avail);

    // Check initial safety
    printf("\n\n========== CHECKING INITIAL SAFETY ==========\n");
    isSafe(processes, avail, max, alloc);

    // Resource request example
    printf("\n\n========== RESOURCE ALLOCATION REQUEST ==========\n");

    // Take user input for resource request
    int choice;
    printf("\nDo you want to make a resource request? (1=Yes, 0=No): ");
    scanf("%d", &choice);

    while (choice == 1) {
        int proc_num;
        int request[R];

        printf("\nEnter process number (0-%d): ", P-1);
        scanf("%d", &proc_num);

        if (proc_num < 0 || proc_num >= P) {
            printf("Invalid process number!\n");
        } else {
            printf("Enter resource request for A, B, C, D: ");
            for (int i = 0; i < R; i++) {
                scanf("%d", &request[i]);
            }

            if (requestResources(proc_num, request, avail, max, alloc)) {
                printf("\n========== UPDATED SYSTEM STATE ==========\n");
                printMatrices(alloc, max, avail);
            }
        }

        printf("\nDo you want to make another request? (1=Yes, 0=No): ");
        scanf("%d", &choice);
    }

    // Final state
    printf("\n\n========== FINAL SYSTEM STATE ==========\n");
    printMatrices(alloc, max, avail);

    printf("\n========== FINAL SAFETY CHECK ==========\n");
    isSafe(processes, avail, max, alloc);

    printf("\n************************************************************\n");
    printf("*              END OF BANKER'S ALGORITHM                   *\n");
    printf("************************************************************\n\n");

    return 0;
}
