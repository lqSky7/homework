#include <stdio.h>
#include <stdbool.h>

void FIFO(int pages[], int n, int frames) {
    int frame[frames], faults = 0, idx = 0;
    for(int i=0; i<frames; i++) frame[i] = -1;
    
    printf("\nFIFO:\n");
    for(int i=0; i<n; i++) {
        bool found = false;
        for(int j=0; j<frames; j++) {
            if(frame[j] == pages[i]) {
                found = true;
                break;
            }
        }
        if(!found) {
            frame[idx] = pages[i];
            idx = (idx + 1) % frames;
            faults++;
            printf("Page %d: Fault %d\n", pages[i], faults);
        }
    }
    printf("Total Faults: %d\n", faults);
}

void LRU(int pages[], int n, int frames) {
    int frame[frames], time[frames], faults = 0;
    for(int i=0; i<frames; i++) frame[i] = -1;
    
    printf("\nLRU:\n");
    for(int i=0; i<n; i++) {
        bool found = false;
        int pos = -1;
        for(int j=0; j<frames; j++) {
            if(frame[j] == pages[i]) {
                found = true;
                time[j] = i;
                break;
            }
        }
        if(!found) {
            int min = 0;
            for(int j=0; j<frames; j++) {
                if(frame[j] == -1) {
                    pos = j;
                    break;
                }
                if(time[j] < time[min]) min = j;
            }
            if(pos == -1) pos = min;
            frame[pos] = pages[i];
            time[pos] = i;
            faults++;
            printf("Page %d: Fault %d\n", pages[i], faults);
        } else {
            for(int j=0; j<frames; j++)
                if(frame[j] == pages[i]) time[j] = i;
        }
    }
    printf("Total Faults: %d\n", faults);
}

void Optimal(int pages[], int n, int frames) {
    int frame[frames], faults = 0;
    for(int i=0; i<frames; i++) frame[i] = -1;
    
    printf("\nOptimal:\n");
    for(int i=0; i<n; i++) {
        bool found = false;
        for(int j=0; j<frames; j++) {
            if(frame[j] == pages[i]) {
                found = true;
                break;
            }
        }
        if(!found) {
            int pos = -1, farthest = i;
            for(int j=0; j<frames; j++) {
                if(frame[j] == -1) {
                    pos = j;
                    break;
                }
                int k;
                for(k=i+1; k<n; k++)
                    if(frame[j] == pages[k]) break;
                if(k > farthest) {
                    farthest = k;
                    pos = j;
                }
            }
            frame[pos] = pages[i];
            faults++;
            printf("Page %d: Fault %d\n", pages[i], faults);
        }
    }
    printf("Total Faults: %d\n", faults);
}

int main() {
    int pages[] = {3,1,4,2,5,4,1,3,5,2,0,1,1,0,2,3,4,5,0,1};
    int n = 20, frames = 3;
    
    FIFO(pages, n, frames);
    LRU(pages, n, frames);
    Optimal(pages, n, frames);
    
    return 0;
}

