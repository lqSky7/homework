#include <stdio.h>

#define MAXV 100
#define MAXE 1000
#define INF 1000000000

typedef struct {
    int u, v, w;
} Edge;

int main(void) {
    int n, e, src;
    Edge edges[MAXE];
    int dist[MAXV], parent[MAXV];

    printf("Enter number of vertices: ");
    scanf("%d", &n);
    printf("Enter number of edges: ");
    scanf("%d", &e);

    if (n <= 0 || n > MAXV || e < 0 || e > MAXE) return 1;

    printf("Enter edges as: source destination weight\n");
    for (int i = 0; i < e; i++) {
        scanf("%d %d %d", &edges[i].u, &edges[i].v, &edges[i].w);
    }

    printf("Enter source vertex: ");
    scanf("%d", &src);

    for (int i = 0; i < n; i++) {
        dist[i] = INF;
        parent[i] = -1;
    }
    dist[src] = 0;

    for (int i = 1; i <= n - 1; i++) {
        for (int j = 0; j < e; j++) {
            int u = edges[j].u, v = edges[j].v, w = edges[j].w;
            if (u >= 0 && u < n && v >= 0 && v < n && dist[u] < INF && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                parent[v] = u;
            }
        }
    }

    int hasNegativeCycle = 0;
    for (int j = 0; j < e; j++) {
        int u = edges[j].u, v = edges[j].v, w = edges[j].w;
        if (u >= 0 && u < n && v >= 0 && v < n && dist[u] < INF && dist[u] + w < dist[v]) {
            hasNegativeCycle = 1;
            break;
        }
    }

    if (hasNegativeCycle) {
        printf("Graph contains a negative weight cycle.\n");
        return 0;
    }

    printf("\nShortest path costs from source %d:\n", src);
    for (int i = 0; i < n; i++) {
        if (dist[i] >= INF) printf("To %d: Unreachable\n", i);
        else printf("To %d: %d\n", i, dist[i]);
    }

    return 0;
}
