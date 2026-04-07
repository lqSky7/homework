#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <unistd.h>

#define MAX_CLIENTS 32
#define BUF 512

typedef struct {
    struct sockaddr_in addr;
    char name[32];
    int active;
} Client;

static void sanitize_name(char *name) {
    int w = 0;
    for (int r = 0; name[r]; r++) {
        unsigned char ch = (unsigned char)name[r];
        if (ch >= 32 && ch <= 126) name[w++] = (char)ch;
    }
    name[w] = '\0';
}

static int same_client(struct sockaddr_in *a, struct sockaddr_in *b) {
    return a->sin_addr.s_addr == b->sin_addr.s_addr && a->sin_port == b->sin_port;
}

int main(int argc, char *argv[]) {
    int port = (argc > 1) ? atoi(argv[1]) : 9003;
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return 1;

    struct sockaddr_in server = {0};
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons(port);

    if (bind(fd, (struct sockaddr *)&server, sizeof(server)) < 0) return 1;

    Client clients[MAX_CLIENTS] = {0};
    printf("UDP multiuser chat server listening on %d\n", port);

    while (1) {
        char buf[BUF];
        struct sockaddr_in from;
        socklen_t flen = sizeof(from);
        int n = recvfrom(fd, buf, sizeof(buf) - 1, 0, (struct sockaddr *)&from, &flen);
        if (n <= 0) continue;
        buf[n] = '\0';

        int idx = -1;
        for (int i = 0; i < MAX_CLIENTS; i++) {
            if (clients[i].active && same_client(&clients[i].addr, &from)) { idx = i; break; }
        }

        if (strncmp(buf, "/join ", 6) == 0) {
            if (idx == -1) {
                for (int i = 0; i < MAX_CLIENTS; i++) {
                    if (!clients[i].active) {
                        clients[i].active = 1;
                        clients[i].addr = from;
                        snprintf(clients[i].name, sizeof(clients[i].name), "%s", buf + 6);
                        clients[i].name[strcspn(clients[i].name, "\r\n")] = 0;
                        sanitize_name(clients[i].name);
                        if (clients[i].name[0] == '\0') {
                            snprintf(clients[i].name, sizeof(clients[i].name), "user%d", i + 1);
                        }
                        idx = i;
                        break;
                    }
                }
            }
        }

        char msg[BUF];
        const char *name = (idx >= 0) ? clients[idx].name : "unknown";
        if (strncmp(buf, "/join ", 6) == 0) {
            snprintf(msg, sizeof(msg), "%s joined the chat", name);
        } else {
            size_t max_body = sizeof(msg) - strlen(name) - 4;
            snprintf(msg, sizeof(msg), "%s: %.*s", name, (int)max_body, buf);
        }

        for (int i = 0; i < MAX_CLIENTS; i++) {
            if (clients[i].active) {
                sendto(fd, msg, strlen(msg), 0, (struct sockaddr *)&clients[i].addr, sizeof(clients[i].addr));
            }
        }
        printf("%s\n", msg);
    }
}
