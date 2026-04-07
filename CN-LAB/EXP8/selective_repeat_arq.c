#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>

#define BUF 128
#define MAXF 512

static void receiver_mode(int port, int ack_drop) {
    int sfd = socket(AF_INET, SOCK_STREAM, 0), cfd;
    int opt = 1;
    setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    bind(sfd, (struct sockaddr *)&addr, sizeof(addr));
    listen(sfd, 1);
    cfd = accept(sfd, NULL, NULL);

    char buf[BUF], ack[BUF];
    int total = 0, window = 0;
    int received[MAXF] = {0};
    srand((unsigned)time(NULL));

    while (1) {
        int n = recv(cfd, buf, sizeof(buf) - 1, 0);
        if (n <= 0) break;
        buf[n] = '\0';

        if (sscanf(buf, "START %d %d", &total, &window) == 2) {
            if (total > MAXF) total = MAXF;
            printf("Session started: total=%d window=%d\n", total, window);
            continue;
        }
        if (strcmp(buf, "END") == 0) break;

        int seq;
        if (sscanf(buf, "FRAME %d", &seq) != 1) continue;
        if (seq >= 0 && seq < total) {
            received[seq] = 1;
            printf("Received frame %d\n", seq);
        }

        if ((rand() % 100) >= ack_drop) {
            snprintf(ack, sizeof(ack), "ACK %d", seq);
            send(cfd, ack, strlen(ack), 0);
            printf("Sent ACK %d\n", seq);
        } else {
            printf("Dropped ACK %d\n", seq);
        }

        int done = 1;
        for (int i = 0; i < total; i++) {
            if (!received[i]) {
                done = 0;
            }
        }
        if (done) break;
    }

    close(cfd);
    close(sfd);
}

static void sender_mode(const char *ip, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in server = {0};
    server.sin_family = AF_INET;
    server.sin_port = htons(port);
    inet_pton(AF_INET, ip, &server.sin_addr);
    connect(fd, (struct sockaddr *)&server, sizeof(server));

    struct timeval tv = {2, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    int total, window;
    printf("Enter total frames: ");
    scanf("%d", &total);
    printf("Enter window size: ");
    scanf("%d", &window);
    if (total > MAXF) total = MAXF;

    int acked[MAXF] = {0};
    int sent[MAXF] = {0};
    char buf[BUF];

    snprintf(buf, sizeof(buf), "START %d %d", total, window);
    send(fd, buf, strlen(buf), 0);

    while (1) {
        int done = 1;
        for (int i = 0; i < total; i++) if (!acked[i]) done = 0;
        if (done) break;

        int base = 0;
        while (base < total && acked[base]) base++;

        for (int i = base; i < base + window && i < total; i++) {
            if (!acked[i] && !sent[i]) {
                snprintf(buf, sizeof(buf), "FRAME %d", i);
                send(fd, buf, strlen(buf), 0);
                sent[i] = 1;
                printf("Sent frame %d\n", i);
            }
        }

        int n = recv(fd, buf, sizeof(buf) - 1, 0);
        if (n > 0) {
            buf[n] = '\0';
            int ack;
            if (sscanf(buf, "ACK %d", &ack) == 1 && ack >= 0 && ack < total) {
                acked[ack] = 1;
                printf("Received ACK %d\n", ack);
            }
        } else {
            printf("Timeout. Retransmitting unacked frames in current window.\n");
            for (int i = base; i < base + window && i < total; i++) {
                if (!acked[i]) sent[i] = 0;
            }
        }
    }

    send(fd, "END", 3, 0);
    close(fd);
}

int main(void) {
    int mode;
    printf("Selective Repeat with socket programming\n1. Sender\n2. Receiver\nChoose: ");
    scanf("%d", &mode);

    if (mode == 1) {
        char ip[32]; int port;
        printf("Receiver IP: "); scanf("%31s", ip);
        printf("Receiver port: "); scanf("%d", &port);
        sender_mode(ip, port);
    } else if (mode == 2) {
        int port, drop;
        printf("Listen port: "); scanf("%d", &port);
        printf("ACK drop percentage (0-100): "); scanf("%d", &drop);
        receiver_mode(port, drop);
    } else {
        printf("Invalid choice.\n");
    }
    return 0;
}
