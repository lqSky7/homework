#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>

#define BUF 128

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

    int expected = 0;
    char buf[BUF], ack[BUF];
    srand((unsigned)time(NULL));

    while (1) {
        int n = recv(cfd, buf, sizeof(buf) - 1, 0);
        if (n <= 0) break;
        buf[n] = '\0';
        if (strcmp(buf, "END") == 0) break;

        int seq;
        if (sscanf(buf, "FRAME %d", &seq) != 1) continue;

        if (seq == expected) {
            printf("Received expected frame %d\n", seq);
            expected++;
        } else {
            printf("Received out-of-order frame %d, expected %d\n", seq, expected);
        }

        if ((rand() % 100) >= ack_drop) {
            if (expected == 0) snprintf(ack, sizeof(ack), "ACK NONE");
            else snprintf(ack, sizeof(ack), "ACK %d", expected - 1);
            send(cfd, ack, strlen(ack), 0);
            printf("Sent %s\n", ack);
        } else {
            printf("Dropped ACK for cumulative %d\n", expected - 1);
        }
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
    printf("Enter total number of frames: ");
    scanf("%d", &total);
    printf("Enter window size: ");
    scanf("%d", &window);

    int base = 0, next = 0;
    char buf[BUF];

    while (base < total) {
        while (next < base + window && next < total) {
            snprintf(buf, sizeof(buf), "FRAME %d", next);
            send(fd, buf, strlen(buf), 0);
            printf("Sent frame %d\n", next);
            next++;
        }

        int n = recv(fd, buf, sizeof(buf) - 1, 0);
        if (n > 0) {
            buf[n] = '\0';
            int ack;
            if (strcmp(buf, "ACK NONE") == 0) {
                printf("Received ACK NONE (no in-order frame yet)\n");
            } else if (sscanf(buf, "ACK %d", &ack) == 1) {
                printf("Received cumulative ACK %d\n", ack);
                if (ack >= base) base = ack + 1;
            }
        } else {
            printf("Timeout. Retransmitting from frame %d\n", base);
            next = base;
        }
    }

    send(fd, "END", 3, 0);
    close(fd);
}

int main(void) {
    int mode;
    printf("Go-Back-N with socket programming\n1. Sender\n2. Receiver\nChoose: ");
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
