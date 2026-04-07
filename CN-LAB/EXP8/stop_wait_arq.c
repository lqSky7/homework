#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>

#define BUF 128

static void run_receiver(int port, int ack_drop_percent) {
    int sfd = socket(AF_INET, SOCK_STREAM, 0), cfd;
    int opt = 1;
    setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(port);
    bind(sfd, (struct sockaddr *)&addr, sizeof(addr));
    listen(sfd, 1);
    cfd = accept(sfd, NULL, NULL);

    char buf[BUF], ack[BUF];
    int expected = 0;
    srand((unsigned)time(NULL));
    while (1) {
        int n = recv(cfd, buf, sizeof(buf) - 1, 0);
        if (n <= 0) break;
        buf[n] = '\0';
        if (strcmp(buf, "END") == 0) break;

        int seq;
        char payload[64];
        sscanf(buf, "%d %63s", &seq, payload);
        printf("Received frame seq=%d data=%s\n", seq, payload);

        if (seq == expected) expected ^= 1;
        if ((rand() % 100) >= ack_drop_percent) {
            snprintf(ack, sizeof(ack), "%d", seq);
            send(cfd, ack, strlen(ack), 0);
            printf("ACK sent for seq=%d\n", seq);
        } else {
            printf("ACK dropped for seq=%d (simulation)\n", seq);
        }
    }

    close(cfd); close(sfd);
}

static void run_sender(const char *ip, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in server = {0};
    server.sin_family = AF_INET; server.sin_port = htons(port); inet_pton(AF_INET, ip, &server.sin_addr);
    connect(fd, (struct sockaddr *)&server, sizeof(server));

    struct timeval tv = {2, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    int n;
    printf("Enter number of frames to send: ");
    scanf("%d", &n);

    int seq = 0;
    char frame[BUF], ack[BUF];
    for (int i = 0; i < n; i++) {
        while (1) {
            snprintf(frame, sizeof(frame), "%d FRAME%d", seq, i + 1);
            send(fd, frame, strlen(frame), 0);
            printf("Sent: %s\n", frame);

            int r = recv(fd, ack, sizeof(ack) - 1, 0);
            if (r > 0) {
                ack[r] = '\0';
                if (atoi(ack) == seq) {
                    printf("ACK received for seq=%d\n", seq);
                    seq ^= 1;
                    break;
                }
            } else {
                printf("Timeout. Resending frame %d\n", i + 1);
            }
        }
    }

    send(fd, "END", 3, 0);
    close(fd);
}

int main(void) {
    int mode;
    printf("Stop-and-Wait ARQ\n1. Sender\n2. Receiver\nChoose: ");
    scanf("%d", &mode);

    if (mode == 1) {
        char ip[32]; int port;
        printf("Receiver IP: "); scanf("%31s", ip);
        printf("Receiver port: "); scanf("%d", &port);
        run_sender(ip, port);
    } else if (mode == 2) {
        int port, drop;
        printf("Listen port: "); scanf("%d", &port);
        printf("ACK drop percentage (0-100): "); scanf("%d", &drop);
        run_receiver(port, drop);
    } else {
        printf("Invalid option.\n");
    }
    return 0;
}
