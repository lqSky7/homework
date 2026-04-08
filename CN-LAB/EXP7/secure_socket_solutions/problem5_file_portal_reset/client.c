#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define BUF 1024

static void trim_newline(char *s) {
    size_t n = strlen(s);
    if (n && s[n - 1] == '\n') s[n - 1] = '\0';
}

int main(int argc, char *argv[]) {
    const char *ip = (argc > 1) ? argv[1] : "127.0.0.1";
    int port = (argc > 2) ? atoi(argv[2]) : 9106;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 1;
    struct sockaddr_in server = {0};
    server.sin_family = AF_INET;
    server.sin_port = htons(port);
    inet_pton(AF_INET, ip, &server.sin_addr);
    if (connect(fd, (struct sockaddr *)&server, sizeof(server)) < 0) return 1;

    char user[128], temp[128], newp[128], req[BUF], resp[BUF];
    printf("Username: ");
    if (!fgets(user, sizeof(user), stdin)) return 1;
    printf("Temporary password: ");
    if (!fgets(temp, sizeof(temp), stdin)) return 1;
    printf("New password: ");
    if (!fgets(newp, sizeof(newp), stdin)) return 1;

    trim_newline(user);
    trim_newline(temp);
    trim_newline(newp);
    snprintf(req, sizeof(req), "%s|%s|%s\n", user, temp, newp);
    send(fd, req, strlen(req), 0);

    int n = recv(fd, resp, sizeof(resp) - 1, 0);
    if (n > 0) {
        resp[n] = '\0';
        printf("%s", resp);
    }
    close(fd);
    return 0;
}

