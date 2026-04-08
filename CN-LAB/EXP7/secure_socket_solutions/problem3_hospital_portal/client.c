#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define BUF 768

static int send_all(int fd, const char *buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = send(fd, buf + off, len - off, 0);
        if (n <= 0) return -1;
        off += (size_t)n;
    }
    return 0;
}

static void trim_newline(char *s) {
    size_t n = strlen(s);
    if (n && s[n - 1] == '\n') s[n - 1] = '\0';
}

int main(int argc, char *argv[]) {
    const char *ip = (argc > 1) ? argv[1] : "127.0.0.1";
    int port = (argc > 2) ? atoi(argv[2]) : 9104;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 1;
    struct sockaddr_in server = {0};
    server.sin_family = AF_INET;
    server.sin_port = htons(port);
    inet_pton(AF_INET, ip, &server.sin_addr);
    if (connect(fd, (struct sockaddr *)&server, sizeof(server)) < 0) return 1;

    char pid[64], user[128], pass[128], req[BUF], resp[BUF];
    printf("Patient ID (6 digits): ");
    if (!fgets(pid, sizeof(pid), stdin)) return 1;
    printf("Username: ");
    if (!fgets(user, sizeof(user), stdin)) return 1;
    printf("Password: ");
    if (!fgets(pass, sizeof(pass), stdin)) return 1;

    trim_newline(pid);
    trim_newline(user);
    trim_newline(pass);
    snprintf(req, sizeof(req), "%s|%s|%s\n", pid, user, pass);
    if (send_all(fd, req, strlen(req)) < 0) return 1;

    int n = recv(fd, resp, sizeof(resp) - 1, 0);
    if (n > 0) {
        resp[n] = '\0';
        printf("%s", resp);
    }
    close(fd);
    return 0;
}
