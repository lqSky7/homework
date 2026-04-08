#include <arpa/inet.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define BUF 1024
#define USERS 3

static int send_all(int fd, const char *buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = send(fd, buf + off, len - off, 0);
        if (n <= 0) return -1;
        off += (size_t)n;
    }
    return 0;
}

typedef struct {
    char username[32];
    char temp_password[32];
    int reset_done;
    char transformed_username[64];
} UserRecord;

static UserRecord records[USERS] = {
    {"aliceaa", "Temp@123", 0, ""},
    {"bobbbbb", "Start#45", 0, ""},
    {"charlie", "Init$678", 0, ""}
};

static int only_alpha_min6(const char *s) {
    if (strlen(s) < 6) return 0;
    for (int i = 0; s[i]; i++) if (!isalpha((unsigned char)s[i])) return 0;
    return 1;
}

static int has_upper(const char *s) {
    for (int i = 0; s[i]; i++) if (isupper((unsigned char)s[i])) return 1;
    return 0;
}

static int has_digit(const char *s) {
    for (int i = 0; s[i]; i++) if (isdigit((unsigned char)s[i])) return 1;
    return 0;
}

static int has_special(const char *s) {
    for (int i = 0; s[i]; i++) if (!isalnum((unsigned char)s[i])) return 1;
    return 0;
}

static int find_user(const char *username) {
    for (int i = 0; i < USERS; i++) if (strcmp(records[i].username, username) == 0) return i;
    return -1;
}

static void reverse_append_123(const char *in, char *out, size_t out_sz) {
    size_t n = strlen(in), k = 0;
    for (size_t i = 0; i < n && k + 1 < out_sz; i++) out[k++] = in[n - 1 - i];
    if (k + 3 < out_sz) {
        out[k++] = '1';
        out[k++] = '2';
        out[k++] = '3';
    }
    out[k] = '\0';
}

static void now_text(char *buf, size_t sz) {
    time_t t = time(NULL);
    struct tm tm_now;
    if (!localtime_r(&t, &tm_now)) {
        snprintf(buf, sz, "timestamp unavailable");
        return;
    }
    strftime(buf, sz, "%Y-%m-%d %H:%M:%S", &tm_now);
}

int main(int argc, char *argv[]) {
    int port = (argc > 1) ? atoi(argv[1]) : 9106;
    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) return 1;
    int opt = 1;
    if (setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt");
        return 1;
    }

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(sfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) return 1;
    if (listen(sfd, 5) < 0) return 1;

    printf("File portal first-login server on %d\n", port);
    while (1) {
        int cfd = accept(sfd, NULL, NULL);
        if (cfd < 0) continue;

        char in[BUF] = {0}, username[128] = {0}, temp[128] = {0}, newp[128] = {0}, out[BUF] = {0};
        int n = recv(cfd, in, sizeof(in) - 1, 0);
        if (n > 0 && sscanf(in, "%127[^|]|%127[^|]|%127[^\n]", username, temp, newp) == 3) {
            char errors[BUF] = {0};
            if (!only_alpha_min6(username)) strcat(errors, "username must be minimum 6 alphabets; ");
            if (strlen(newp) < 10 || !has_upper(newp) || !has_digit(newp) || !has_special(newp))
                strcat(errors, "new_password must be >=10 with uppercase, digit, special char; ");

            int idx = find_user(username);
            if (!errors[0]) {
                if (idx < 0) strcat(errors, "user not found; ");
                else if (records[idx].reset_done) strcat(errors, "password reset already completed; ");
                else if (strcmp(records[idx].temp_password, temp) != 0) strcat(errors, "temp_password mismatch; ");
            }

            if (errors[0]) {
                snprintf(out, sizeof(out), "RESET_FAILED|%.990s\n", errors);
            } else {
                char transformed[64], ts[64];
                reverse_append_123(username, transformed, sizeof(transformed));
                strncpy(records[idx].transformed_username, transformed, sizeof(records[idx].transformed_username) - 1);
                records[idx].transformed_username[sizeof(records[idx].transformed_username) - 1] = '\0';
                records[idx].reset_done = 1;
                now_text(ts, sizeof(ts));
                snprintf(out, sizeof(out), "RESET_SUCCESS|processed_username=%s|timestamp=%s\n", records[idx].transformed_username, ts);
            }
        } else {
            snprintf(out, sizeof(out), "RESET_FAILED|Invalid request format\n");
        }
        if (send_all(cfd, out, strlen(out)) < 0) perror("send");
        close(cfd);
    }
}
