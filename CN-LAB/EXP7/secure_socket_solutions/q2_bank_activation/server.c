#include <arpa/inet.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define BUF 512

static int is_alpha_string(const char *s) {
    if (!s || !*s) return 0;
    for (int i = 0; s[i]; i++) {
        if (!isalpha((unsigned char)s[i])) return 0;
    }
    return 1;
}

static int has_digit(const char *s) {
    for (int i = 0; s[i]; i++) if (isdigit((unsigned char)s[i])) return 1;
    return 0;
}

static int has_special(const char *s) {
    for (int i = 0; s[i]; i++) if (!isalnum((unsigned char)s[i])) return 1;
    return 0;
}

static void reverse_append_123(const char *in, char *out, size_t out_sz) {
    size_t n = strlen(in);
    size_t k = 0;
    for (size_t i = 0; i < n && k + 1 < out_sz; i++) out[k++] = in[n - 1 - i];
    if (k + 3 < out_sz) {
        out[k++] = '1';
        out[k++] = '2';
        out[k++] = '3';
    }
    out[k] = '\0';
}

int main(int argc, char *argv[]) {
    int port = (argc > 1) ? atoi(argv[1]) : 9101;
    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) return 1;

    int opt = 1;
    setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(sfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) return 1;
    if (listen(sfd, 5) < 0) return 1;

    printf("Bank activation server on %d\n", port);
    while (1) {
        int cfd = accept(sfd, NULL, NULL);
        if (cfd < 0) continue;

        char in[BUF] = {0}, username[128] = {0}, password[128] = {0}, out[BUF] = {0};
        int n = recv(cfd, in, sizeof(in) - 1, 0);
        if (n > 0 && sscanf(in, "%127[^|]|%127[^\n]", username, password) == 2) {
            char errors[BUF] = {0};
            if (strlen(username) < 6 || !is_alpha_string(username))
                strcat(errors, "Username must be >=6 alphabets only; ");
            if (strlen(password) < 8 || !has_digit(password) || !has_special(password))
                strcat(errors, "Password must be >=8 with digit and special char; ");

            if (errors[0]) {
                snprintf(out, sizeof(out), "ACCOUNT_REJECTED|%.480s\n", errors);
            } else {
                char enc_user[160];
                reverse_append_123(username, enc_user, sizeof(enc_user));
                snprintf(out, sizeof(out), "ACCOUNT_ACTIVATED|processed_username=%s\n", enc_user);
            }
        } else {
            snprintf(out, sizeof(out), "ACCOUNT_REJECTED|Invalid request format\n");
        }
        send(cfd, out, strlen(out), 0);
        close(cfd);
    }
}
