#include <arpa/inet.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define BUF 768
#define MAX_IDS 128

static int send_all(int fd, const char *buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = send(fd, buf + off, len - off, 0);
        if (n <= 0) return -1;
        off += (size_t)n;
    }
    return 0;
}

static char stored_ids[MAX_IDS][16];
static int stored_count = 0;

static int all_digits_len(const char *s, int len) {
    if ((int)strlen(s) != len) return 0;
    for (int i = 0; s[i]; i++) if (!isdigit((unsigned char)s[i])) return 0;
    return 1;
}

static int only_alpha_min6(const char *s) {
    if (strlen(s) < 6) return 0;
    for (int i = 0; s[i]; i++) if (!isalpha((unsigned char)s[i])) return 0;
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

static int duplicate_patient_id(const char *id) {
    for (int i = 0; i < stored_count; i++) if (strcmp(stored_ids[i], id) == 0) return 1;
    return 0;
}

static void add_patient_id(const char *id) {
    if (stored_count < MAX_IDS) {
        strncpy(stored_ids[stored_count], id, sizeof(stored_ids[stored_count]) - 1);
        stored_ids[stored_count][sizeof(stored_ids[stored_count]) - 1] = '\0';
        stored_count++;
    }
}

static void build_enc_username(const char *username, char *out, size_t out_sz) {
    size_t n = strlen(username), k = 0;
    for (size_t i = 0; i < n && k + 1 < out_sz; i++) out[k++] = toupper((unsigned char)username[n - 1 - i]);
    if (k + 3 < out_sz) {
        out[k++] = 'H';
        out[k++] = 'S';
        out[k++] = 'P';
    }
    out[k] = '\0';
}

int main(int argc, char *argv[]) {
    int port = (argc > 1) ? atoi(argv[1]) : 9104;
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

    printf("Hospital registration server on %d\n", port);
    while (1) {
        int cfd = accept(sfd, NULL, NULL);
        if (cfd < 0) continue;
        char in[BUF] = {0}, pid[64] = {0}, user[128] = {0}, pass[128] = {0}, out[BUF] = {0};
        int n = recv(cfd, in, sizeof(in) - 1, 0);
        if (n > 0 && sscanf(in, "%63[^|]|%127[^|]|%127[^\n]", pid, user, pass) == 3) {
            char errors[BUF] = {0};
            if (!all_digits_len(pid, 6)) strcat(errors, "patient_id must be exactly 6 digits; ");
            if (!only_alpha_min6(user)) strcat(errors, "username must be alphabets only, min 6 chars; ");
            if (strlen(pass) < 8 || !has_digit(pass) || !has_special(pass))
                strcat(errors, "password must be >=8 with digit and special char; ");
            if (!errors[0] && duplicate_patient_id(pid)) strcat(errors, "duplicate patient_id; ");

            if (errors[0]) {
                snprintf(out, sizeof(out), "REGISTRATION_FAILED|%.730s\n", errors);
            } else {
                char enc_user[192];
                build_enc_username(user, enc_user, sizeof(enc_user));
                add_patient_id(pid);
                snprintf(out, sizeof(out), "REGISTRATION_OK|patient_id=%s|encrypted_username=%s\n", pid, enc_user);
            }
        } else {
            snprintf(out, sizeof(out), "REGISTRATION_FAILED|Invalid request format\n");
        }
        if (send_all(cfd, out, strlen(out)) < 0) perror("send");
        close(cfd);
    }
}
