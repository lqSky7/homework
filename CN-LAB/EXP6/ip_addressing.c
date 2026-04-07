#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include <math.h>
#include <ctype.h>

static int parse_dotted(const char *s, uint32_t *ip) {
    int a, b, c, d;
    if (sscanf(s, "%d.%d.%d.%d", &a, &b, &c, &d) != 4) return 0;
    if (a < 0 || a > 255 || b < 0 || b > 255 || c < 0 || c > 255 || d < 0 || d > 255) return 0;
    *ip = ((uint32_t)a << 24) | ((uint32_t)b << 16) | ((uint32_t)c << 8) | (uint32_t)d;
    return 1;
}

static int parse_binary32(const char *s, uint32_t *ip) {
    char bits[33];
    int k = 0;
    for (size_t i = 0; s[i]; i++) {
        if (s[i] == '.') continue;
        if (s[i] != '0' && s[i] != '1') return 0;
        if (k >= 32) return 0;
        bits[k++] = s[i];
    }
    if (k != 32) return 0;
    bits[32] = '\0';
    *ip = 0;
    for (int i = 0; i < 32; i++) *ip = (*ip << 1) | (bits[i] - '0');
    return 1;
}

static int parse_hex(const char *s, uint32_t *ip) {
    char buf[64];
    int k = 0;
    size_t len = strlen(s);
    for (size_t i = 0; i < len; i++) {
        if (s[i] == '.' || s[i] == ':' || s[i] == 'x' || s[i] == 'X') continue;
        if (s[i] == '0' && i + 1 < len && (s[i + 1] == 'x' || s[i + 1] == 'X')) continue;
        if (!isxdigit((unsigned char)s[i])) return 0;
        if (k >= 8) return 0;
        buf[k++] = s[i];
    }
    if (k != 8) return 0;
    buf[8] = '\0';
    *ip = (uint32_t)strtoul(buf, NULL, 16);
    return 1;
}

static void print_binary(uint32_t ip) {
    for (int i = 31; i >= 0; i--) {
        printf("%d", (ip >> i) & 1);
        if (i % 8 == 0 && i != 0) printf(".");
    }
    printf("\n");
}

static void print_dotted(uint32_t ip) {
    printf("%u.%u.%u.%u\n", (ip >> 24) & 255, (ip >> 16) & 255, (ip >> 8) & 255, ip & 255);
}

static void print_hex(uint32_t ip) {
    printf("0x%08X\n", ip);
}

static int classful_prefix(uint32_t ip) {
    uint8_t first = (ip >> 24) & 255;
    if (first <= 127) return 8;
    if (first <= 191) return 16;
    if (first <= 223) return 24;
    return -1;
}

static const char *class_name(uint32_t ip) {
    uint8_t first = (ip >> 24) & 255;
    if (first <= 127) return "Class A";
    if (first <= 191) return "Class B";
    if (first <= 223) return "Class C";
    if (first <= 239) return "Class D (Multicast)";
    return "Class E";
}

static uint32_t mask_from_prefix(int p) {
    if (p <= 0) return 0;
    if (p >= 32) return 0xFFFFFFFFu;
    return 0xFFFFFFFFu << (32 - p);
}

static int is_classless_mode(const char *addressing_mode) {
    if (!addressing_mode) return 0;
    return strcasecmp(addressing_mode, "classless") == 0;
}

static void part61(void) {
    char input[128];
    int in_fmt, out_fmt;
    uint32_t ip;

    printf("Input format (1=binary, 2=dotted-decimal, 3=hex): ");
    scanf("%d", &in_fmt);
    printf("Enter IPv4 address: ");
    scanf("%127s", input);

    int ok = 0;
    if (in_fmt == 1) ok = parse_binary32(input, &ip);
    else if (in_fmt == 2) ok = parse_dotted(input, &ip);
    else if (in_fmt == 3) ok = parse_hex(input, &ip);

    if (!ok) {
        printf("Invalid input format/value.\n");
        return;
    }

    printf("Output format (1=binary, 2=dotted-decimal, 3=hex): ");
    scanf("%d", &out_fmt);
    if (out_fmt == 1) print_binary(ip);
    else if (out_fmt == 2) print_dotted(ip);
    else if (out_fmt == 3) print_hex(ip);
    else printf("Invalid output format.\n");
}

static void part62(void) {
    char ipstr[64], addressing_mode[16], maskstr[64];
    uint32_t ip, mask = 0;
    int net_bits, num_subnets;

    printf("Enter IPv4 address (dotted-decimal): ");
    scanf("%63s", ipstr);
    if (!parse_dotted(ipstr, &ip)) {
        printf("Invalid IPv4 address.\n");
        return;
    }

    printf("Addressing type (classless/classful): ");
    scanf("%15s", addressing_mode);

    printf("Enter subnet mask in dotted format (or 'none'): ");
    scanf("%63s", maskstr);
    if (strcmp(maskstr, "none") != 0 && strcmp(maskstr, "NONE") != 0) {
        if (!parse_dotted(maskstr, &mask)) {
            printf("Invalid subnet mask.\n");
            return;
        }
    }

    if (is_classless_mode(addressing_mode)) {
        printf("Enter number of bits allocated to network ID: ");
        scanf("%d", &net_bits);
    } else {
        net_bits = classful_prefix(ip);
        if (net_bits < 0) {
            printf("Class D/E not valid for host subnetting.\n");
            return;
        }
    }

    if (mask == 0) mask = mask_from_prefix(net_bits);
    if (net_bits < 1 || net_bits > 30) {
        printf("Network ID bits must be between 1 and 30.\n");
        return;
    }

    printf("Enter number of subnets: ");
    scanf("%d", &num_subnets);
    if (num_subnets <= 0) {
        printf("Invalid subnet count.\n");
        return;
    }

    int subnet_bits = 0;
    while ((1 << subnet_bits) < num_subnets) subnet_bits++;
    int new_prefix = net_bits + subnet_bits;
    if (new_prefix > 30) {
        printf("Too many subnets for given network bits.\n");
        return;
    }

    uint32_t new_mask = mask_from_prefix(new_prefix);
    uint32_t base_network = ip & mask_from_prefix(net_bits);
    int host_bits = 32 - new_prefix;
    uint32_t block = (1u << host_bits);

    printf("\n--- Subnetting Results ---\n");
    printf("Subnet mask used: ");
    print_dotted(new_mask);

    for (int s = 0; s < num_subnets; s++) {
        uint32_t network = base_network + (uint32_t)s * block;
        uint32_t broadcast = network + block - 1;
        uint32_t first = (block > 2) ? network + 1 : network;
        uint32_t last = (block > 2) ? broadcast - 1 : broadcast;

        printf("\nSubnet %d\n", s + 1);
        printf("(i) Class name: %s\n", is_classless_mode(addressing_mode) ? "N/A (classless)" : class_name(ip));
        printf("(ii) Subnet mask: "); print_dotted(new_mask);
        printf("(iii) Network ID: "); print_dotted(network);
        printf("(iv) Usable host range: "); print_dotted(first); printf(" to "); print_dotted(last);
        printf("(v) Total number of addresses: %u\n", block);
        printf("(vi) First address: "); print_dotted(first);
        printf("(vii) Last address: "); print_dotted(last);
        printf("(viii) Broadcast address: "); print_dotted(broadcast);
    }
}

int main(void) {
    int choice;
    printf("IPv4 Addressing Lab\n");
    printf("1. Task 6.1 (format conversion)\n");
    printf("2. Task 6.2 (subnetting calculations)\n");
    printf("Choose: ");
    if (scanf("%d", &choice) != 1) return 1;

    if (choice == 1) part61();
    else if (choice == 2) part62();
    else printf("Invalid option.\n");

    return 0;
}
