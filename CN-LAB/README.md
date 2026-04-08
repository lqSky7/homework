# CN-LAB Assignment Pack

This folder contains detailed material for all 10 CN lab experiments.

## Folder map
- `EXP1` Networking basics notes + command table template
- `EXP2` Error detection mechanisms in C (VRC, LRC, CRC, Checksum)
- `EXP3` Hamming code (sender/receiver)
- `EXP4` Lab activity report template (topology, DHCP, shared storage)
- `EXP5` Lab activity report template (website + SNMP tool manual)
- `EXP6` IPv4 conversion and subnetting (classful/classless)
- `EXP7` TCP client/server programs (math + chat)
  - Includes additional secure socket practice questions
- `EXP8` ARQ flow control programs with socket-based simulation
- `EXP9` UDP multiuser chat program
- `EXP10` Routing algorithms (Dijkstra and Bellman-Ford)

## Compile examples
```bash
gcc CN-LAB/EXP2/error_detection.c -o CN-LAB/EXP2/error_detection
gcc CN-LAB/EXP3/hamming_code.c -o CN-LAB/EXP3/hamming_code
gcc CN-LAB/EXP6/ip_addressing.c -lm -o CN-LAB/EXP6/ip_addressing
gcc CN-LAB/EXP10/dijkstra_routing.c -o CN-LAB/EXP10/dijkstra_routing
gcc CN-LAB/EXP10/bellman_ford_routing.c -o CN-LAB/EXP10/bellman_ford_routing
```

Socket-based files may need two terminals (server and client).

## Extra practice question bank
- `EXP7/secure_socket_practice_questions.md`:
  secure credential-handling and validation-based client-server questions.
  
## Static routing 
Here is your complete master list with the router ports and their connected PCs all in one place:

### **Left Network (Router0 & Switch0)**
**Router0 Interfaces:**
* **Port connecting down to Switch0:** `192.168.1.1` *(This is the Default Gateway for these PCs)*
* **Port connecting right to Router1:** `10.1.1.1`

**Connected PCs (IP / Subnet Mask / Gateway):**
* **PC0:** `192.168.1.10` / `255.255.255.0` / `192.168.1.1`
* **PC1:** `192.168.1.11` / `255.255.255.0` / `192.168.1.1`

---

### **Middle Network (Router1 & Switch1)**
**Router1 Interfaces:**
* **Port connecting left to Router0:** `10.1.1.2`
* **Port connecting down to Switch1:** `192.168.2.1` *(This is the Default Gateway for these PCs)*
* **Port connecting right to Router2:** `10.2.2.1`

**Connected PCs (IP / Subnet Mask / Gateway):**
* **PC2:** `192.168.2.10` / `255.255.255.0` / `192.168.2.1`
* **PC3:** `192.168.2.11` / `255.255.255.0` / `192.168.2.1`

---

### **Right Network (Router2 & Switch2)**
**Router2 Interfaces:**
* **Port connecting left to Router1:** `10.2.2.2`
* **Port connecting down to Switch2:** `192.168.3.1` *(This is the Default Gateway for these PCs)*

**Connected PCs (IP / Subnet Mask / Gateway):**
* **PC4:** `192.168.3.10` / `255.255.255.0` / `192.168.3.1`
* **PC5:** `192.168.3.11` / `255.255.255.0` / `192.168.3.1`

---

With everything nicely organized, are you ready to try plugging those Static Routes back into Router1 and run that test ping?
