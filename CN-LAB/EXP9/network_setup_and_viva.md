# Experiment 9 Notes (Topology + UDP Multiuser Chat)

## Lab execution checklist
1. Build and wire topology with router/switch/CAT6.
2. Configure DHCP for dynamic client addressing.
3. Verify all nodes are in same reachable network.
4. Start UDP chat server on one node.
5. Start multiple chat clients from different nodes.
6. Verify multiuser messaging.
7. Attach step-by-step photos from your lab.

## Viva-style questions and answers
1. **Why UDP for multiuser chat demo?**  
   UDP is lightweight and connectionless, so one server can handle many users with simple packet broadcasts.

2. **Difference between TCP and UDP in one line?**  
   TCP is reliable/connection-oriented, while UDP is faster/connectionless with no delivery guarantee.

3. **Why DHCP is useful in a multi-node lab?**  
   It automatically gives valid IP configuration to clients and reduces manual errors.

4. **What happens if two hosts get same IP?**  
   IP conflict occurs and communication becomes unstable or fails.

5. **What does subnet mask do?**  
   It separates network part and host part of an IP address.
