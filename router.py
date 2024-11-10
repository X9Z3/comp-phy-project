import random
import numpy as np

class Router:
    def __init__(self, address, data_rate, queue_size, num_nodes, start_time, end_time, q_weight_factor, log_results, convergance_time, probabilistic_routing=True):
        self.address = address
        self.data_rate = data_rate
        self.queue_size = queue_size
        self.num_nodes = num_nodes
        self.start_time = start_time
        self.end_time = end_time
        self.q_weight_factor = q_weight_factor
        self.log_results = log_results
        self.convergance_time = convergance_time
        self.probabilistic_routing = probabilistic_routing
        
        self.neighbors = {}  # Stores neighbor info
        self.routing_table = {}  # Routing table storing probabilities
        self.queue = {}  # Buffer queue for each neighbor
        
        self.init_ant_routing_table(initial_prob=1.0 / num_nodes)
        
    def add_neighbor(self, port, neighbor_address, bandwidth, propagation_delay):
        self.neighbors[port] = {
            'neighbor_address': neighbor_address,
            'bandwidth': bandwidth,
            'propagation_delay': propagation_delay
        }
        self.queue[port] = []  # Initialize buffer for each port
        # Update routing table with new neighbor
        for destination in range(self.num_nodes):
            if destination != self.address:
                if destination not in self.routing_table:
                    self.routing_table[destination] = {}
                self.routing_table[destination][port] = 1.0 / len(self.neighbors)

    def init_ant_routing_table(self, initial_prob):
        for destination in range(self.num_nodes):
            if destination != self.address:
                self.routing_table[destination] = {}
                for port, neighbor_info in self.neighbors.items():
                    self.routing_table[destination][port] = initial_prob

    def choose_next_hop(self, packet):
        destination = packet['destination']
        enter_port = packet.get('enter_port', -1)
        
        if destination not in self.routing_table or not self.routing_table[destination]:
            raise ValueError("No valid ports available for routing")
        
        # Probabilistic routing decision
        if self.probabilistic_routing:
            ports = list(self.routing_table[destination].keys())
            probabilities = [self.routing_table[destination][port] for port in ports]
            
            # Remove the entry port from options
            if enter_port in ports:
                index = ports.index(enter_port)
                ports.pop(index)
                probabilities.pop(index)
            
            if len(ports) == 1:
                return ports[0]
            
            if not ports:
                raise ValueError("No valid ports available for routing")
            
            total_probability = sum(probabilities)
            if total_probability == 0:
                raise ValueError("No valid ports available for routing")
            normalized_probabilities = [p / total_probability for p in probabilities]
            
            return np.random.choice(ports, p=normalized_probabilities)
        
        # Non-probabilistic, greedy routing
        max_prob_port = max(self.routing_table[destination], key=self.routing_table[destination].get)
        return max_prob_port

    def process_hello_packet(self, port, source_address):
        # When a hello packet is received, reply with a hello reply
        packet = {
            'type': 'hello_reply',
            'source_address': self.address,
            'destination': source_address,
            'timestamp': self.current_time()
        }
        self.enqueue_packet(packet, port)

    def process_data_packet(self, packet):
        # If the packet has reached its destination
        if packet['destination'] == self.address:
            self.deliver_to_application(packet)
        else:
            try:
                next_port = self.choose_next_hop(packet)
                packet['hops'] += 1
                packet['timestamp'] = self.current_time()
                self.enqueue_packet(packet, next_port)
            except ValueError as e:
                print(f"Error: {e}. Packet dropped.")

    def enqueue_packet(self, packet, port):
        if len(self.queue[port]) < self.queue_size:
            self.queue[port].append(packet)
        else:
            # Drop packet if queue is full (for simplicity)
            if packet['type'] == 'data':
                print(f"Packet to {packet['destination']} dropped due to full queue")

    def transmit_packet(self, port):
        if self.queue[port]:
            packet = self.queue[port].pop(0)
            self.send(packet, port)

    def send(self, packet, port):
        neighbor_info = self.neighbors[port]
        delay = packet['length'] / neighbor_info['bandwidth'] + neighbor_info['propagation_delay']
        # Simulate sending (here we just print, replace with actual send logic)
        print(f"Sending packet {packet} to port {port} with delay {delay}")

    def deliver_to_application(self, packet):
        print(f"Packet {packet} delivered to application at Router {self.address}")

    def current_time(self):
        # Placeholder for simulation time
        return random.uniform(0, 1)  # Replace with actual time management

    def update_routing_table(self, destination, neighbor, prob):
        port = self.get_port_for_neighbor(neighbor)
        self.routing_table[destination][port] = prob

    def get_port_for_neighbor(self, neighbor):
        for port, info in self.neighbors.items():
            if info['neighbor_address'] == neighbor:
                return port
        raise ValueError(f"Neighbor {neighbor} not found")

# Example usage
router = Router(address=1, data_rate=1000, queue_size=10, num_nodes=5, start_time=0, end_time=10, q_weight_factor=0.5, log_results=True, convergance_time=5)
router.add_neighbor(port=0, neighbor_address=2, bandwidth=100, propagation_delay=0.01)
router.add_neighbor(port=1, neighbor_address=3, bandwidth=150, propagation_delay=0.02)

# Simulate a data packet arriving
packet = {
    'type': 'data',
    'source_address': 0,
    'destination': 3,
    'length': 100,
    'hops': 0
}
router.process_data_packet(packet)

# Transmit a packet from port 0
router.transmit_packet(port=0)
