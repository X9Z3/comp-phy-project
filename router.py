import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random


class AntNet:
    def __init__(self, graph, num_ants, alpha=2, beta=2, evaporation_rate=0.5, iterations=100, exploration_prob=0.1):
        self.graph = graph  # Graph represented as adjacency matrix
        self.num_nodes = len(graph)
        self.num_ants = num_ants
        self.alpha = alpha  # Pheromone influence
        self.beta = beta  # Heuristic influence
        self.evaporation_rate = evaporation_rate
        self.iterations = iterations
        self.exploration_prob = exploration_prob  # Probability of random exploration
        self.pheromone = np.ones((self.num_nodes, self.num_nodes)) * 0.1  # Initialize pheromones with a small positive value to encourage exploration
        self.best_path = None
        self.best_cost = float('inf')

    def heuristic(self, node, neighbor):
        """Inverse of the distance as heuristic."""
        distance = self.graph[node][neighbor]
        epsilon = 1e-3  # Small constant to avoid division by zero
        if distance == 0:  # Handle cases where there's no direct connection
            return 0  # Treat as a blocked path
        return 1 / (distance + epsilon)

    def probabilistic_choice(self, node, unvisited):
        """Select next node probabilistically,with a chance for exploration."""
        probabilities = []
        for neighbor in unvisited:
            if self.graph[node][neighbor] == 0:
                probabilities.append(0)  # No path available
            else:
                tau = self.pheromone[node][neighbor] ** self.alpha
                eta = self.heuristic(node, neighbor) ** self.beta
                probabilities.append(tau * eta)

        if sum(probabilities) == 0:  # No valid paths
            print(f"No valid paths from node {node}, choosing randomly")
            return random.choice(list(unvisited))  # Choose a random unvisited node

        probabilities = np.array(probabilities) / (sum(probabilities) + 1e-6)  # Avoid division by zero
        return random.choices(list(unvisited), weights=probabilities, k=1)[0]

    def generate_ant_path(self, start, end):
        """Simulate a single ant's path."""
        current_node = start
        path = [current_node]
        cost = 0
        unvisited = set(range(self.num_nodes)) - {start}

        while unvisited:
            next_node = self.probabilistic_choice(current_node, unvisited)
            if self.graph[current_node][next_node] == 0:  # Handle blocked path
                print(f"Blocked path from {current_node} to {next_node}, skipping")
                unvisited.remove(next_node)  # Remove this blocked node from consideration
                continue
            path.append(next_node)
            print(f"Ant moving from node {current_node} to node {next_node}")
            cost += self.graph[current_node][next_node]
            current_node = next_node
            unvisited.remove(next_node)
            if current_node == end:
                break

        if current_node != end:
            print(f"Ant failed to reach the destination from start {start}")
            return None, float('inf')  # Invalid path if the ant doesn't reach the destination

        print(f"Ant reached end {end} with path {path} and cost {cost}")
        return path, cost

    def update_pheromone(self, all_paths):
        """Update pheromone levels based on ants' paths."""
        self.pheromone *= (1 - self.evaporation_rate)  # Evaporation
        for path, cost in all_paths:
            pheromone_deposit = 1 / (cost + 1e-6) if cost > 0 else 0  # Avoid division by zero
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += pheromone_deposit  # Reinforcement

    def optimize(self, start, end):
        """Run the AntNet optimization."""
        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1}/{self.iterations}")
            all_paths = []
            for ant in range(self.num_ants):
                path, cost = self.generate_ant_path(start, end)
                if path:
                    all_paths.append((path, cost))
                    if cost < self.best_cost:
                        self.best_cost = cost
                        self.best_path = path
            self.update_pheromone(all_paths)
            print(f"End of Iteration {iteration + 1}, Best Cost: {self.best_cost}\n")

        return self.best_path, self.best_cost

    def visualize_graph(self, show_route=True):
        if self.best_path:
            edges_to_highlight = [(self.best_path[i], self.best_path[i + 1]) if self.best_path[i] < self.best_path[i + 1] else (self.best_path[i + 1], self.best_path[i]) for i in range(len(self.best_path) - 1)]
        else:
            print("No current best path to visualize.")
            show_route = False

        # Adjacency matrix to numpy array to NetworkX graph
        graph_x = nx.from_numpy_array(np.array(self.graph))

        # Not all graphs can be drawn planar, but if it can, then use that positional layout
        if nx.is_planar(graph_x):
            pos = nx.planar_layout(graph_x)
        else:
            pos = nx.spring_layout(graph_x)

        # Specify custom colors for nodes and edges
        if show_route:
            node_colors = ['red' if node in self.best_path else 'blue' for node in graph_x.nodes()]
            edge_colors = ['red' if edge in edges_to_highlight else 'gray' for edge in graph_x.edges()]
        else:
            node_colors = ['blue'] * self.num_nodes
            edge_colors = ['gray'] * self.num_nodes

        # Extract weights and scale them for line thickness
        edge_weights = [graph_x[u][v]['weight'] for u, v in graph_x.edges()]
        edge_thickness = [1 + 5 * weight / max(edge_weights) for weight in edge_weights]  # Scaled thickness

        # Draw the graph
        nx.draw(
            graph_x,
            pos,
            with_labels=True,
            node_color=node_colors,
            edge_color=edge_colors,
            node_size=50,
            font_color='white',
            font_weight='bold',
            width=edge_thickness
        )

        plt.show()
        dijkstra_path = nx.dijkstra_path(graph_x, source=0, target=7)
        dijkstra_cost = nx.path_weight(graph_x, dijkstra_path, 'weight')
        print(f'Dijkstra shortest path: {dijkstra_path}')
        print(f'Dijkstra path cost: {dijkstra_cost}')


def create_random_graph(n, weight_range=(1, 10), edge_probability=0.3):
    """
    Creates a random undirected graph with n nodes. G is a NetworkX Graph object. Returns its adjacency list.

    Parameters:
    - n: Number of nodes.
    - weight_range: Tuple indicating the range of edge weights (inclusive).
    - edge_probability: Probability of an edge existing between any two nodes.

    Returns:
    - adj_matrix: A 2D list representing the adjacency matrix.
    """
    G = nx.Graph()  # Initialize an undirected graph

    # Add nodes
    for i in range(n):
        G.add_node(i)

    # Add edges with random weights
    for i in range(n):
        for j in range(i + 1, n):  # Avoid duplicate edges
            if random.random() < edge_probability:
                weight = random.randint(*weight_range)
                G.add_edge(i, j, weight=weight)

   # Convert to numpy array
    adj_matrix = nx.to_numpy_array(G, weight='weight')  # Include edge weights
    return adj_matrix.tolist()


# Example Usage:
if __name__ == "__main__":
    # More complex graph as adjacency matrix (0 indicates no path)
    graph = [
        [0, 2, 4, 6, 0, 7, 0, 0],  # Node 0, removed direct path to node 7
        [2, 0, 0, 1, 8, 5, 3, 0],  # Node 1
        [4, 0, 0, 3, 5, 0, 8, 0],  # Node 2
        [6, 1, 3, 0, 2, 0, 7, 4],  # Node 3
        [0, 8, 5, 2, 0, 4, 6, 9],  # Node 4
        [7, 5, 0, 0, 4, 0, 3, 5],  # Node 5
        [0, 3, 8, 7, 6, 3, 0, 1],  # Node 6
        [0, 0, 0, 4, 9, 5, 1, 0],  # Node 7, removed direct path from node 0
    ]

    use_random_graph = True

    # Generate a random graph with 8 nodes
    n_nodes = 2000 # Takes a pinch but works with 2000+ nodes
    graph = create_random_graph(n_nodes, weight_range=(1, 20), edge_probability=0.005) if use_random_graph else graph
    ant_net = AntNet(graph, num_ants=15, alpha=2, beta=2, evaporation_rate=0.5, iterations=100, exploration_prob=0.15)
    start_node = 0
    end_node = 7
    best_path, best_cost = ant_net.optimize(start=start_node, end=end_node)

    print("\nBest Path:", best_path)
    print("Best Cost:", best_cost)

    ant_net.visualize_graph()
