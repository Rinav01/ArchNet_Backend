import uuid
from typing import List, Dict, Set
from app.models.node import Node
from app.models.edge import Edge

class ValidationService:
    @staticmethod
    def topological_sort(nodes: List[Node], edges: List[Edge]) -> List[Node]:
        """Perform topological sort on the nodes based on edges to detect cycles and execution order.
        Raises ValueError if a cycle is detected.
        """
        node_ids = {node.id for node in nodes}
        adj: Dict[uuid.UUID, List[uuid.UUID]] = {nid: [] for nid in node_ids}
        in_degree: Dict[uuid.UUID, int] = {nid: 0 for nid in node_ids}

        # Build graph representation
        for edge in edges:
            # Skip edges pointing outside our active nodes list
            if edge.from_node_id in node_ids and edge.to_node_id in node_ids:
                adj[edge.from_node_id].append(edge.to_node_id)
                in_degree[edge.to_node_id] += 1

        # Queue of nodes with no incoming connections (sources)
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        sorted_ids: List[uuid.UUID] = []

        while queue:
            # Maintain stable sorting order by popping the first element
            u = queue.pop(0)
            sorted_ids.append(u)

            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(sorted_ids) != len(nodes):
            raise ValueError("Invalid architecture: Graph contains cycles (loops are not allowed).")

        # Map sorted UUIDs back to node objects
        node_map = {node.id: node for node in nodes}
        return [node_map[nid] for nid in sorted_ids]

    @staticmethod
    def validate_graph(nodes: List[Node], edges: List[Edge]) -> List[Node]:
        """Validates the neural network DAG.
        Checks:
        1. Graph is not empty
        2. Exactly one Input node exists
        3. No cycles exist (Topological Sort checks this)
        4. Detects disconnected graphs (reachability from Input node)
        
        Returns the topologically sorted list of nodes if valid.
        """
        if not nodes:
            raise ValueError("Invalid architecture: Graph is empty.")

        # Find the input layer
        input_nodes = [n for n in nodes if n.type.lower() == "input"]
        if len(input_nodes) == 0:
            raise ValueError("Invalid architecture: Missing 'Input' layer.")
        if len(input_nodes) > 1:
            raise ValueError("Invalid architecture: Multiple 'Input' layers detected. Only one is allowed.")
        
        input_node = input_nodes[0]

        # 1. Sort the graph (checks for cycles)
        sorted_nodes = ValidationService.topological_sort(nodes, edges)

        # 2. Check reachability from Input node (detect disconnected graph parts)
        node_ids = {node.id for node in nodes}
        adj: Dict[uuid.UUID, List[uuid.UUID]] = {nid: [] for nid in node_ids}
        for edge in edges:
            if edge.from_node_id in node_ids and edge.to_node_id in node_ids:
                adj[edge.from_node_id].append(edge.to_node_id)

        visited: Set[uuid.UUID] = set()
        queue = [input_node.id]

        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        queue.append(neighbor)

        # If any node is not reachable from the input node, we have disconnected components
        unreachable = node_ids - visited
        if unreachable:
            unreachable_labels = [n.label for n in nodes if n.id in unreachable]
            raise ValueError(
                f"Invalid architecture: Disconnected layers detected {unreachable_labels}. "
                "All layers must connect to the path starting from the 'Input' layer."
            )

        return sorted_nodes
