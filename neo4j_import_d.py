import os
from neo4j import GraphDatabase

# --- Neo4j Configuration ---
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_DATABASE = "graphmatching"
if os.environ.get("NEO4J_PASSWORD"):
    NEO4J_AUTH = (NEO4J_USERNAME, os.environ.get("NEO4J_PASSWORD"))
else:
    print("Warning: NEO4J_PASSWORD environment variable is not set.")


def create_detection_graph(nodes_list, edges_list, graph_id, graph_prefix, node_techniques, edge_pairs):
    """
    Creates nodes and edges for a detection graph and appends them to the provided lists.

    Args:
        nodes_list (list): The list to which created nodes will be appended.
        edges_list (list): The list to which created edges will be appended.
        graph_id (str): The identifier for the graph.
        graph_prefix (str): A prefix for node and edge IDs to ensure uniqueness.
        node_techniques (list[str]): A list of technique IDs for the nodes.
        edge_pairs (list[tuple[int, int]]): A list of (source_idx, target_idx) tuples for edges,
                                           where indices refer to the `node_techniques` list.
    """
    created_node_ids = []
    # Create nodes
    for i, technique_id in enumerate(node_techniques):
        node_id = f"{graph_prefix}_n{i+1}"
        node = {
            'id': node_id,
            'type': 'action',
            'graph_id': graph_id,
            'technique_id': technique_id
        }
        nodes_list.append(node)
        created_node_ids.append(node_id)

    # Create edges
    for i, (source_idx, target_idx) in enumerate(edge_pairs):
        edge_id = f"{graph_prefix}_e{i+1}"
        edge = {
            'id': edge_id,
            'type': 'dynamic_line',
            'graph_id': graph_id,
            'source': created_node_ids[source_idx],
            'target': created_node_ids[target_idx]
        }
        edges_list.append(edge)


# Neo4j import
def import_graph(tx, nodes, edges):
    for node in nodes:
        tx.run(
            "MERGE (n:DetectionNode {id: $id}) "
            "SET n.type = $type, n.graph_id = $graph_id, n.technique_id = $technique_id",
            id=node.get("id"), type=node.get("type"), graph_id=node.get("graph_id"), technique_id=node.get("technique_id")
        )

    for edge in edges:
        tx.run(
            "MATCH (src:DetectionNode {id: $source}), (dst:DetectionNode {id: $target}) "
            "MERGE (src)-[r:DetectionEdge {id: $id, type: $type, graph_id: $graph_id}]->(dst)",
            id=edge.get("id"), type=edge.get("type"), graph_id=edge.get("graph_id"), source=edge.get("source"), target=edge.get("target")
        )


driver = GraphDatabase.driver(NEO4J_URL, auth=NEO4J_AUTH, database=NEO4J_DATABASE)
nodes, edges = [], []

# --- Create detection graph 1 ---
# All 3 nodes exist in the reference graph.
create_detection_graph(
    nodes_list=nodes,
    edges_list=edges,
    graph_id="d1_3_3",
    graph_prefix="d1",
    node_techniques=[
        'T1566.001',  # Phishing: Spearphishing Attachment
        'T1059.001',  # Command and Scripting Interpreter: PowerShell
        'T1573',      # Encrypted Channel
    ],
    edge_pairs=[(0, 1), (1, 2)]
)
# --- Create detection graph 2 ---
# Duplicate of detection graph 1
create_detection_graph(
    nodes_list=nodes,
    edges_list=edges,
    graph_id="d2_3_3",
    graph_prefix="d2",
    node_techniques=[
        'T1566.001',  # Phishing: Spearphishing Attachment
        'T1059.001',  # Command and Scripting Interpreter: PowerShell
        'T1573',      # Encrypted Channel
    ],
    edge_pairs=[(0, 1), (1, 2)]
)
# --- Create detection graph 3 ---
# None of the nodes exist in the reference graph.
create_detection_graph(
    nodes_list=nodes,
    edges_list=edges,
    graph_id="d3_3_0",
    graph_prefix="d3",
    node_techniques=[
        'T1078.002',  # Valid Accounts: Domain Accounts
        'T1133',      # External Remote Services
        'T1071.001',  # Application Layer Protocol: Web Protocols
    ],
    edge_pairs=[(0, 1), (1, 2)]
)


with driver.session(database=NEO4J_DATABASE) as session:
    session.execute_write(import_graph, nodes, edges)

print(f"Imported {len(nodes)} nodes and {len(edges)} edges into Neo4j.")


# --- Example Cypher Queries ---
'''
MATCH (n:DetectionNode)-[r:DetectionEdge {graph_id: "d_3_3"}]->(m:DetectionNode)
RETURN n, r, m
'''
