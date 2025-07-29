import json
import argparse
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


def flatten_properties(properties):
    """
    Parses a list of property pairs, separating special keys and flattening the rest.
    Complex values in the flattened properties are serialized to JSON strings.
    """
    flat_props = {}
    special_props = {"tactic_id": None, "technique_id": None, "confidence": None}
    
    for p in properties:
        if isinstance(p, list) and len(p) == 2:
            key, value = p
            if key in special_props:
                special_props[key] = value
            else:
                # serialize non-primitive types to a JSON string.
                if not isinstance(value, (str, int, float, bool, type(None))):
                    value = json.dumps(value)
                flat_props[key] = value
                
    return flat_props, special_props["tactic_id"], special_props["technique_id"], special_props["confidence"]


def import_graph(tx, nodes, edges):
    """
    Imports nodes and edges into the Neo4j database.
    Nodes are created or merged based on their IDs, and edges are created between nodes.
    """
    for node in nodes:
        tx.run(
            "MERGE (n:AttackFlowNode {id: $id}) "
            "SET n.type = $type, n.graph_id = $graph_id, n += $properties"
            + (", n.tactic_id = $tactic_id" if "tactic_id" in node else "")
            + (", n.technique_id = $technique_id" if "technique_id" in node else "")
            + (", n.confidence = $confidence" if "confidence" in node else ""),
            id=node.get("id"),
            type=node.get("type"),
            graph_id=node.get("graph_id"),
            properties=node.get("properties"),
            tactic_id=node.get("tactic_id"),
            technique_id=node.get("technique_id"),
            confidence=node.get("confidence")
        )

    for edge in edges:
        tx.run(
            "MATCH (src:AttackFlowNode {id: $source}), (dst:AttackFlowNode {id: $target}) "
            "MERGE (src)-[r:AttackFlowEdge {id: $id, type: $type, graph_id: $graph_id}]->(dst)",
            id=edge.get("id"),
            type=edge.get("type"),
            graph_id=edge.get("graph_id"),
            source=edge.get("source"),
            target=edge.get("target")
        )
                     


def main():
    parser = argparse.ArgumentParser(description="Import Attack Flow reference graph into Neo4j.")
    parser.add_argument("--input", "-i", required=True, help="Path to Attack Flow reference graph (.afb) file")
    parser.add_argument("--graph_id", "-g", required=True, help="Graph ID to use for import")
    args = parser.parse_args()

    # load the AttackFlow JSON
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    driver = GraphDatabase.driver(NEO4J_URL, auth=NEO4J_AUTH, database=NEO4J_DATABASE)

    # build lookup tables to store latches-anchor and anchors-node mapping
    latches_to_anchor, anchors_to_node = {}, {}

    for obj in data["objects"]:
        obj_id = obj.get("id")
        instance = obj.get("instance")
        anchors = obj.get("anchors", {})
        latches = obj.get('latches', {})

        if obj_id and instance:
            # store latches to anchor mapping
            for latch in latches:
                if latch not in latches_to_anchor:
                    latches_to_anchor[latch] = instance
            # store anchors to node mapping
            for anchor in anchors.values():
                if anchor not in anchors_to_node:
                    anchors_to_node[anchor] = instance

    # parse nodes and edges
    nodes, edges = [], []
    graph_id = args.graph_id

    for obj in data["objects"]:
        obj_id = obj.get("id")
        instance = obj.get("instance")
        properties = obj.get("properties", [])
        if obj_id and instance:
            if properties:
                flat_props, tactic_id, technique_id, confidence = flatten_properties(properties)
                node = {
                    "id": instance,
                    "type": obj_id,
                    "graph_id": graph_id,
                    "properties": flat_props
                }
                if tactic_id is not None:
                    node["tactic_id"] = tactic_id
                if technique_id is not None:
                    node["technique_id"] = technique_id
                if confidence is not None:
                    node["confidence"] = confidence
                nodes.append(node)
            elif obj_id == "dynamic_line":
                edge = {
                    "id": instance,
                    "type": obj_id,
                    "graph_id": graph_id,
                    "source": anchors_to_node[latches_to_anchor[obj.get("source")]],
                    "target": anchors_to_node[latches_to_anchor[obj.get("target")]]
                }
                edges.append(edge)

    with driver.session(database=NEO4J_DATABASE) as session:
        session.execute_write(import_graph, nodes, edges)

    print(f"Imported {len(nodes)} nodes and {len(edges)} edges into the Neo4j database.")


if __name__ == "__main__":
    main()


# view above graph in Neo4j Browser
'''
MATCH (n:AttackFlowNode)-[r:AttackFlowEdge {graph_id: "af_blackbasta_ransomware"}]->(m:AttackFlowNode)
RETURN n, r, m
'''
