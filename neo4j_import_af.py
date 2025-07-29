import argparse
import json
import os
import requests
from neo4j import GraphDatabase

# --- Neo4j Configuration ---
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_DATABASE = "graphmatching"
if os.environ.get("NEO4J_PASSWORD"):
    NEO4J_AUTH = (NEO4J_USERNAME, os.environ.get("NEO4J_PASSWORD"))
else:
    raise ValueError("NEO4J_PASSWORD environment variable is not set.")
# --- MITRE ATT&CK Data ---
MITRE_ATTACK_DATA = requests.get('https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json').json()
TECHNIQUES = {
    technique['external_references'][0]['external_id']: technique['name']
    for technique in MITRE_ATTACK_DATA['objects']
    if technique['type'] == 'attack-pattern' and not technique.get('revoked')
}


def get_technique_name(tid):
    """
    Looks up the name of a technique or sub-technique from the pre-loaded data.
    Formats sub-technique names as 'Technique: Sub-Technique'.
    """
    if not tid:
        return "Unknown Technique"
    try:
        # For sub-techniques (e.g., T1234.001), get the parent ID (T1234)
        if '.' in tid:
            parent_id = tid.split('.')[0]
            return f"{TECHNIQUES[parent_id]}: {TECHNIQUES[tid]}"
        return TECHNIQUES[tid]
    except KeyError:
        print(f"Warning: Unknown technique ID '{tid}'")
        return "Unknown Technique"


def import_graph(tx, nodes, edges):
    """
    Neo4j transaction function to import nodes and edges.
    Uses MERGE to avoid creating duplicate techniques.
    """
    for node in nodes:
        tx.run(
            """
            MERGE (n:Technique {technique_id: $technique_id})
            ON CREATE SET n.name = $name
            ON MATCH SET n.name = $name
            """,
            technique_id=node.get("technique_id"),
            name=node.get("name")
        )

    for edge in edges:
        tx.run(
            """
            MATCH (src:Technique {technique_id: $source})
            MATCH (dst:Technique {technique_id: $target})
            MERGE (src)-[r:AttackFlow {
                graph_id: $graph_id,
                source: $source,
                target: $target,
                type: $type,
                conditionId: $conditionId
            }]->(dst)
            """,
            graph_id=edge.get("graph_id"),
            source=edge.get("source"),
            target=edge.get("target"),
            type=edge.get("type"),
            conditionId=edge.get("conditionId")
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

    # Node (src, target) -> dynamic_line -> latches -> anchors -> Node (action, operator)
    # build lookup tables to store latches-anchor, anchors-node, ids-technique, and operator-target mapping
    latches_to_anchor, anchors_to_node, ids_to_technique, operator_to_target = {}, {}, {}, {}
    actions, operators, dynamic_lines = {}, {}, {}

    # First pass: collect latches_to_anchor, anchors_to_node, ids_to_technique, actions, operators, edges
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
            if obj_id == "action":
                props = {p[0]: p[1] for p in obj.get("properties", [])}
                technique_id = props.get("technique_id")
                if technique_id:
                    ids_to_technique[instance] = technique_id
                actions[instance] = obj
            elif obj_id in ["AND_operator", "OR_operator"]:
                operators[instance] = obj
            elif obj_id == "dynamic_line":
                dynamic_lines[instance] = obj

    # Second pass: collect operator to target mapping
    for inst, obj in dynamic_lines.items():
        source = obj.get("source")
        target = obj.get("target")
        try:
            anchor_src = latches_to_anchor[source]
            anchor_target = latches_to_anchor[target]
            node_src = anchors_to_node[anchor_src]
            node_target = anchors_to_node[anchor_target]
        except KeyError as e:
            print(f"Error tracing the source and target of dynamic_line: {inst}")
            continue
        if node_src in operators.keys() and node_target in actions.keys():
            operator_to_target[node_src] = node_target

    nodes, edges = [], []
    graph_id = args.graph_id
    # Third pass: build nodes for Neo4j import
    for inst, obj in actions.items():
        obj_id = obj.get("id")
        props = {p[0]: p[1] for p in obj.get("properties", [])}
        if props.get("confidence") == "certain":
            node = {"type": obj_id, "technique_id": props.get("technique_id")}
            node['name'] = get_technique_name(node["technique_id"])
            nodes.append(node)

    for inst, obj in dynamic_lines.items():
        try:
            source_id = anchors_to_node[latches_to_anchor[obj.get("source")]]
            target_id = anchors_to_node[latches_to_anchor[obj.get("target")]]
            if target_id in operators.keys():
                condition_id = target_id
                target_id = operator_to_target[target_id]
                if operators[target_id].get("id") == "AND_operator":
                    edge_type = "Indirect_AND"
                elif operators[target_id].get("id") == "OR_operator":
                    edge_type = "Indirect_OR"
            else:
                condition_id = "NIL"
                edge_type = "Direct"
            edge = {
                "type": obj.get("id"),
                "graph_id": graph_id,
                "edge_type": edge_type,
                "condition_id": condition_id,
                "source":  ids_to_technique[source_id],
                "target":  ids_to_technique[target_id]
            }
        except KeyError as e:
            print(f"Error processing edge: {inst}")
            continue
        edges.append(edge)

    with driver.session(database=NEO4J_DATABASE) as session:
        session.execute_write(import_graph, nodes, edges)
    
    # troubleshooting
    # print(f"Imported {len(nodes)} nodes and {len(edges)} edges into the Neo4j database.")
    # print("Actions:", actions.keys())
    # print("Edges:", dynamic_lines.keys())
    # print("Operators:", operators.keys())
    # print("Latches to Anchor:", latches_to_anchor)
    # print("Anchors to Node:", anchors_to_node)
    # print("IDs to Technique:", ids_to_technique)
    # print("Operator to Target:", operator_to_target)


if __name__ == "__main__":
    main()


# view above graph in Neo4j Browser
'''
MATCH (n:Technique)-[r:AttackFlow {graph_id: "af_blackbasta_ransomware"}]->(m:Technique)
RETURN n, r, m
'''
