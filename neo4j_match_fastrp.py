import pandas as pd
import numpy as np
from graphdatascience import GraphDataScience

# --- Connection Details ---
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "<password>"
NEO4J_DATABASE = "graphmatching"
# --- GDS Configuration ---
# Name for the in-memory graph projections
GRAPH1_NAME = "attackflow_projection"
GRAPH2_NAME = "d_3_3_projection"
GRAPH3_NAME = "d_3_0_projection"


def compare_graphs_with_fastrp(gds: GraphDataScience):
    """
    Compares two graphs using the FastRP algorithm and returns the similarity scores.
    """
    # ensure a clean slate by dropping graphs if they already exist
    for graph_name in [GRAPH1_NAME, GRAPH2_NAME, GRAPH3_NAME]:
        if gds.graph.exists(graph_name)["exists"]:
            gds.graph.get(graph_name).drop()
            print(f"Dropped existing graph projection: {graph_name}")

    # --- 1. Project graphs into GDS memory ---
    print("Step 1: Projecting graphs into GDS memory...")
    # load the first graph projection
    gds.graph.project.cypher(
        GRAPH1_NAME,
        """
        MATCH (n:AttackFlowNode)
        WHERE n.graph_id = "af_blackbasta_ransomware"
        WITH n, split(substring(COALESCE(n.technique_id, 'T0'), 1), '.') AS parts
        WITH n,
             toFloat(parts[0]) AS main_part,
             (CASE WHEN size(parts) > 1 THEN toFloat(parts[1]) / 1000.0 ELSE 0.0 END) AS sub_part
        RETURN id(n) AS id, main_part + sub_part AS technique_id
        """,
        """
        MATCH (n:AttackFlowNode)-[r:AttackFlowEdge]->(m:AttackFlowNode)
        WHERE r.graph_id = "af_blackbasta_ransomware"
        RETURN id(n) AS source, id(m) AS target
        """,
    )
    # load the second graph projection
    gds.graph.project.cypher(
        GRAPH2_NAME,
        """
        MATCH (n:DetectionNode)
        WHERE n.graph_id = "d_3_3"
        WITH n, split(substring(COALESCE(n.technique_id, 'T0'), 1), '.') AS parts
        WITH n,
             toFloat(parts[0]) AS main_part,
             (CASE WHEN size(parts) > 1 THEN toFloat(parts[1]) / 1000.0 ELSE 0.0 END) AS sub_part
        RETURN id(n) AS id, main_part + sub_part AS technique_id
        """,
        """
        MATCH (n:DetectionNode)-[r:DetectionEdge]->(m:DetectionNode)
        WHERE r.graph_id = "d_3_3"
        RETURN id(n) AS source, id(m) AS target
        """,
    )
    # load the third graph projection
    gds.graph.project.cypher(
        GRAPH3_NAME,
        """
        MATCH (n:DetectionNode)
        WHERE n.graph_id = "d3_3_3"
        WITH n, split(substring(COALESCE(n.technique_id, 'T0'), 1), '.') AS parts
        WITH n,
             toFloat(parts[0]) AS main_part,
             (CASE WHEN size(parts) > 1 THEN toFloat(parts[1]) / 1000.0 ELSE 0.0 END) AS sub_part
        RETURN id(n) AS id, main_part + sub_part AS technique_id
        """,
        """
        MATCH (n:DetectionNode)-[r:DetectionEdge]->(m:DetectionNode)
        WHERE r.graph_id = "d3_3_3"
        RETURN id(n) AS source, id(m) AS target
        """,
    )
    print("Graph projections created successfully.")

    # --- 2. Generate Node Embeddings with FastRP & Compute Average Embeddings for each Graph---
    print("\nStep 2: Generating node embeddings using FastRP...Computing average embeddings for each graph...")
    fastrp_stream_config = {
        "embeddingDimension": 256,
        "iterationWeights": [1.0, 1.0, 0.8, 0.6, 0.4, 0.2],
        "featureProperties": ['technique_id'],  # Use 'featureProperties' for stream mode
        # "normalizationStrength": 0.2,  # control the influence of common technique_ids on the embeddings
        # "randomSeed": 42  # for reproducibility
    }
    # By using gds.fastRP.stream(), we can calculate embeddings and get them
    # directly without writing them back to the database.
    def get_average_embedding(graph_name: str, config: dict):
        """Streams embeddings for a graph and computes the average."""
        try:
            g = gds.graph.get(graph_name)
            # Explicitly check for an empty graph to provide a clear warning.
            if g.node_count() == 0:
                print(f"Warning: Graph '{graph_name}' is empty. Skipping embedding generation.")
                return None
            embeddings_df = gds.fastRP.stream(g, **config)
        except Exception as e:
            # GDS can throw an error if the graph is empty or other issues occur.
            print(f"Warning: Could not generate embeddings for graph '{graph_name}'. Error: {e}")
            return None
        if embeddings_df.empty:
            return None
        # The 'embedding' column contains the vector for each node.
        # We convert it to a NumPy array and compute the mean across all nodes.
        embeddings = np.array(embeddings_df["embedding"].tolist())
        return embeddings.mean(axis=0)
    avg_emb_g1 = get_average_embedding(GRAPH1_NAME, fastrp_stream_config)
    avg_emb_g2 = get_average_embedding(GRAPH2_NAME, fastrp_stream_config)
    avg_emb_g3 = get_average_embedding(GRAPH3_NAME, fastrp_stream_config)
    print(f"Average embedding for Graph1 (AttackFlowNode, af_blackbasta_ransomware):")
    print(avg_emb_g1 if avg_emb_g1 is not None else "No embeddings found.")
    print(f"\nAverage embedding for Graph2 (DetectionNode, d_3_3):")
    print(avg_emb_g2 if avg_emb_g2 is not None else "No embeddings found.")
    print(f"\nAverage embedding for Graph3 (DetectionNode, d_3_0):")
    print(avg_emb_g3 if avg_emb_g3 is not None else "No embeddings found.")

    # -- 3. Compute Cosine Similarity of Embeddings --
    print("\nStep 3: Computing cosine similarity between embeddings...")
    def cosine_similarity(vec1, vec2):
        if vec1 is None or vec2 is None:
            return None
        norm1 = np.linalg.norm(vec1)  # calculates the norm or magnitude of a vector
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return None
        # Cosine Similarity = (A ⋅ B) / (||A|| ||B||), where A and B are vectors, A ⋅ B is their dot product, and ||A|| and ||B|| are their magnitudes
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    cos_sim_g1_g2 = cosine_similarity(avg_emb_g1, avg_emb_g2)
    cos_sim_g1_g3 = cosine_similarity(avg_emb_g1, avg_emb_g3)
    cos_sim_g2_g3 = cosine_similarity(avg_emb_g2, avg_emb_g3)
    print(
    f"\nCosine similarity between Graph1 and Graph2 (average embeddings): {cos_sim_g1_g2 if cos_sim_g1_g2 is not None else 'N/A'}"
    )
    print(
    f"Cosine similarity between Graph1 and Graph3 (average embeddings): {cos_sim_g1_g3 if cos_sim_g1_g3 is not None else 'N/A'}"
    )
    print(
    f"Cosine similarity between Graph2 and Graph3 (average embeddings): {cos_sim_g2_g3 if cos_sim_g2_g3 is not None else 'N/A'}"
    )

    # --- 5. Cleanup ---
    print("\nStep 4: Cleaning up resources...")
    for graph_name in [GRAPH1_NAME, GRAPH2_NAME, GRAPH3_NAME]:
        if gds.graph.exists(graph_name)["exists"]:
            gds.graph.get(graph_name).drop()
    # Since we used gds.fastRP.stream(), we don't need to remove the embedding
    # property from the database, as it was never written.
    print("Cleanup finished.")


if __name__ == "__main__":
    try:
        # Establish connection to the database and GDS
        gds = GraphDataScience(NEO4J_URL, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), database=NEO4J_DATABASE)
        compare_graphs_with_fastrp(gds)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if 'gds' in locals() and gds:
            gds.close()
