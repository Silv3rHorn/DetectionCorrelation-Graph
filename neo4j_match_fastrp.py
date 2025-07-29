import argparse
import os
import pandas as pd
import numpy as np

from graphdatascience import GraphDataScience

# --- Connection Details ---
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_DATABASE = "graphmatching"
if os.environ.get("NEO4J_PASSWORD"):
    NEO4J_AUTH = (NEO4J_USERNAME, os.environ.get("NEO4J_PASSWORD"))


def main():
    parser = argparse.ArgumentParser(description="Compare graph embeddings using FastRP and Neo4j GDS.")
    parser.add_argument('--mode', choices=['normal', 'test'], default='normal', help='normal or test mode (default: normal)')
    parser.add_argument('--af_graph_id', '-r', help='AttackFlow graph_id (required for normal mode)')
    parser.add_argument('--detection_graph_ids', '-d', help='Comma-separated Detection graph_ids (required for normal mode)')
    parser.add_argument('--g1', help='First graph_id (required for test mode)')
    parser.add_argument('--g2', help='Second graph_id (required for test mode)')
    parser.add_argument('--random', action='store_true', help='If set, do not use randomSeed in FastRP config')
    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'normal':
        if not args.af_graph_id or not args.detection_graph_ids:
            parser.error('In normal mode, --af_graph_id and --detection_graph_ids are required.')
        detection_ids = [x.strip() for x in args.detection_graph_ids.split(',') if x.strip()]
        if not detection_ids:
            parser.error('At least one detection_graph_id must be specified.')
    elif args.mode == 'test':
        if not args.g1 or not args.g2:
            parser.error('In test mode, --g1 and --g2 are required.')

    # Establish connection to the database and GDS
    gds = GraphDataScience(NEO4J_URL, auth=NEO4J_AUTH, database=NEO4J_DATABASE)

    def project_graph(graph_name, node_label, graph_id):
        gds.graph.project.cypher(
            graph_name,
            f'''
            MATCH (n:{node_label})
            WHERE n.graph_id = "{graph_id}"
            WITH n, split(substring(COALESCE(n.technique_id, 'T0'), 1), '.') AS parts
            WITH n,
                 toFloat(parts[0]) AS main_part,
                 (CASE WHEN size(parts) > 1 THEN toFloat(parts[1]) / 1000.0 ELSE 0.0 END) AS sub_part
            RETURN id(n) AS id, main_part + sub_part AS technique_id
            ''',
            f'''
            MATCH (n:{node_label})-[r]- (m:{node_label})
            WHERE r.graph_id = "{graph_id}"
            RETURN id(n) AS source, id(m) AS target
            '''
        )

    def get_average_embedding(graph_name, config):
        try:
            g = gds.graph.get(graph_name)
            if g.node_count() == 0:
                print(f"Warning: Graph '{graph_name}' is empty. Skipping embedding generation.")
                return None
            embeddings_df = gds.fastRP.stream(g, **config)  # has a single column, where each row contains the embedding vector for each node
        except Exception as e:
            print(f"Warning: Could not generate embeddings for graph '{graph_name}'. Error: {e}")
            return None
        if embeddings_df.empty:
            return None
        # converts the column of embedding vectors into a python list of lists - [[0.1, 0.5, ...], [0.9, 0.2, ...], ...]
        # converts python list into numpy array (2d matrix where each row is a node's embedding)
        embeddings = np.array(embeddings_df["embedding"].tolist())
        # calculate the mean down each column
        # take first number from every embedding vector, and average them, repeat for second number and so on
        return embeddings.mean(axis=0)  # result is a single vector (a 1D array of 256 numbers)

    def cosine_similarity(vec1, vec2):
        if vec1 is None or vec2 is None:
            return None
        norm1 = np.linalg.norm(vec1)  # calculates the norm or magnitude of a vector
        norm2 = np.linalg.norm(vec2)  # calculates the norm or magnitude of a vector
        if norm1 == 0 or norm2 == 0:
            return None
        # Cosine Similarity = (A ⋅ B) / (||A|| ||B||)
        # where A and B are vectors, A ⋅ B is their dot product, and ||A|| and ||B|| are their magnitudes
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    # FastRP config
    fastrp_stream_config = {
        "embeddingDimension": 256,
        "iterationWeights": [1.0, 1.0, 0.8, 0.6, 0.4, 0.2],
        "featureProperties": ['technique_id'],
        "propertyRatio": 0.0,  # controls the influence of technique_id on the embeddings
        # "normalizationStrength": -0.2  # control the influence of common technique_ids on the embeddings
    }
    if not args.random:
        fastrp_stream_config["randomSeed"] = 42

    try:
        if args.mode == 'normal':
            # Project AttackFlow graph
            af_graph_name = f"af_{args.af_graph_id}_projection"
            if gds.graph.exists(af_graph_name)["exists"]:
                gds.graph.get(af_graph_name).drop()
            project_graph(af_graph_name, "AttackFlowNode", args.af_graph_id)
            avg_emb_af = get_average_embedding(af_graph_name, fastrp_stream_config)
            print(f"Average embedding for AttackFlow graph ({args.af_graph_id}):")
            print(avg_emb_af if avg_emb_af is not None else "No embeddings found.")

            # Project and compare each Detection graph
            for det_id in detection_ids:
                det_graph_name = f"det_{det_id}_projection"
                if gds.graph.exists(det_graph_name)["exists"]:
                    gds.graph.get(det_graph_name).drop()
                project_graph(det_graph_name, "DetectionNode", det_id)
                avg_emb_det = get_average_embedding(det_graph_name, fastrp_stream_config)
                print(f"\nAverage embedding for Detection graph ({det_id}):")
                print(avg_emb_det if avg_emb_det is not None else "No embeddings found.")
                cos_sim = cosine_similarity(avg_emb_af, avg_emb_det)
                print(f"Cosine similarity between AttackFlow ({args.af_graph_id}) and Detection ({det_id}): {cos_sim if cos_sim is not None else 'N/A'}")

            # Cleanup
            print("\nCleaning up resources...")
            if gds.graph.exists(af_graph_name)["exists"]:
                gds.graph.get(af_graph_name).drop()
            for det_id in detection_ids:
                det_graph_name = f"det_{det_id}_projection"
                if gds.graph.exists(det_graph_name)["exists"]:
                    gds.graph.get(det_graph_name).drop()
            print("Cleanup finished.")

        elif args.mode == 'test':
            g1_name = f"test_{args.g1}_projection"
            g2_name = f"test_{args.g2}_projection"
            if gds.graph.exists(g1_name)["exists"]:
                gds.graph.get(g1_name).drop()
            if gds.graph.exists(g2_name)["exists"]:
                gds.graph.get(g2_name).drop()
            # Always use DetectionNode for test mode
            project_graph(g1_name, "DetectionNode", args.g1)
            project_graph(g2_name, "DetectionNode", args.g2)
            avg_emb_1 = get_average_embedding(g1_name, fastrp_stream_config)
            avg_emb_2 = get_average_embedding(g2_name, fastrp_stream_config)
            print(f"Average embedding for Graph 1 ({args.g1}):")
            print(avg_emb_1 if avg_emb_1 is not None else "No embeddings found.")
            print(f"\nAverage embedding for Graph 2 ({args.g2}):")
            print(avg_emb_2 if avg_emb_2 is not None else "No embeddings found.")
            cos_sim = cosine_similarity(avg_emb_1, avg_emb_2)
            print(f"\nCosine similarity between Graph 1 ({args.g1}) and Graph 2 ({args.g2}): {cos_sim if cos_sim is not None else 'N/A'}")

            # Cleanup
            print("\nCleaning up resources...")
            if gds.graph.exists(g1_name)["exists"]:
                gds.graph.get(g1_name).drop()
            if gds.graph.exists(g2_name)["exists"]:
                gds.graph.get(g2_name).drop()
            print("Cleanup finished.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if 'gds' in locals() and gds:
            gds.close()


if __name__ == "__main__":
    main()
