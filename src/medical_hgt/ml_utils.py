import openai
import torch
import heapq

from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


def find_most_relevant_nodes(batch, z_dict, question_nodes_embedding, subgraph_tuples, prime_gk, k=3):

    # Using a heap for efficient minimum distance tracking
    relevant_nodes_heap = []
    heapq.heapify(relevant_nodes_heap)

    for node_uid, node_type in subgraph_tuples:

        if len(batch[node_type]) == 0:
            continue
        node_index = torch.where(batch[node_type].node_uid == node_uid)[0][0]
        node_embeddings = z_dict[node_type][node_index]

        # Calculate distance
        distance = torch.norm(question_nodes_embedding - node_embeddings, p=2)

        # Get node information
        node_info = prime_gk.nodes[node_uid]
        node_info_string = f"The {node_info['type']} {node_info['name']}"

        # Update the relevant nodes list and distances
        if len(relevant_nodes_heap) < k:
            heapq.heappush(relevant_nodes_heap, (distance, node_info_string))
        else:
            # Only update if the current distance is greater than the smallest in the heap
            if distance > relevant_nodes_heap[0][0]:
                heapq.heappop(relevant_nodes_heap)
                heapq.heappush(relevant_nodes_heap, (distance, node_info_string))

    # Convert heap to a list of relevant nodes, sorting by distance
    relevant_nodes_list = [node for _, node in sorted(relevant_nodes_heap, reverse=True)]

    return relevant_nodes_list
