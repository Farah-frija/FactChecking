from typing import Any, Dict, List, Tuple
from pydantic import Field
from datetime import date
from sklearn.metrics.pairwise import cosine_similarity
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from pyvis.network import Network
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
import asyncio
import Data.Process_data as process
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_community.graphs.graph_document import Node, Relationship
vectors={}
# Load the .env file
load_dotenv()

# Define date formats
date_formats = [
    "2025-05-13 15:33:02+00:00",
    "2024-12-25 10:30:45+00:00", 
    "2023-07-04 08:15:20+00:00",
    "2022-01-01 00:00:00+00:00",
    "2021-11-11 11:11:11+00:00"
]

# Initialize LLM and graph transformer
llm = ChatOllama(temperature=0, model="mistral")
relationship_properties = {
    "date": Field(
        ...,
        description=f"date: format YYYY-MM-DD HH:MM:SS+00:00. Format examples are {date_formats}"
    )
}

graph_transformer = LLMGraphTransformer(llm=llm, relationship_properties=relationship_properties)
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', token='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')

async def get_sentence_embeddings(texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
    """
    Asynchronously calculates sentence embeddings for a list of texts using SentenceTransformer.
    """
    def _compute_embeddings_sync(text_batch: List[str]) -> np.ndarray:
        return model.encode(text_batch, convert_to_numpy=True)
    
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = await asyncio.get_event_loop().run_in_executor(
            None, _compute_embeddings_sync, batch
        )
        embeddings.extend(batch_embeddings)
    
    return embeddings
async def get_tag_embeddings(texts: List[List[str]], batch_size: int = 32) -> List[np.ndarray]:
    """
    Asynchronously calculates sentence embeddings for a list of texts using SentenceTransformer.
    """
    print('textss',texts)
    def _compute_embeddings_sync(text_batch: List[List[str]]) -> np.ndarray:
        all_mean_embeddings = []
        for inner_list in text_batch:
            if len(inner_list) == 0:
                # Handle empty inner list by returning a zero vector or skipping
                mean_embedding = np.zeros((768,))
            else:
                embeddings = model.encode(inner_list, convert_to_numpy=True)  # shape (len(inner_list), 768)
                mean_embedding = embeddings.mean(axis=0)  # average across texts in inner list
            all_mean_embeddings.append(mean_embedding)
        return np.vstack(all_mean_embeddings)  # shape (batch_size, 768)


    
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = await asyncio.get_event_loop().run_in_executor(
            None, _compute_embeddings_sync, batch
        )
        embeddings.extend(batch_embeddings)
    print('calculated_embeddings',np.array(embeddings).shape)
    return np.array(embeddings)

# Extract graph data from input text
async def extract_graph_data(start: int, limit: int):
    """
    Asynchronously extracts graph data from input text using a graph transformer.
    """
    Data = process.Data
    Data = Data[start:limit + 1]
    documents = []
    texts = []
    metadata_list = []
    tags=[]
    for data in Data:
        text = data['combined_text']
        tag=data['tags_split']
        print('Processing text:', text[:100] + '...' if len(text) > 100 else text)
        
        doc_metadata = {
            'tags_split': data.get('tags_split', []),
            'source': data.get('source', 'unknown'),
            'pubDate': data.get('pubDate', None),
            'original_data': data
        }
        
        document = Document(page_content=text, metadata=doc_metadata)
        documents.append(document)
        texts.append(text)
        tags.append(tag)
        metadata_list.append(doc_metadata)
    tags_embeddings= await get_tag_embeddings(tags)
    embeddings = await get_sentence_embeddings(texts)
    graph_documents = await graph_transformer.aconvert_to_graph_documents(documents)
    
    enhanced_graph_documents = []
    
    for i, graph_doc in enumerate(graph_documents):
        enhanced_doc = {
            'graph_document': graph_doc,
            'metadata': metadata_list[i],
            'embedding': embeddings[i] if i < len(embeddings) else None,
            'news_id': start + i,
            'embedding_tag': tags_embeddings[i] if i < len(embeddings) else None,
        }
        enhanced_graph_documents.append(enhanced_doc)
        news_id= start + i
        vectors[news_id]=embeddings[i] 
    print(f"Processed {len(enhanced_graph_documents)} graph documents")
    
    return {
        'enhanced_graph_documents': enhanced_graph_documents,
        'embeddings': embeddings,
        'tags_embeddings':tags_embeddings,
        'texts': texts,
        'metadata_list': metadata_list
    }

def enrich_graph_with_news_ids(enhanced_graph_documents, start_index: int) -> Tuple[List[Node], List[Node], List[Relationship]]:
    """
    Enriches graph by creating 'has' relationships from news nodes to content nodes.
    Returns proper Node and Relationship objects.
    """
    enriched_nodes = []
    news_metadata_nodes = []
    has_relationships = []
    
    for i, enhanced_doc in enumerate(enhanced_graph_documents):
        graph_doc = enhanced_doc['graph_document']
        metadata = enhanced_doc['metadata']
        embedding = enhanced_doc['embedding']
        news_id = start_index + i
        embedding_tag=enhanced_doc['embedding_tag']
        
        current_news_nodes = []
        
        # Process all nodes in this graph document
        for node in graph_doc.nodes:
            if not hasattr(node, 'properties'):
                node.properties = {}
            
            node.properties.update({
                'source': metadata.get('source', 'unknown'),
                'original_news_index': i
            })
            
            current_news_nodes.append(node)
            enriched_nodes.append(node)
        
        # Create a dedicated news metadata node as a proper Node object
        news_metadata_node = Node(
            id=f'news_{news_id}',
            type='news_metadata',
            properties={
                'news_id': news_id,
                'pub_date': metadata.get('pubDate', 'unknown'),
                'source': metadata.get('source', 'unknown'),
                'tags': metadata.get('tags_split', []),
                'embedding': embedding,
                'text_preview': enhanced_doc.get('text', '')[:200] + '...' if enhanced_doc.get('text') and len(enhanced_doc.get('text')) > 200 else enhanced_doc.get('text', ''),
                'node_count': len(current_news_nodes),
                'original_index': i,
                'embedding_tag':embedding_tag
            }
        )
        
        news_metadata_nodes.append(news_metadata_node)
        
        # Create 'has' relationships as proper Relationship objects
        for content_node in current_news_nodes:
            has_relationship = Relationship(
                source=news_metadata_node,
                target=content_node,
                type='has',
                properties={
                    'relationship_type': 'contains',
                    'direction': 'news_to_content'
                }
            )
            has_relationships.append(has_relationship)
        
        # Add the original relationships from the graph document
        for rel in graph_doc.relationships:
            has_relationships.append(rel)
        
        print(f"Processed news ID {news_id} from {metadata.get('source', 'unknown')} with {len(current_news_nodes)} nodes and {len(current_news_nodes)} 'has' relationships")
    
    return enriched_nodes, news_metadata_nodes, has_relationships

async def calculate_news_similarities(news_nodes: List[Node], similarity_threshold: float = 0) -> List[Relationship]:
    """
    Calculates cosine similarities between news nodes and creates bidirectional edges as Relationship objects.
    """
    def _compute_similarities_sync():
        embeddings = []
        valid_indices = []
        
        for i, node in enumerate(news_nodes):
            embedding = node.properties.get('embedding_tag')
            if embedding is not None and len(embedding) > 0:
                embeddings.append(embedding)
                valid_indices.append(i)
        
        if not embeddings:
            return []
        
        embeddings_array = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        edges = []
        n = len(embeddings_array)
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity >= similarity_threshold:
                    node_i = news_nodes[valid_indices[i]]
                    node_j = news_nodes[valid_indices[j]]
                    
                    # Create bidirectional edges as Relationship objects
                    edge_1 = Relationship(
                        source=node_i,
                        target=node_j,
                        type='similarity',
                        properties={
                            'similarity_score': float(similarity),
                            'direction': 'bidirectional'
                        }
                    )
                    edge_2 = Relationship(
                        source=node_j,
                        target=node_i,
                        type='similarity',
                        properties={
                            'similarity_score': float(similarity),
                            'direction': 'bidirectional'
                        }
                    )
                    
                    edges.extend([edge_1, edge_2])
        
        return edges
    def _compute_tags_similarities_sync():
        embeddings = []
        valid_indices = []
        
        for i, node in enumerate(news_nodes):
            embedding = node.properties.get('embedding_tag')
            if embedding is not None:
                embeddings.append(embedding)
                valid_indices.append(i)
        
        if not embeddings:
            return []
        
        embeddings_array = np.array(embeddings)
        print('shapee',embeddings_array.shape)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        edges = []
        n = len(embeddings_array)
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity >= similarity_threshold:
                    node_i = news_nodes[valid_indices[i]]
                    node_j = news_nodes[valid_indices[j]]
                    
                    # Create bidirectional edges as Relationship objects
                    edge_1 = Relationship(
                        source=node_i,
                        target=node_j,
                        type='similarity',
                        properties={
                            'similarity_score': float(similarity),
                            'direction': 'bidirectional'
                        }
                    )
                    edge_2 = Relationship(
                        source=node_j,
                        target=node_i,
                        type='similarity',
                        properties={
                            'similarity_score': float(similarity),
                            'direction': 'bidirectional'
                        }
                    )
                    
                    edges.extend([edge_1, edge_2])
        
        return edges
    
    edges = await asyncio.get_event_loop().run_in_executor(
        None, _compute_tags_similarities_sync
    )
    
    print(f"Created {len(edges)} similarity edges with threshold {similarity_threshold}")
    return edges

# Update the complete workflow function
async def process_news_graph_with_similarities(start: int, limit: int, similarity_threshold: float = 0):
    """
    Complete workflow: Extract graph data, create news relations, and calculate similarities.
    """
    extraction_result = await extract_graph_data(start, limit)
    enhanced_docs = extraction_result['enhanced_graph_documents']
    
    enriched_nodes, news_metadata_nodes, has_relationships = enrich_graph_with_news_ids(enhanced_docs, start)
    #similarity_edges = await calculate_news_similarities(news_metadata_nodes, similarity_threshold)
    
    return {
        'enhanced_docs':enhanced_docs,
        'enriched_nodes': enriched_nodes,
        'news_metadata_nodes': news_metadata_nodes,
        'has_relationships': has_relationships,
        #'similarity_edges': similarity_edges,
        'total_news_processed': len(news_metadata_nodes),
        'total_has_relationships': len(has_relationships),
        'extraction_metadata': {
            'start_index': start,
            'limit': limit,
            'similarity_threshold': similarity_threshold
        }
    }

def get_color_for_type(node_type: str) -> str:
    """Returns a color based on node type for visualization."""
    color_map = {
        'PERSON': '#FF9A76',
        'ORGANIZATION': '#A3D8F4',
        'LOCATION': '#9DEF96',
        'EVENT': '#FFD166',
        'DATE': '#B392AC',
        'news_metadata': '#FF6B6B'
    }
    return color_map.get(node_type, '#CCCCCC')

def visualize_graph(enriched_nodes, news_metadata_nodes, has_relationships):
    """
    Visualizes the complete knowledge graph including news metadata, has relationships, and similarity edges.
    """
    net = Network(height="1200px", width="100%", directed=True,
                  notebook=False, bgcolor="#222222", font_color="white", filter_menu=True, cdn_resources='remote')
    
    # Add enriched content nodes
    for node in enriched_nodes:
        try:
            net.add_node(
                node.id, 
                label=node.id, 
                title=f"{node.type}\nSource: {node.properties.get('source', 'unknown')}",
                group=node.type,
                color=get_color_for_type(node.type)
            )
        except Exception as e:
            print(f"Error adding node {node.id}: {e}")
            continue
    
    # Add news metadata nodes
    for news_node in news_metadata_nodes:
        try:
            net.add_node(
                news_node.id,
                label=f"News_{news_node.properties['news_id']}",
                title=f"Source: {news_node.properties['source']}\n"
                      f"Date: {news_node.properties['pub_date']}\n"
                      f"Tags: {', '.join(news_node.properties['tags'])}\n"
                      f"Nodes: {news_node.properties['node_count']}",
                group='news_metadata',
                color='#FF6B6B',
                shape='box',
                size=25
            )
        except Exception as e:
            print(f"Error adding news node {news_node.id}: {e}")
            continue
    
    # Add 'has' relationships (news -> content)
    for rel in has_relationships:
        try:
            net.add_edge(
                rel.source.id,
                rel.target.id,
                label=rel.type,
                color='#6A0572',
                width=2,
                arrows='to',
                title=f"{rel.type}"
            )
        except Exception as e:
            print(f"Error adding has relationship: {e}")
            continue
    
    # Add similarity edges
   
    
    # Configure graph layout
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 150,
                "springConstant": 0.05
            },
            "minVelocity": 0.5,
            "solver": "forceAtlas2Based"
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200
        }
    }
    """)
    
    output_file = "knowledge_graph.html"
    try:
        net.save_graph(output_file)
        print(f"Enhanced graph with relations saved to {os.path.abspath(output_file)}")
        return net
    except Exception as e:
        print(f"Error saving graph: {e}")
        return None

# Update the main generation function
async def generate_knowledge_graph(start: int, limit: int, similarity_threshold: float = 0):
    """
    Generates and visualizes a complete knowledge graph with news relations and similarities.
    """
    result = await process_news_graph_with_similarities(start, limit, similarity_threshold)
    
    net = visualize_graph(
        result['enriched_nodes'],
        result['news_metadata_nodes'],
        result['has_relationships'],
        #result['similarity_edges']
    )
    
    print(f"Processed {result['total_news_processed']} news articles")
    print(f"Created {result['total_has_relationships']} 'has' relationships")

    
    return net,result['enhanced_docs']
