# quick_insert.py
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import Data.Process_data as process
def quick_insert_to_qdrant(json_file_path: str):
    """Quick function to insert JSON data into Qdrant"""
    
    # Initialize clients
    client = QdrantClient(
        url="'https://e7ccdcf9-79a9-4693-8990-8dbd4464de96.europe-west3-0.gcp.cloud.qdrant.io:6333",
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.5hLPP46Qw0Cda9nX4cV8mEMxF97Vv3MsWAp_hCnel0I"
    )
    
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    # Create collection
    collection_name = "news_collection"
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
    
    # Load data
    data = process.Data
    records = data if isinstance(data, list) else data.get('documents', [])
    
    # Process and insert
    points = []
    records=records[1341:]
    for i, record in enumerate(records):
        
        text = record.get('combined_text', '')
        embedding = model.encode(text).tolist()
        
        point = PointStruct(
            id=i + 1,
            vector=embedding,
            payload={
                'combined_text': text,
                'tags_split': record.get('tags_split', []),
                'source': record.get('source', 'unknown'),
                'pubDate': record.get('pubDate', ''),
            }
        )
        points.append(point)
        
        # Insert in batches of 100
        if len(points) >= 10:
            client.upsert(collection_name=collection_name, points=points)
            points = []
            print(f"Inserted up to record {i + 1}")
    
    # Insert remaining points
    if points:
        client.upsert(collection_name=collection_name, points=points)
    
    print(f"Inserted {len(records)} records into Qdrant")

# Run it
quick_insert_to_qdrant("Data/news_data_docs.json")