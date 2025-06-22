"""
api.py

FastAPI app for serving recommendations and graph queries.
"""
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from recommender_system.gnn_recommender import SimpleGCN, fetch_graph
import os

app = FastAPI()

# Global variables to hold the model and graph data
model = None
data = None
node_list = None
node_id_map = None

@app.on_event("startup")
def load_model():
    """Load the GNN model and graph data at startup."""
    global model, data, node_list, node_id_map
    
    print("Loading GNN model and graph data...")
    
    # 1. Fetch graph data
    data = fetch_graph()
    
    # 2. Initialize model architecture
    model = SimpleGCN(num_features=data.num_features, hidden_channels=16)
    
    # 3. Load the trained model weights
    model_path = "model/recommender_model.pth"
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model file '{model_path}' not found. "
            "Please run gnn_recommender.py to train and save the model first."
        )
    model.load_state_dict(torch.load(model_path))
        
    # 4. Set model to evaluation mode
    model.eval()
    print("GNN model and graph data loaded successfully.")


class RecommendationRequest(BaseModel):
    user_id: int # This is the Neo4j internal ID for the user
    top_k: int = 5

@app.post("/recommend")
def recommend(req: RecommendationRequest):
    if not model or not data:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please try again in a moment.")

    # This is a simplified recommendation logic.
    # It doesn't use the user_id yet, it just gives a global top-k recommendation.
    # A real implementation would use the user_id to personalize the scores.
    
    with torch.no_grad():
        # Get model output (scores for all nodes)
        all_scores = model(data)
        
    # We assume the second column contains the positive recommendation score
    recommendation_scores = all_scores[:, 1]
    
    # Find indices of nodes that are 'Items'
    item_indices = [
        i for i, node in enumerate(node_list) if 'Item' in node['labels']
    ]
    
    if not item_indices:
        return {"user_id": req.user_id, "recommendations": []}

    # Get scores only for the item nodes
    item_scores = recommendation_scores[item_indices]
    
    # Get the top_k items
    # Ensure k is not larger than the number of items
    k = min(req.top_k, len(item_scores))
    top_k_scores, top_k_relative_indices = torch.topk(item_scores, k=k)
    
    # Map relative indices back to original graph indices
    top_k_absolute_indices = [item_indices[i] for i in top_k_relative_indices]
    
    # Get the names of the recommended items
    recommended_items = [
        node_list[i]['name'] for i in top_k_absolute_indices
    ]
    
    return {"user_id": req.user_id, "recommendations": recommended_items}

@app.get("/health")
def health():
    return {"status": "ok"}
