# Graph based Recommender System

This project demonstrates a recommender system using Graph Neural Networks, FastAPI and Neo4j.

## Features
- GraphDB integration for knowledge graph retrieval
- FastAPI endpoints for workflow orchestration

## Setup
1. Create a `.env` file with your required API keys:
   ```
   NEO4J_URI=
   NEO4J_USER=
   NEO4J_PASSWORD=
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Getting Started

### 1. Install dependencies
```sh
pip install -r requirements.txt
```

### 2. Run Neo4j locally
- Download Neo4j Community Edition from https://neo4j.com/download/
- Start the Neo4j server (default bolt://localhost:7687)

### 3. Build the Knowledge Graph
```sh
python graph_builder.py
```

### 4. Train the GNN Recommender
```sh
python gnn_recommender.py
```

### 5. Start the API server
```sh
uvicorn api:app --reload
```


## Notes
- For production, secure your Neo4j instance and tune the GNN model as needed.

---
