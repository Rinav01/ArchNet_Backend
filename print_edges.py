import uuid
from app.config.database import SessionLocal
from app.models.edge import Edge
from app.models.node import Node

db = SessionLocal()
project_id = uuid.UUID("58968d48-7b2f-4721-a936-543214d10e09")

nodes = {node.id: node for node in db.query(Node).filter(Node.project_id == project_id).all()}
edges = db.query(Edge).filter(Edge.project_id == project_id).all()

for edge in edges:
    from_node = nodes.get(edge.from_node_id)
    to_node = nodes.get(edge.to_node_id)
    print(f"Edge: {from_node.label} ({from_node.type}) -> {to_node.label} ({to_node.type})")
db.close()
