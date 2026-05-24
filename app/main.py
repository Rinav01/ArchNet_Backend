from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter

from app.config.settings import settings
from app.config.database import engine, Base
from app.auth.dependencies import get_graphql_context
from app.graphql.schema import schema

# Create database tables at application startup (MVP local convenience)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="MLBuilder Backend",
    description="Production-Ready MLBuilder Backend API",
    version="1.0.0"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production requirements
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Strawberry GraphQL FastAPI router with custom context dependency
graphql_router = GraphQLRouter(
    schema,
    context_getter=get_graphql_context
)

# Include the GraphQL endpoint
app.include_router(graphql_router, prefix="/graphql")

@app.get("/")
def read_root():
    """Service health check endpoint"""
    return {
        "status": "healthy",
        "service": "MLBuilder Backend API",
        "graphql_endpoint": "/graphql"
    }
