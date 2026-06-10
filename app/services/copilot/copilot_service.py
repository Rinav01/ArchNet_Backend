import uuid
import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.models.project import Project
from app.models.node import Node
from app.models.edge import Edge
from app.ir.ir_graph import IRGraph
from app.services.copilot.graph_context import GraphContextBuilder
from app.services.copilot.prompt_builder import CopilotPromptBuilder
from app.services.copilot.graph_agent import CopilotGraphAgent
from app.services.copilot.execution_engine import CopilotExecutionEngine

logger = logging.getLogger("mlbuilder.copilot.copilot_service")

class CopilotService:
    @staticmethod
    def generate_architecture(db: Session, project_id: uuid.UUID, prompt: str) -> IRGraph:
        """Generates a complete neural network graph using the LLM and replaces
        the project's architecture with it.
        """
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project with ID {project_id} not found.")

        # Build prompt
        system_p, user_p = CopilotPromptBuilder.build_generation_prompt(prompt, project.framework)

        # Call Agent
        response_text = CopilotGraphAgent.execute_agent(system_p, user_p, json_response=True)
        generated_graph = CopilotGraphAgent.parse_json_content(response_text)

        # Execute replacement in database
        return CopilotExecutionEngine.execute_graph_replacement(db, project_id, generated_graph)

    @staticmethod
    def modify_architecture(db: Session, project_id: uuid.UUID, prompt: str) -> IRGraph:
        """Modifies the existing project graph based on a natural language prompt."""
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project with ID {project_id} not found.")

        # Load existing nodes and edges to build IRGraph
        db_nodes = db.query(Node).filter(Node.project_id == project_id).all()
        db_edges = db.query(Edge).filter(Edge.project_id == project_id).all()
        ir_graph = IRGraph.from_db(project, db_nodes, db_edges)

        # Retrieve Context
        graph_context = GraphContextBuilder.get_graph_summary(ir_graph)

        # Build prompt
        system_p, user_p = CopilotPromptBuilder.build_modification_prompt(prompt, graph_context)

        # Call Agent
        response_text = CopilotGraphAgent.execute_agent(system_p, user_p, json_response=True)
        modified_graph = CopilotGraphAgent.parse_json_content(response_text)

        # Execute replacement in database
        return CopilotExecutionEngine.execute_graph_replacement(db, project_id, modified_graph)

    @staticmethod
    def explain_architecture(db: Session, project_id: uuid.UUID) -> str:
        """Generates a detailed Markdown explanation of the model architecture."""
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project with ID {project_id} not found.")

        # Load existing nodes and edges to build IRGraph
        db_nodes = db.query(Node).filter(Node.project_id == project_id).all()
        db_edges = db.query(Edge).filter(Edge.project_id == project_id).all()
        ir_graph = IRGraph.from_db(project, db_nodes, db_edges)

        # Retrieve Context
        graph_context = GraphContextBuilder.get_graph_summary(ir_graph)

        # Build prompt
        system_p, user_p = CopilotPromptBuilder.build_explanation_prompt(graph_context)

        # Call Agent (not JSON response, it returns text explanation)
        return CopilotGraphAgent.execute_agent(system_p, user_p, json_response=False)

    @staticmethod
    def refactor_architecture(db: Session, project_id: uuid.UUID) -> List[Dict[str, Any]]:
        """Suggests memory/latency/parameter optimizations and AI refactoring actions."""
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project with ID {project_id} not found.")

        # Load existing nodes and edges to build IRGraph
        db_nodes = db.query(Node).filter(Node.project_id == project_id).all()
        db_edges = db.query(Edge).filter(Edge.project_id == project_id).all()
        ir_graph = IRGraph.from_db(project, db_nodes, db_edges)

        # Retrieve Context
        graph_context = GraphContextBuilder.get_graph_summary(ir_graph)

        # Build prompt
        system_p, user_p = CopilotPromptBuilder.build_refactoring_prompt(graph_context)

        # Call Agent
        response_text = CopilotGraphAgent.execute_agent(system_p, user_p, json_response=True)
        return CopilotGraphAgent.parse_json_content(response_text)
