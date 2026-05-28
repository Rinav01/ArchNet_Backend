from typing import Any, Generator, Union
from graphql import GraphQLError, DocumentNode, FieldNode, SelectionSetNode, OperationDefinitionNode
from strawberry.extensions import SchemaExtension

class GraphQLDepthLimiter(SchemaExtension):
    """Strawberry GraphQL Extension that analyzes query AST documents at execution time
    and rejects deep nested operations exceeding a threshold (max depth = 4) to prevent DoS attacks.
    """
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth

    def on_executing(self) -> Generator[None, None, None]:
        # Retrieve the parsed AST document
        execution_context = self.execution_context
        document = execution_context.graphql_document
        
        if document:
            depth = self._calculate_depth(document)
            if depth > self.max_depth:
                raise GraphQLError(
                    f"GraphQL Query blocked: maximum query depth exceeded. "
                    f"Parsed query depth is {depth}, but the allowed maximum is {self.max_depth}."
                )
        yield

    def _calculate_depth(self, node: Any) -> int:
        """Recursively parses standard graphql-core AST nodes to compute query nesting depth."""
        if isinstance(node, DocumentNode):
            if not node.definitions:
                return 0
            return max(self._calculate_depth(defn) for defn in node.definitions)
            
        elif isinstance(node, OperationDefinitionNode):
            if not node.selection_set:
                return 0
            return self._calculate_depth(node.selection_set)
            
        elif isinstance(node, SelectionSetNode):
            if not node.selections:
                return 0
            return max(self._calculate_depth(sel) for sel in node.selections)
            
        elif isinstance(node, FieldNode):
            # Ignore standard introspection fields (__schema, __type)
            if node.name.value.startswith("__"):
                return 0
                
            if not node.selection_set:
                return 1
            return 1 + self._calculate_depth(node.selection_set)
            
        return 0
