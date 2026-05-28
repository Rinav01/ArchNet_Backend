from typing import Any, Generator
from graphql import GraphQLError, DocumentNode, FieldNode, SelectionSetNode, OperationDefinitionNode
from strawberry.extensions import SchemaExtension

# Cost configuration weights
FIELD_COSTS = {
    # High cost expensive compilers / child sandboxes
    "generatepytorchcode": 15,
    "generatetensorflowcode": 15,
    "validateprojectcompilation": 15,
    "triggerasynccompilation": 15,
    
    # Medium cost relational database queries
    "project": 5,
    "projects": 5,
    "nodes": 3,
    "edges": 3,
    
    # Standard base scalars / trivial queries
    "me": 1,
    "id": 1,
    "name": 1,
    "email": 1
}

DEFAULT_FIELD_COST = 1

class GraphQLCostLimiter(SchemaExtension):
    """Strawberry GraphQL Extension that parses the query AST to sum up operational weights
    and blocks abusive complex queries before execution (max cost limit = 50).
    """
    def __init__(self, max_cost: int = 50):
        self.max_cost = max_cost

    def on_executing(self) -> Generator[None, None, None]:
        execution_context = self.execution_context
        document = execution_context.graphql_document
        
        if document:
            total_cost = self._calculate_cost(document)
            if total_cost > self.max_cost:
                raise GraphQLError(
                    f"GraphQL Query blocked: maximum query complexity exceeded. "
                    f"Parsed query cost is {total_cost}, but the allowed maximum is {self.max_cost}."
                )
        yield

    def _calculate_cost(self, node: Any) -> int:
        """Recursively parses standard graphql-core AST nodes to compute query cost."""
        if isinstance(node, DocumentNode):
            if not node.definitions:
                return 0
            return sum(self._calculate_cost(defn) for defn in node.definitions)
            
        elif isinstance(node, OperationDefinitionNode):
            if not node.selection_set:
                return 0
            return self._calculate_cost(node.selection_set)
            
        elif isinstance(node, SelectionSetNode):
            if not node.selections:
                return 0
            return sum(self._calculate_cost(sel) for sel in node.selections)
            
        elif isinstance(node, FieldNode):
            field_name = node.name.value.lower().strip()
            
            # Ignore standard introspection fields
            if field_name.startswith("__"):
                return 0
                
            cost = FIELD_COSTS.get(field_name, DEFAULT_FIELD_COST)
            if node.selection_set:
                cost += self._calculate_cost(node.selection_set)
            return cost
            
        return 0
