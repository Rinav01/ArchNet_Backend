import uuid
from typing import Dict, List, Any

class CollaborationService:
    def __init__(self):
        # Maps project_id (uuid.UUID) -> client_id (str) -> user info dict
        self._rooms: Dict[uuid.UUID, Dict[str, Dict[str, Any]]] = {}
        # Simple harmonious pastel colors for collaborators
        self._colors = [
            "#FF6B6B", "#4DABF7", "#51CF66", "#FCC419", 
            "#FF922B", "#AE3EC9", "#94D82D", "#15AABF"
        ]

    def register_user_joined(
        self, 
        project_id: uuid.UUID, 
        user_id: uuid.UUID, 
        username: str, 
        client_id: str
    ) -> Dict[str, Any]:
        """Registers a newly connected user in the project collaboration session.
        Returns the initial room presence state.
        """
        if project_id not in self._rooms:
            self._rooms[project_id] = {}

        # Cycle color assignment based on current room count
        color_idx = len(self._rooms[project_id]) % len(self._colors)
        color = self._colors[color_idx]

        user_info = {
            "client_id": client_id,
            "user_id": str(user_id),
            "username": username,
            "color": color,
            "cursor": {"x": 0.0, "y": 0.0},
            "selection": None
        }

        self._rooms[project_id][client_id] = user_info
        return user_info

    def register_user_left(self, project_id: uuid.UUID, client_id: str) -> Dict[str, Any] | None:
        """Removes a disconnected user client from the collaboration session."""
        if project_id in self._rooms and client_id in self._rooms[project_id]:
            user_info = self._rooms[project_id].pop(client_id)
            if not self._rooms[project_id]:
                del self._rooms[project_id]
            return user_info
        return None

    def update_user_cursor(self, project_id: uuid.UUID, client_id: str, x: float, y: float) -> Dict[str, Any] | None:
        """Updates the cursor position for a connected client inside a project session."""
        if project_id in self._rooms and client_id in self._rooms[project_id]:
            self._rooms[project_id][client_id]["cursor"] = {"x": x, "y": y}
            return self._rooms[project_id][client_id]
        return None

    def update_user_selection(self, project_id: uuid.UUID, client_id: str, node_id: str | None) -> Dict[str, Any] | None:
        """Updates the active node layer selection for a client."""
        if project_id in self._rooms and client_id in self._rooms[project_id]:
            self._rooms[project_id][client_id]["selection"] = node_id
            return self._rooms[project_id][client_id]
        return None

    def get_room_state(self, project_id: uuid.UUID) -> List[Dict[str, Any]]:
        """Returns all active presence details of users connected to a project."""
        if project_id in self._rooms:
            return list(self._rooms[project_id].values())
        return []

# Instantiate global Collaboration Presence Service
collaboration_service = CollaborationService()
