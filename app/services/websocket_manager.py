import json
import uuid
import asyncio
import redis.asyncio as aioredis
from fastapi import WebSocket
from typing import Dict, List, Any
from app.config.settings import settings

class WebSocketManager:
    def __init__(self):
        # Maps project_uuid -> list of connected websockets
        self.active_connections: Dict[uuid.UUID, List[WebSocket]] = {}
        # Maps project_uuid -> Redis Pub/Sub listener asyncio Task
        self.redis_tasks: Dict[uuid.UUID, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, project_id: uuid.UUID):
        await websocket.accept()
        if project_id not in self.active_connections:
            self.active_connections[project_id] = []
        self.active_connections[project_id].append(websocket)

        # If this is the first local connection in the project room, subscribe to Redis channel
        if len(self.active_connections[project_id]) == 1:
            task = asyncio.create_task(self._redis_subscription_listener(project_id))
            self.redis_tasks[project_id] = task

    async def disconnect(self, websocket: WebSocket, project_id: uuid.UUID):
        if project_id in self.active_connections:
            if websocket in self.active_connections[project_id]:
                self.active_connections[project_id].remove(websocket)
            
            # If room is empty, unsubscribe from Redis and clean up
            if not self.active_connections[project_id]:
                del self.active_connections[project_id]
                if project_id in self.redis_tasks:
                    task = self.redis_tasks[project_id]
                    task.cancel()
                    del self.redis_tasks[project_id]

    async def broadcast_to_project(self, project_id: uuid.UUID, event: Dict[str, Any]):
        """Directly broadcasts a python dictionary event message to all local room connections."""
        if project_id in self.active_connections:
            disconnected_sockets = []
            for websocket in self.active_connections[project_id]:
                try:
                    await websocket.send_json(event)
                except Exception:
                    disconnected_sockets.append(websocket)
            
            # Clean up any dead sockets detected during broadcast
            for dead_socket in disconnected_sockets:
                await self.disconnect(dead_socket, project_id)

    async def _redis_subscription_listener(self, project_id: uuid.UUID):
        """Asynchronous background loop listening to Redis Pub/Sub channel for the project room."""
        pubsub = None
        r = None
        try:
            r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
            pubsub = r.pubsub()
            channel = f"mlbuilder:project:{str(project_id)}"
            await pubsub.subscribe(channel)

            async for message in pubsub.listen():
                if message and message["type"] == "message":
                    data = message["data"]
                    try:
                        event = json.loads(data)
                        await self.broadcast_to_project(project_id, event)
                    except Exception as parse_err:
                        print(f"[WebSocketManager Error] Failed to parse PubSub message: {parse_err}")
        except asyncio.CancelledError:
            try:
                if pubsub:
                    await pubsub.unsubscribe(channel)
                if r:
                    await r.close()
            except Exception:
                pass
        except Exception as conn_err:
            print(f"[WebSocketManager Error] Redis subscription error: {conn_err}")
            await asyncio.sleep(5)
            if project_id in self.active_connections:
                task = asyncio.create_task(self._redis_subscription_listener(project_id))
                self.redis_tasks[project_id] = task

# Singleton WebSocket manager instance
ws_manager = WebSocketManager()
