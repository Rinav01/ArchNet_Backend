import json
import uuid
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from app.auth.security import decode_access_token
from app.services.websocket_manager import ws_manager
from app.services.collaboration_service import collaboration_service
from app.services.crdt_resolver import CRDTOperationResolver
from app.services.event_dispatcher import EventDispatcher
from app.config.database import SessionLocal
from app.models.user import User

logger = logging.getLogger("mlbuilder.ws_router")
ws_router = APIRouter()

@ws_router.websocket("/ws/projects/{project_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    project_id: str,
    token: str | None = Query(None)
):
    try:
        proj_uuid = uuid.UUID(project_id)
    except ValueError:
        await websocket.close(code=4000, reason="Invalid project ID format.")
        return

    # Verify authorization token
    if not token:
        await websocket.close(code=4001, reason="Missing token query parameter.")
        return

    payload = decode_access_token(token)
    if not payload:
        await websocket.close(code=4002, reason="Invalid or expired authorization token.")
        return

    # Load authenticated user details from database
    db = SessionLocal()
    try:
        user_id_str = payload.get("sub")
        if not user_id_str:
            await websocket.close(code=4003, reason="Subject identifier is missing.")
            return
        
        user_uuid = uuid.UUID(user_id_str)
        user = db.query(User).filter(User.id == user_uuid).first()
        if not user:
            await websocket.close(code=4004, reason="User record not found.")
            return
            
        # Verify project access/existence
        from app.models.project import Project
        project = db.query(Project).filter(Project.id == proj_uuid).first()
        if not project:
            await websocket.close(code=4006, reason="Project not found.")
            return
            
        # Check permissions: Admins bypass, others must own or it must be public
        if user.role != "admin" and project.user_id != user.id and not project.is_public:
            await websocket.close(code=4007, reason="Forbidden: You do not have access to this project.")
            return
            
        username = user.username
        user_id = user.id
        user_role = user.role
    except Exception as db_err:
        await websocket.close(code=4005, reason=f"Database authentication error: {str(db_err)}")
        return
    finally:
        db.close()

    # Generate a unique connection Client ID for this tab/client
    client_id = f"client_{uuid.uuid4().hex[:8]}"

    # Register room connection in the WebSocket Manager
    await ws_manager.connect(websocket, proj_uuid)
    
    # 1. Register presence & publish UserJoined event over Redis Pub/Sub
    user_info = collaboration_service.register_user_joined(proj_uuid, user_id, username, client_id)
    room_presence = collaboration_service.get_room_state(proj_uuid)

    try:
        # Acknowledge connection and initialize session details for newly connected socket
        await websocket.send_json({
            "type": "SessionInit",
            "client_id": client_id,
            "presence": room_presence
        })

        # Broadcast presence join to other connected developers
        EventDispatcher.get_redis().publish(
            f"mlbuilder:project:{str(proj_uuid)}",
            json.dumps({
                "type": "UserJoined",
                "user": user_info
            })
        )

        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                msg_type = msg.get("type")

                # Handle ping/pong heartbeat
                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                # 2. Handle collaboration presence cursor coordinates
                elif msg_type == "presence_cursor":
                    x = float(msg.get("x", 0.0))
                    y = float(msg.get("y", 0.0))
                    collaboration_service.update_user_cursor(proj_uuid, client_id, x, y)
                    
                    # Direct multi-node broadcast over Redis Pub/Sub
                    EventDispatcher.get_redis().publish(
                        f"mlbuilder:project:{str(proj_uuid)}",
                        json.dumps({
                            "type": "UserCursor",
                            "client_id": client_id,
                            "x": x,
                            "y": y
                        })
                    )

                # 3. Handle active visual node selections
                elif msg_type == "presence_selection":
                    node_id = msg.get("node_id")
                    collaboration_service.update_user_selection(proj_uuid, client_id, node_id)
                    
                    EventDispatcher.get_redis().publish(
                        f"mlbuilder:project:{str(proj_uuid)}",
                        json.dumps({
                            "type": "UserSelection",
                            "client_id": client_id,
                            "node_id": node_id
                        })
                    )

                # 4. Handle collaborative CRDT editing operations
                elif msg_type == "operation":
                    op = msg.get("op", {})
                    
                    # Block viewers from performing operations
                    if user_role == "viewer":
                        await websocket.send_json({
                            "type": "OperationRejected",
                            "op": op,
                            "reason": "Forbidden: Viewers cannot perform collaborative editing operations."
                        })
                        continue
                    
                    # Resolve operational edits within a transaction
                    db_session = SessionLocal()
                    try:
                        res = CRDTOperationResolver.apply_operation(db_session, proj_uuid, op)
                        if res.get("success"):
                            # Distribute synchronized action over Redis Pub/Sub
                            EventDispatcher.get_redis().publish(
                                f"mlbuilder:project:{str(proj_uuid)}",
                                json.dumps({
                                    "type": "OperationApplied",
                                    "client_id": client_id,
                                    "op": op,
                                    "result": res
                                })
                            )
                        else:
                            # Send rejection alert directly to the initiating developer socket
                            await websocket.send_json({
                                "type": "OperationRejected",
                                "op": op,
                                "reason": res.get("reason", "Conflict rejection.")
                            })
                    except Exception as op_err:
                        logger.error(f"Failed to process collaborative operation: {op_err}", exc_info=True)
                        await websocket.send_json({
                            "type": "OperationRejected",
                            "op": op,
                            "reason": f"Internal process error: {str(op_err)}"
                        })
                    finally:
                        db_session.close()

            except Exception as parse_err:
                logger.warning(f"Failed to process websocket text: {parse_err}")
                
    except WebSocketDisconnect:
        pass
    finally:
        # Deregister socket from active connection map
        await ws_manager.disconnect(websocket, proj_uuid)
        
        # 5. Clean up presence tracking & publish UserLeft event
        left_user = collaboration_service.register_user_left(proj_uuid, client_id)
        if left_user:
            EventDispatcher.get_redis().publish(
                f"mlbuilder:project:{str(proj_uuid)}",
                json.dumps({
                    "type": "UserLeft",
                    "client_id": client_id,
                    "user_id": left_user["user_id"]
                })
            )
