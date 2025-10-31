import asyncio
import json
import uuid
import traceback
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette import EventSourceResponse
from pydantic import BaseModel

from src.utils import logger

# Import MCP server functionality
try:
    from mcp.server import Server
    from mcp.types import JSONRPCRequest, JSONRPCNotification, JSONRPCMessage
    from src.mcp_servers.knowledge_base_server import KnowledgeBaseServer
    from src.knowledge_base import knowledge_base
except ImportError as e:
    logger.error(f"Failed to import MCP server modules: {e}")
    KnowledgeBaseServer = None

mcp = APIRouter(prefix="/mcp", tags=["mcp", "Model Context Protocol"])

# MCP Session Management
class MCPSession(BaseModel):
    """MCP Session model"""
    session_id: str
    initialized: bool = False

# Global session storage and MCP server
_mcp_sessions: Dict[str, Dict[str, Any]] = {}
_mcp_server_instance = None

def get_mcp_server():
    """Get or create MCP server instance"""
    global _mcp_server_instance
    if _mcp_server_instance is None:
        if KnowledgeBaseServer is None:
            raise HTTPException(status_code=500, detail="MCP server not available")
        
        kb_server = KnowledgeBaseServer()
        _mcp_server_instance = kb_server.server
        logger.info("MCP server instance created")
    
    return _mcp_server_instance


def create_session() -> str:
    """Create a new MCP session"""
    session_id = str(uuid.uuid4())
    _mcp_sessions[session_id] = {
        "message_queue": asyncio.Queue(),
        "initialized": False
    }
    return session_id


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session by ID"""
    return _mcp_sessions.get(session_id)


async def handle_jsonrpc_message(session_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle incoming JSON-RPC message and return response if needed"""
    mcp_server = get_mcp_server()
    
    session = get_session(session_id)
    if not session:
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {
                "code": -32000,
                "message": "Invalid session"
            }
        }
    
    try:
        # Convert dict to proper JSON-RPC message object
        if "method" in message:
            if "id" in message:
                # Request
                jsonrpc_msg = JSONRPCRequest(
                    jsonrpc="2.0",
                    id=message["id"],
                    method=message["method"],
                    params=message.get("params")
                )
            else:
                # Notification
                jsonrpc_msg = JSONRPCNotification(
                    jsonrpc="2.0",
                    method=message["method"],
                    params=message.get("params")
                )
        else:
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {
                    "code": -32600,
                    "message": "Invalid Request"
                }
            }
        
        # Process message through MCP server
        # Note: This is a simplified implementation
        # In a real MCP server, you would use the server's message handling
        
        if jsonrpc_msg.method == "tools/list":
            # Handle tools list request
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {
                    "tools": [
                        {
                            "name": "query_knowledge_base",
                            "description": "Query knowledge bases",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query_text": {"type": "string"},
                                    "db_id": {"type": "string"},
                                    "mode": {"type": "string", "enum": ["local", "global", "hybrid", "naive", "mix"]},
                                    "top_k": {"type": "integer", "default": 10}
                                },
                                "required": ["query_text"]
                            }
                        }
                    ]
                }
            }
        elif jsonrpc_msg.method == "tools/call":
            # Handle tool call request
            params = jsonrpc_msg.params or {}
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name == "query_knowledge_base":
                # Use the KnowledgeBaseServer instance
                kb_server = KnowledgeBaseServer()
                result = await kb_server._query_knowledge_base(arguments)
                
                # Extract content
                content = ""
                if hasattr(result, 'content') and result.content:
                    for item in result.content:
                        if hasattr(item, 'text'):
                            content += item.text + "\n"
                        else:
                            content += str(item) + "\n"
                
                return {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "result": {
                        "content": [{"type": "text", "text": content.strip()}]
                    }
                }
        
        # Default response for unhandled methods
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {
                "code": -32601,
                "message": "Method not found"
            }
        }
        
    except Exception as e:
        logger.error(f"Error handling JSON-RPC message: {e}")
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }
    
    return None

# =============================================================================
# === MCP Standard Endpoints ===
# =============================================================================

@mcp.post("/session")
async def create_mcp_session():
    """Create a new MCP session (Standard MCP endpoint)"""
    session_id = create_session()
    return {"session_id": session_id}


@mcp.post("/message/{session_id}")
async def handle_mcp_message(session_id: str, request: Request):
    """Handle JSON-RPC message from client (Standard MCP endpoint)"""
    try:
        message = await request.json()
        
        # Validate JSON-RPC format
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            raise HTTPException(status_code=400, detail="Invalid JSON-RPC message")
        
        response = await handle_jsonrpc_message(session_id, message)
        
        if response:
            return response
        else:
            return {"status": "processed"}
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@mcp.get("/events/{session_id}")
async def stream_mcp_events(session_id: str):
    """Stream server-to-client events via SSE (Standard MCP endpoint)"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    async def event_generator():
        """Generate SSE events"""
        try:
            # Send initial connection event
            yield {
                "event": "connected",
                "data": json.dumps({"session_id": session_id, "status": "connected"})
            }
            
            # Stream messages from queue
            while True:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(
                        session["message_queue"].get(), 
                        timeout=30.0
                    )
                    
                    yield {
                        "event": "message",
                        "data": json.dumps(message)
                    }
                    
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {
                        "event": "keepalive",
                        "data": json.dumps({"timestamp": asyncio.get_event_loop().time()})
                    }
                    
        except Exception as e:
            logger.error(f"Error in event stream: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
    
    return EventSourceResponse(event_generator())


@mcp.get("/status")
async def get_mcp_status():
    """Get MCP server status (Standard MCP endpoint)"""
    try:
        mcp_server = get_mcp_server()
        return {
            "status": "running",
            "protocol": "MCP",
            "transport": "http_sse",
            "active_sessions": len(_mcp_sessions),
            "version": "1.0.0",
            "server_info": {
                "name": "Yuxi-Know Knowledge Base MCP Server",
                "version": "1.0.0"
            }
        }
    except Exception as e:
        logger.error(f"Error getting MCP status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "protocol": "MCP",
            "transport": "http_sse"
        }


@mcp.delete("/session/{session_id}")
async def close_mcp_session(session_id: str):
    """Close a MCP session (Standard MCP endpoint)"""
    if session_id in _mcp_sessions:
        del _mcp_sessions[session_id]
        return {"status": "closed", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


# =============================================================================
# === Legacy Compatibility Endpoints (Deprecated) ===
# =============================================================================

@mcp.get("/legacy/info")
async def get_legacy_info():
    """Information about legacy endpoints (for migration purposes)"""
    return {
        "message": "Legacy REST API endpoints have been deprecated",
        "migration_guide": {
            "old_endpoints": [
                "GET /mcp/tools",
                "POST /mcp/tools/call", 
                "POST /mcp/query",
                "GET /mcp/knowledge-bases"
            ],
            "new_approach": "Use standard MCP protocol with JSON-RPC messages",
            "steps": [
                "1. Create session: POST /mcp/session",
                "2. Send JSON-RPC messages: POST /mcp/message/{session_id}",
                "3. Listen to events: GET /mcp/events/{session_id}",
                "4. Close session: DELETE /mcp/session/{session_id}"
            ]
        },
        "mcp_protocol_info": {
            "specification": "https://modelcontextprotocol.io/",
            "transport": "HTTP + Server-Sent Events",
            "message_format": "JSON-RPC 2.0"
        }
    }