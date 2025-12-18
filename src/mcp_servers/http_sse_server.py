#!/usr/bin/env python3
"""
MCP HTTP+SSE Transport Server

This implements the MCP protocol using HTTP POST for client-to-server communication
and Server-Sent Events (SSE) for server-to-client streaming, following MCP specification.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, Optional
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette import EventSourceResponse
import uvicorn

from mcp.server import Server
from mcp.types import (
    JSONRPCMessage,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCNotification,
)

# Import knowledge base functionality
try:
    import os
    import sys
    
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.mcp_servers.knowledge_base_server import KnowledgeBaseServer
except ImportError as e:
    logging.error(f"Failed to import knowledge base modules: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPHTTPSSETransport:
    """MCP HTTP+SSE Transport Implementation"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.mcp_server = None
        self.window_seconds = 10
        self.max_requests = 50
        
    async def initialize_mcp_server(self):
        """Initialize the MCP server instance"""
        if self.mcp_server is None:
            kb_server = KnowledgeBaseServer()
            self.mcp_server = kb_server.server
            
    def create_session(self) -> str:
        """Create a new session and return session ID"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "message_queue": asyncio.Queue(),
            "initialized": False,
            "timestamps": deque()
        }
        return session_id
        
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        return self.sessions.get(session_id)
        
    async def handle_jsonrpc_message(self, session_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle incoming JSON-RPC message and return response if needed"""
        await self.initialize_mcp_server()
        
        session = self.get_session(session_id)
        if not session:
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {
                    "code": -32000,
                    "message": "Invalid session"
                }
            }
        now = asyncio.get_event_loop().time()
        timestamps = session.get("timestamps")
        while timestamps and now - timestamps[0] > self.window_seconds:
            timestamps.popleft()
        if len(timestamps) >= self.max_requests:
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {
                    "code": -32001,
                    "message": "Rate limit exceeded"
                }
            }
        timestamps.append(now)
        
        try:
            # Convert dict to JSONRPCMessage
            if "id" in message:
                if "method" in message:
                    # Request
                    jsonrpc_msg = JSONRPCRequest(
                        id=message["id"],
                        method=message["method"],
                        params=message.get("params")
                    )
                else:
                    # Response (shouldn't happen in this direction)
                    return None
            else:
                # Notification
                jsonrpc_msg = JSONRPCNotification(
                    method=message["method"],
                    params=message.get("params")
                )
            
            # Process message through MCP server
            response = await self.mcp_server.handle_message(jsonrpc_msg)
            
            if response:
                # Convert response back to dict
                if hasattr(response, 'model_dump'):
                    resp_dict = response.model_dump()
                else:
                    resp_dict = response.__dict__
                await session["message_queue"].put({"session_id": session_id, "response": resp_dict})
                return resp_dict
                    
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
        
        return None


# Global transport instance
transport = MCPHTTPSSETransport()

# FastAPI app
app = FastAPI(title="MCP HTTP+SSE Server", version="1.0.0")


@app.post("/mcp/session")
async def create_session():
    """Create a new MCP session"""
    session_id = transport.create_session()
    return {"session_id": session_id}


@app.post("/mcp/message/{session_id}")
async def handle_message(session_id: str, request: Request):
    """Handle JSON-RPC message from client"""
    try:
        message = await request.json()
        
        # Validate JSON-RPC format
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            raise HTTPException(status_code=400, detail="Invalid JSON-RPC message")
        
        response = await transport.handle_jsonrpc_message(session_id, message)
        
        if response:
            return response
        else:
            return {"status": "processed"}
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/events/{session_id}")
async def stream_events(session_id: str):
    """Stream server-to-client events via SSE"""
    session = transport.get_session(session_id)
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


@app.get("/mcp/status")
async def get_status():
    """Get server status"""
    return {
        "status": "running",
        "protocol": "MCP HTTP+SSE",
        "transport": "http_sse",
        "active_sessions": len(transport.sessions),
        "version": "1.0.0"
    }


@app.delete("/mcp/session/{session_id}")
async def close_session(session_id: str):
    """Close a session"""
    if session_id in transport.sessions:
        del transport.sessions[session_id]
        return {"status": "closed"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


async def main():
    """Main entry point for HTTP+SSE server"""
    logger.info("Starting MCP HTTP+SSE Server...")
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        timeout_keep_alive=30,
        limit_concurrency=200
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
