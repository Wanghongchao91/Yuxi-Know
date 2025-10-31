import asyncio
import json
import uuid
import traceback
from typing import Any, Dict, Optional, List, AsyncGenerator
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from src.utils.logger import logger

# Import MCP server functionality
try:
    from mcp.server import Server
    from mcp.types import JSONRPCRequest, JSONRPCNotification, JSONRPCMessage
    from src.mcp_servers.knowledge_base_server import KnowledgeBaseServer
    from src.knowledge_base import knowledge_base
    MCP_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import MCP server modules: {e}")
    KnowledgeBaseServer = None
    MCP_AVAILABLE = False

mcp = APIRouter(prefix="/mcp", tags=["mcp", "Model Context Protocol"])

# Global MCP server instance and active streams
_mcp_server_instance = None
_active_streams: Dict[str, Dict[str, Any]] = {}

class MCPMessage(BaseModel):
    """Standard MCP JSON-RPC Message"""
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

def get_mcp_server():
    """Get or create MCP server instance"""
    global _mcp_server_instance
    if _mcp_server_instance is None:
        if not MCP_AVAILABLE or KnowledgeBaseServer is None:
            raise HTTPException(status_code=500, detail="MCP server not available")
        
        try:
            _mcp_server_instance = KnowledgeBaseServer()
            logger.info("MCP server instance created successfully")
        except Exception as e:
            logger.error(f"Failed to create MCP server instance: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize MCP server: {str(e)}")
    
    return _mcp_server_instance

async def handle_mcp_message(message: MCPMessage) -> Optional[MCPMessage]:
    """Handle MCP JSON-RPC message according to standard protocol"""
    try:
        mcp_server = get_mcp_server()
        
        # Handle initialization
        if message.method == "initialize":
            params = message.params or {}
            client_info = params.get("clientInfo", {})
            protocol_version = params.get("protocolVersion", "2024-11-05")
            
            # Validate protocol version
            supported_versions = ["2024-11-05", "2025-06-18"]
            if protocol_version not in supported_versions:
                return MCPMessage(
                    id=message.id,
                    error={
                        "code": -32602,
                        "message": f"Unsupported protocol version: {protocol_version}. Supported: {supported_versions}"
                    }
                )
            
            return MCPMessage(
                id=message.id,
                result={
                    "protocolVersion": protocol_version,
                    "capabilities": {
                        "tools": {},
                        "logging": {},
                        "prompts": {},
                        "resources": {}
                    },
                    "serverInfo": {
                        "name": "Yuxi-Know Knowledge Base MCP Server",
                        "version": "1.0.0"
                    }
                }
            )
        
        # Handle tools/list
        elif message.method == "tools/list":
            tools = [
                {
                    "name": "query_knowledge_base",
                    "description": "Query all available knowledge bases with intelligent routing",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query_text": {
                                "type": "string", 
                                "description": "The query text to search for"
                            },
                            "db_id": {
                                "type": "string", 
                                "description": "Specific database ID to query (optional)"
                            },
                            "mode": {
                                "type": "string", 
                                "enum": ["local", "global", "hybrid", "naive", "mix"], 
                                "default": "mix",
                                "description": "Query mode to use"
                            },
                            "top_k": {
                                "type": "integer", 
                                "default": 10, 
                                "minimum": 1, 
                                "maximum": 100,
                                "description": "Number of results to return"
                            }
                        },
                        "required": ["query_text"]
                    }
                },
                {
                    "name": "list_knowledge_bases",
                    "description": "List all available knowledge bases and their information",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            ]
            
            return MCPMessage(
                id=message.id,
                result={"tools": tools}
            )
        
        # Handle tools/call
        elif message.method == "tools/call":
            params = message.params or {}
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if not tool_name:
                return MCPMessage(
                    id=message.id,
                    error={
                        "code": -32602,
                        "message": "Missing required parameter: name"
                    }
                )
            
            if tool_name == "query_knowledge_base":
                result = await mcp_server._query_knowledge_base(arguments)
            elif tool_name == "list_knowledge_bases":
                result = await mcp_server._list_knowledge_bases()
            else:
                return MCPMessage(
                    id=message.id,
                    error={
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                )
            
            # Extract content from result
            content = []
            if hasattr(result, 'content') and result.content:
                for item in result.content:
                    if hasattr(item, 'text'):
                        content.append({"type": "text", "text": item.text})
                    elif isinstance(item, dict):
                        content.append(item)
                    else:
                        content.append({"type": "text", "text": str(item)})
            
            return MCPMessage(
                id=message.id,
                result={"content": content}
            )
        
        # Handle ping (keepalive)
        elif message.method == "ping":
            return MCPMessage(
                id=message.id,
                result={}
            )
        
        # Handle notifications (no response needed)
        elif message.method in ["notifications/initialized", "notifications/cancelled"]:
            logger.info(f"Received notification: {message.method}")
            return None
        
        # Unknown method
        else:
            return MCPMessage(
                id=message.id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {message.method}"
                }
            )
    
    except Exception as e:
        logger.error(f"Error handling MCP message: {e}")
        logger.error(traceback.format_exc())
        return MCPMessage(
            id=message.id,
            error={
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        )

# =============================================================================
# === Standard MCP Streamable HTTP Endpoint ===
# =============================================================================

@mcp.api_route("", methods=["GET", "POST"])
async def mcp_endpoint(
    request: Request,
    accept: Optional[str] = Header(None),
    last_event_id: Optional[str] = Header(None, alias="Last-Event-ID")
):
    """
    Standard MCP Streamable HTTP endpoint
    
    - GET: Opens SSE stream for server-to-client communication
    - POST: Sends JSON-RPC message from client to server
    """
    try:
        if not MCP_AVAILABLE:
            raise HTTPException(status_code=500, detail="MCP server not available")
        
        if request.method == "GET":
            # Handle SSE stream request
            if not accept or "text/event-stream" not in accept:
                raise HTTPException(
                    status_code=406, 
                    detail="Accept header must include text/event-stream"
                )
            
            # Create stream ID for this connection
            stream_id = str(uuid.uuid4())
            
            async def event_generator() -> AsyncGenerator[Dict[str, str], None]:
                """Generate SSE events for MCP communication"""
                try:
                    # Store stream info
                    _active_streams[stream_id] = {
                        "created_at": datetime.now(),
                        "last_activity": datetime.now(),
                        "message_queue": asyncio.Queue(),
                        "connected": True
                    }
                    
                    logger.info(f"MCP SSE stream opened: {stream_id}")
                    
                    # Send connection event
                    yield {
                        "id": str(uuid.uuid4()),
                        "event": "connected",
                        "data": json.dumps({
                            "stream_id": stream_id,
                            "timestamp": datetime.now().isoformat(),
                            "message": "MCP SSE stream established"
                        })
                    }
                    
                    # Keep connection alive and handle messages
                    while _active_streams.get(stream_id, {}).get("connected", False):
                        try:
                            # Wait for messages with timeout
                            message = await asyncio.wait_for(
                                _active_streams[stream_id]["message_queue"].get(),
                                timeout=30.0
                            )
                            
                            yield {
                                "id": str(uuid.uuid4()),
                                "event": "message",
                                "data": json.dumps(message)
                            }
                            
                        except asyncio.TimeoutError:
                            # Send keepalive ping
                            yield {
                                "id": str(uuid.uuid4()),
                                "event": "ping",
                                "data": json.dumps({
                                    "timestamp": datetime.now().isoformat()
                                })
                            }
                            
                        except Exception as e:
                            logger.error(f"Error in SSE stream: {e}")
                            yield {
                                "id": str(uuid.uuid4()),
                                "event": "error",
                                "data": json.dumps({
                                    "error": str(e),
                                    "timestamp": datetime.now().isoformat()
                                })
                            }
                            break
                
                except Exception as e:
                    logger.error(f"Fatal error in SSE generator: {e}")
                    yield {
                        "id": str(uuid.uuid4()),
                        "event": "error",
                        "data": json.dumps({
                            "error": f"Fatal SSE error: {str(e)}",
                            "timestamp": datetime.now().isoformat()
                        })
                    }
                finally:
                    # Clean up stream
                    if stream_id in _active_streams:
                        _active_streams[stream_id]["connected"] = False
                        del _active_streams[stream_id]
                    logger.info(f"MCP SSE stream closed: {stream_id}")
            
            return EventSourceResponse(event_generator())
        
        elif request.method == "POST":
            # Handle JSON-RPC message
            try:
                message_data = await request.json()
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON")
            
            # Validate JSON-RPC format
            if not isinstance(message_data, dict) or message_data.get("jsonrpc") != "2.0":
                raise HTTPException(status_code=400, detail="Invalid JSON-RPC message format")
            
            # Create MCP message object
            try:
                mcp_message = MCPMessage(**message_data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid message format: {str(e)}")
            
            # Handle the message
            response = await handle_mcp_message(mcp_message)
            
            # Determine response type based on Accept header
            if accept and "text/event-stream" in accept:
                # Client wants SSE stream response
                async def response_generator() -> AsyncGenerator[Dict[str, str], None]:
                    if response:
                        yield {
                            "id": str(uuid.uuid4()),
                            "event": "response",
                            "data": json.dumps(response.dict(exclude_none=True))
                        }
                    
                    # Close stream after response
                    yield {
                        "id": str(uuid.uuid4()),
                        "event": "close",
                        "data": json.dumps({"message": "Response complete"})
                    }
                
                return EventSourceResponse(response_generator())
            
            else:
                # Return JSON response
                if response:
                    return JSONResponse(
                        content=response.dict(exclude_none=True),
                        status_code=200
                    )
                else:
                    # Notification - no response
                    return JSONResponse(
                        content={"status": "accepted"},
                        status_code=202
                    )
        
        else:
            raise HTTPException(status_code=405, detail="Method not allowed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in MCP endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# === MCP Server Information Endpoints ===
# =============================================================================

@mcp.get("/status")
async def get_mcp_status():
    """Get MCP server status and capabilities"""
    try:
        if not MCP_AVAILABLE:
            return {
                "status": "error",
                "error": "MCP dependencies not available",
                "protocol": "MCP",
                "transport": "streamable-http",
                "version": "2024-11-05"
            }
        
        mcp_server = get_mcp_server()
        return {
            "status": "running",
            "protocol": "MCP",
            "transport": "streamable-http",
            "version": "2024-11-05",
            "server_info": {
                "name": "Yuxi-Know Knowledge Base MCP Server",
                "version": "1.0.0"
            },
            "active_streams": len(_active_streams),
            "available": True,
            "capabilities": {
                "tools": ["query_knowledge_base", "list_knowledge_bases"],
                "logging": True,
                "prompts": False,
                "resources": False
            },
            "endpoint": "/api/mcp"
        }
    except Exception as e:
        logger.error(f"Error getting MCP status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "protocol": "MCP",
            "transport": "streamable-http",
            "available": False
        }

@mcp.get("/info")
async def get_mcp_info():
    """Get detailed MCP server information"""
    return {
        "name": "Yuxi-Know Knowledge Base MCP Server",
        "description": "Standard MCP server providing access to knowledge base tools",
        "protocol": "Model Context Protocol (MCP)",
        "version": "2024-11-05",
        "transport": "Streamable HTTP",
        "specification": "https://modelcontextprotocol.io/specification/",
        "usage": {
            "connection": "Single endpoint at /api/mcp supporting both GET (SSE) and POST (JSON-RPC)",
            "initialization": "Send initialize JSON-RPC request with protocolVersion and clientInfo",
            "tools": "Use tools/list to get available tools, tools/call to execute them"
        },
        "tools": [
            {
                "name": "query_knowledge_base",
                "description": "Query knowledge bases with intelligent routing",
                "parameters": ["query_text", "db_id", "mode", "top_k"]
            },
            {
                "name": "list_knowledge_bases", 
                "description": "List all available knowledge bases",
                "parameters": []
            }
        ],
        "compliance": {
            "standard": "MCP Specification 2024-11-05",
            "transport": "Streamable HTTP (replaces deprecated HTTP+SSE)",
            "json_rpc": "2.0",
            "features": ["tools", "logging", "streaming"]
        }
    }

# =============================================================================
# === Stream Management ===
# =============================================================================

@mcp.get("/streams")
async def list_active_streams():
    """List all active SSE streams (for debugging)"""
    streams = []
    for stream_id, stream_data in _active_streams.items():
        streams.append({
            "stream_id": stream_id,
            "created_at": stream_data["created_at"].isoformat(),
            "last_activity": stream_data["last_activity"].isoformat(),
            "connected": stream_data.get("connected", False)
        })
    
    return {
        "active_streams": len(streams),
        "streams": streams
    }

# Background task to clean up inactive streams
async def cleanup_inactive_streams():
    """Clean up streams that have been inactive for too long"""
    while True:
        try:
            now = datetime.now()
            inactive_streams = []
            
            for stream_id, stream_data in _active_streams.items():
                last_activity = stream_data.get("last_activity", now)
                if (now - last_activity).total_seconds() > 3600:  # 1 hour timeout
                    inactive_streams.append(stream_id)
            
            for stream_id in inactive_streams:
                if stream_id in _active_streams:
                    _active_streams[stream_id]["connected"] = False
                    del _active_streams[stream_id]
                    logger.info(f"Cleaned up inactive stream: {stream_id}")
            
            await asyncio.sleep(300)  # Check every 5 minutes
            
        except Exception as e:
            logger.error(f"Error in stream cleanup: {e}")
            await asyncio.sleep(60)

# Start cleanup task when module is imported
asyncio.create_task(cleanup_inactive_streams())