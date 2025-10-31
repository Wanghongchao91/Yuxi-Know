import asyncio
import json
import uuid
import traceback
from typing import Any, Dict, Optional, List, AsyncGenerator
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field

from src.utils.logging_config import logger

# Import MCP server functionality
try:
    from mcp.server import Server
    from mcp.types import JSONRPCRequest, JSONRPCNotification, JSONRPCMessage
    from src.mcp_servers.knowledge_base_server import KnowledgeBaseServer
    from src.knowledge import knowledge_base

    MCP_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import MCP server modules: {e}")
    KnowledgeBaseServer = None
    MCP_AVAILABLE = False

mcp = APIRouter(prefix="/mcp", tags=["mcp", "Model Context Protocol"])

# Global MCP server instance
_mcp_server_instance = None

class MCPMessage(BaseModel):
    """Standard MCP JSON-RPC Message"""
    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: Optional[str] = Field(default=None, description="Request ID")
    method: Optional[str] = Field(default=None, description="Method name")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Method parameters")
    result: Optional[Any] = Field(default=None, description="Result for successful response")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error information")

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

async def handle_mcp_request(message: MCPMessage) -> Optional[MCPMessage]:
    """Handle MCP JSON-RPC request according to standard protocol"""
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
                        "tools": {
                            "listChanged": False
                        },
                        "logging": {},
                        "prompts": {
                            "listChanged": False
                        },
                        "resources": {
                            "listChanged": False,
                            "subscribe": False,
                            "listChanged": False
                        }
                    },
                    "serverInfo": {
                        "name": "Yuxi-Know Knowledge Base MCP Server",
                        "version": "1.0.0"
                    }
                }
            )
        
        # Handle tools/list
        elif message.method == "tools-list":
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
            
            try:
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
                
                # Format result according to MCP standard
                content = []
                if hasattr(result, 'content') and result.content:
                    for item in result.content:
                        if hasattr(item, 'text'):
                            content.append({
                                "type": "text",
                                "text": item.text
                            })
                        elif isinstance(item, dict):
                            content.append(item)
                        else:
                            content.append({
                                "type": "text", 
                                "text": str(item)
                            })
                else:
                    # If no content, add a default text result
                    content.append({
                        "type": "text",
                        "text": str(result) if result else "Tool executed successfully"
                    })
                
                return MCPMessage(
                    id=message.id,
                    result={
                        "content": content,
                        "isError": False
                    }
                )
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                return MCPMessage(
                    id=message.id,
                    error={
                        "code": -32603,
                        "message": f"Tool execution failed: {str(e)}"
                    }
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
        logger.error(f"Error handling MCP request: {e}")
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
async def mcp_streamable_http_endpoint(
    request: Request,
    accept: Optional[str] = Header(None)
):
    """
    Standard MCP Streamable HTTP endpoint - Updated for 2025-03-26 specification
    
    This endpoint implements the official MCP Streamable HTTP transport:
    - GET: Returns endpoint information or upgrades to SSE for streaming
    - POST: Handles client-to-server JSON-RPC messages
    
    Reference: https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/
    """
    try:
        if not MCP_AVAILABLE:
            return JSONResponse(
                status_code=503,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": "MCP server not available"
                    }
                }
            )
        
        if request.method == "GET":
            # Check if client wants to upgrade to SSE for streaming
            accept_header = request.headers.get("accept", "")
            
            if "text/event-stream" in accept_header:
                # Client wants SSE streaming - establish streamable connection
                logger.info("MCP Streamable HTTP GET request - upgrading to SSE stream")
                
                # Generate unique session ID for this stream
                session_id = str(uuid.uuid4())
                
                async def sse_event_generator() -> AsyncGenerator[Dict[str, str], None]:
                    """Generate SSE events for MCP server-to-client communication"""
                    try:
                        logger.info(f"MCP SSE stream established: {session_id}")
                        
                        # Send initial connection event
                        yield {
                            "event": "connected",
                            "data": json.dumps({
                                "sessionId": session_id,
                                "timestamp": datetime.now().isoformat(),
                                "protocol": "MCP",
                                "version": "2024-11-05"
                            })
                        }
                        
                        # Keep stream alive and handle server-initiated messages
                        message_count = 0
                        while True:
                            try:
                                # Wait for any server-initiated events or keepalive
                                await asyncio.sleep(30.0)  # 30-second keepalive interval
                                
                                # Send periodic ping to keep connection alive
                                message_count += 1
                                yield {
                                    "event": "ping",
                                    "data": json.dumps({
                                        "timestamp": datetime.now().isoformat(),
                                        "messageCount": message_count
                                    })
                                }
                                
                            except asyncio.CancelledError:
                                logger.info(f"MCP SSE stream cancelled: {session_id}")
                                break
                            except Exception as e:
                                logger.error(f"Error in MCP SSE stream: {e}")
                                yield {
                                    "event": "error", 
                                    "data": json.dumps({
                                        "error": str(e),
                                        "timestamp": datetime.now().isoformat()
                                    })
                                }
                                break
                                
                    except Exception as e:
                        logger.error(f"Fatal error in MCP SSE generator: {e}")
                        yield {
                            "event": "error",
                            "data": json.dumps({
                                "error": f"Fatal SSE error: {str(e)}",
                                "timestamp": datetime.now().isoformat()
                            })
                        }
                    finally:
                        logger.info(f"MCP SSE stream closed: {session_id}")
                
                # Return SSE response with session ID header
                response = EventSourceResponse(sse_event_generator())
                response.headers["Mcp-Session-Id"] = session_id
                response.headers["Cache-Control"] = "no-cache"
                return response
                
            else:
                # Client wants basic endpoint info (non-streaming)
                logger.info("MCP Streamable HTTP GET request - returning endpoint info")
                return JSONResponse(
                    status_code=200,
                    content={
                        "protocol": "Model Context Protocol",
                        "version": "2024-11-05",
                        "transport": "Streamable HTTP",
                        "endpoints": {
                            "mcp": "/api/mcp",
                            "status": "/api/mcp/status"
                        },
                        "capabilities": {
                            "streaming": True,
                            "sse_upgrade": True
                        }
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Cache-Control": "no-cache"
                    }
                )
        
        elif request.method == "POST":
            # Handle JSON-RPC message from client to server
            try:
                message_data = await request.json()
            except json.JSONDecodeError:
                return JSONResponse(
                    status_code=400,
                    content={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": "Parse error: Invalid JSON"
                        }
                    }
                )
            
            # Validate basic JSON-RPC structure
            if not isinstance(message_data, dict):
                return JSONResponse(
                    status_code=400,
                    content={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request: Message must be a JSON object"
                        }
                    }
                )
            
            # Validate JSON-RPC version
            if message_data.get("jsonrpc") != "2.0":
                return JSONResponse(
                    status_code=400,
                    content={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request: jsonrpc field must be '2.0'"
                        }
                    }
                )
            
            # Create MCP message object
            try:
                mcp_message = MCPMessage(**message_data)
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32600,
                            "message": f"Invalid Request: {str(e)}"
                        }
                    }
                )
            
            # Process the MCP request
            logger.info(f"Processing MCP request: {mcp_message.method}")
            response = await handle_mcp_request(mcp_message)
            
            # Return appropriate response
            if response is None:
                # This was a notification - return 202 Accepted with no content
                return JSONResponse(
                    status_code=202,
                    content={"status": "accepted"},
                    headers={
                        "Content-Type": "application/json",
                        "Cache-Control": "no-cache"
                    }
                )
            else:
                # Return the JSON-RPC response
                json_response = JSONResponse(
                    status_code=200,
                    content=response.dict(exclude_none=True),
                    headers={
                        "Content-Type": "application/json",
                        "Cache-Control": "no-cache"
                    }
                )
                
                # Add session ID if this is an initialize response
                if mcp_message.method == "initialize" and hasattr(response, 'result'):
                    # Generate session ID for new initialization
                    session_id = str(uuid.uuid4())
                    json_response.headers["Mcp-Session-Id"] = session_id
                    logger.info(f"MCP session initialized: {session_id}")
                
                return json_response
        
        else:
            # This should never happen due to FastAPI routing, but handle defensively
            return JSONResponse(
                status_code=405,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": "Method not allowed"
                    }
                }
            )
            
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in MCP Streamable HTTP endpoint: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
        )

# =============================================================================
# === MCP Server Information Endpoints (for debugging/discovery) ===
# =============================================================================

@mcp.get("/status")
async def get_mcp_status():
    """Get MCP server status and capabilities (non-standard but useful for debugging)"""
    try:
        if not MCP_AVAILABLE:
            return {
                "status": "error",
                "error": "MCP dependencies not available",
                "protocol": "MCP",
                "transport": "streamable-http",
                "version": "2024-11-05",
                "available": False
            }
        
        # Test server initialization
        mcp_server = get_mcp_server()
        
        return {
            "status": "running",
            "protocol": "Model Context Protocol (MCP)",
            "version": "2024-11-05",
            "transport": "Streamable HTTP",
            "serverInfo": {
                "name": "Yuxi-Know Knowledge Base MCP Server",
                "version": "1.0.0"
            },
            "available": True,
            "capabilities": {
                "tools": ["query_knowledge_base", "list_knowledge_bases"],
                "logging": True,
                "prompts": False,
                "resources": False
            },
            "endpoints": {
                "main": "/api/mcp",
                "status": "/api/mcp/status",
                "info": "/api/mcp/info"
            },
            "compliance": {
                "standard": "MCP Specification 2024-11-05",
                "transport": "Streamable HTTP",
                "json_rpc": "2.0"
            }
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
    """Get detailed MCP server information (non-standard but useful for documentation)"""
    return {
        "name": "Yuxi-Know Knowledge Base MCP Server",
        "description": "Standard MCP server providing access to knowledge base tools",
        "protocol": "Model Context Protocol (MCP)",
        "version": "2024-11-05",
        "transport": "Streamable HTTP",
        "specification": "https://modelcontextprotocol.io/specification/2024-11-05/",
        "implementation": {
            "language": "Python",
            "framework": "FastAPI",
            "server": "Yuxi-Know Knowledge Base"
        },
        "usage": {
            "connection": "Single endpoint at /api/mcp with GET (SSE) and POST (JSON-RPC)",
            "initialization": "Send initialize JSON-RPC request",
            "tools": {
                "list": "Use tools/list method",
                "call": "Use tools/call method with name and arguments"
            }
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
            "transport": "Streamable HTTP (official)",
            "json_rpc": "2.0 (strict)",
            "features": ["tools", "logging", "streaming"]
        }
    }