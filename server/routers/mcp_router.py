import asyncio
import json
import uuid
import traceback
from typing import Any, Dict, Optional, List, AsyncGenerator, Union, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field, validator

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

# Configuration constants
class MCPConfig:
    """MCP server configuration constants"""
    PROTOCOL_VERSION = "2.0"
    SUPPORTED_PROTOCOL_VERSIONS = ["2024-11-05", "2025-06-18"]
    SERVER_NAME = "Yuxi-Know Knowledge Base MCP Server"
    SERVER_VERSION = "1.0.0"
    SSE_KEEPALIVE_INTERVAL = 30.0  # seconds
    MAX_TOP_K = 100
    DEFAULT_TOP_K = 10

class ErrorCode(Enum):
    """Standard JSON-RPC error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

class QueryMode(Enum):
    """Available query modes"""
    LOCAL = "local"
    GLOBAL = "global" 
    HYBRID = "hybrid"
    NAIVE = "naive"
    MIX = "mix"

mcp = APIRouter(prefix="/mcp", tags=["mcp", "Model Context Protocol"])

# Global MCP server instance
_mcp_server_instance = None

@dataclass
class MCPTool:
    """MCP tool definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]

class MCPMessage(BaseModel):
    """Standard MCP JSON-RPC Message"""
    jsonrpc: str = Field(default=MCPConfig.PROTOCOL_VERSION, description="JSON-RPC version")
    id: Optional[Union[str, int, float]] = Field(default=None, description="Request ID")
    method: Optional[str] = Field(default=None, description="Method name")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Method parameters")
    result: Optional[Any] = Field(default=None, description="Result for successful response")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error information")
    
    @validator('jsonrpc')
    def validate_jsonrpc(cls, v):
        if v != MCPConfig.PROTOCOL_VERSION:
            raise ValueError(f"jsonrpc must be {MCPConfig.PROTOCOL_VERSION}")
        return v

class MCPErrorResponse(BaseModel):
    """Standardized error response"""
    code: int
    message: str
    data: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "data": self.data}

# Predefined MCP tools
MCP_TOOLS: List[MCPTool] = [
    MCPTool(
        name="query_knowledge_base",
        description="Query all available knowledge bases with intelligent routing. IMPORTANT: Call list_knowledge_bases first to get available db_id values",
        input_schema={
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string", 
                    "description": "The query text to search for"
                },
                "db_id": {
                    "type": "string", 
                    "description": "Specific database ID to query (optional). Use list_knowledge_bases to get available db_id values"
                },
                "mode": {
                    "type": "string", 
                    "enum": [mode.value for mode in QueryMode], 
                    "default": QueryMode.MIX.value,
                    "description": "Query mode to use"
                },
                "top_k": {
                    "type": "integer", 
                    "default": MCPConfig.DEFAULT_TOP_K, 
                    "minimum": 1, 
                    "maximum": MCPConfig.MAX_TOP_K,
                    "description": "Number of results to return"
                }
            },
            "required": ["query_text"]
        }
    ),
    MCPTool(
        name="list_knowledge_bases",
        description="List all available knowledge bases and their information",
        input_schema={
            "type": "object",
            "properties": {},
            "required": []
        }
    )
]

def create_error_response(code: ErrorCode, message: str, data: Any = None) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "code": code.value,
        "message": message,
        "data": data
    }

def create_jsonrpc_error_response(error_code: ErrorCode, message: str, request_id: Optional[Any] = None) -> MCPMessage:
    """Create JSON-RPC error response"""
    return MCPMessage(
        id=request_id,
        error=create_error_response(error_code, message)
    )

def get_mcp_server() -> KnowledgeBaseServer:
    """Get or create MCP server instance with proper error handling"""
    global _mcp_server_instance
    
    if _mcp_server_instance is not None:
        return _mcp_server_instance
    
    if not MCP_AVAILABLE or KnowledgeBaseServer is None:
        logger.error("MCP server not available - dependencies missing")
        raise HTTPException(
            status_code=503, 
            detail="MCP server dependencies not available"
        )
    
    try:
        _mcp_server_instance = KnowledgeBaseServer()
        logger.info("MCP server instance created successfully")
        return _mcp_server_instance
    except Exception as e:
        logger.error(f"Failed to create MCP server instance: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to initialize MCP server: {str(e)}"
        )

class MCPRequestHandler:
    """Handles MCP JSON-RPC requests with modular approach"""
    
    def __init__(self, mcp_server: KnowledgeBaseServer):
        self.mcp_server = mcp_server
        self.handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "ping": self._handle_ping,
            "notifications/initialized": self._handle_notification,
            "notifications/cancelled": self._handle_notification,
        }
    
    async def handle_request(self, message: MCPMessage) -> Optional[MCPMessage]:
        """Route request to appropriate handler"""
        if not message.method:
            return create_jsonrpc_error_response(
                ErrorCode.INVALID_REQUEST, 
                "Missing method field", 
                message.id
            )
        
        handler = self.handlers.get(message.method)
        if not handler:
            return create_jsonrpc_error_response(
                ErrorCode.METHOD_NOT_FOUND,
                f"Method not found: {message.method}",
                message.id
            )
        
        try:
            return await handler(message)
        except Exception as e:
            logger.error(f"Error in handler {message.method}: {e}", exc_info=True)
            return create_jsonrpc_error_response(
                ErrorCode.INTERNAL_ERROR,
                f"Handler error: {str(e)}",
                message.id
            )
    
    async def _handle_initialize(self, message: MCPMessage) -> MCPMessage:
        """Handle initialize request"""
        params = message.params or {}
        protocol_version = params.get("protocolVersion", "2024-11-05")
        
        if protocol_version not in MCPConfig.SUPPORTED_PROTOCOL_VERSIONS:
            return create_jsonrpc_error_response(
                ErrorCode.INVALID_PARAMS,
                f"Unsupported protocol version: {protocol_version}. Supported: {MCPConfig.SUPPORTED_PROTOCOL_VERSIONS}",
                message.id
            )
        
        return MCPMessage(
            id=message.id,
            result={
                "protocolVersion": protocol_version,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "logging": {},
                    "prompts": {"listChanged": False},
                    "resources": {
                        "listChanged": False,
                        "subscribe": False,
                        "listChanged": False
                    }
                },
                "serverInfo": {
                    "name": MCPConfig.SERVER_NAME,
                    "version": MCPConfig.SERVER_VERSION
                }
            }
        )
    
    async def _handle_tools_list(self, message: MCPMessage) -> MCPMessage:
        """Handle tools/list request"""
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema
            }
            for tool in MCP_TOOLS
        ]
        
        return MCPMessage(
            id=message.id,
            result={"tools": tools}
        )
    
    async def _handle_tools_call(self, message: MCPMessage) -> MCPMessage:
        """Handle tools/call request"""
        params = message.params or {}
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if not tool_name:
            return create_jsonrpc_error_response(
                ErrorCode.INVALID_PARAMS,
                "Missing required parameter: name",
                message.id
            )
        
        # Validate tool exists
        tool = next((t for t in MCP_TOOLS if t.name == tool_name), None)
        if not tool:
            return create_jsonrpc_error_response(
                ErrorCode.METHOD_NOT_FOUND,
                f"Unknown tool: {tool_name}",
                message.id
            )
        
        try:
            if tool_name == "query_knowledge_base":
                # Validate db_id if provided
                db_id = arguments.get("db_id")
                if db_id:
                    # First check if knowledge bases are available
                    available_kbs = await self.mcp_server._list_knowledge_bases()
                    if hasattr(available_kbs, 'content') and available_kbs.content:
                        kb_list = json.loads(available_kbs.content[0].text) if available_kbs.content[0].text else []
                        if isinstance(kb_list, list) and not any(kb.get('id') == db_id for kb in kb_list):
                            return create_jsonrpc_error_response(
                                ErrorCode.INVALID_PARAMS,
                                f"Database ID '{db_id}' not found. Use list_knowledge_bases to get available database IDs.",
                                message.id
                            )
                
                result = await self.mcp_server._query_knowledge_base(arguments)
            elif tool_name == "list_knowledge_bases":
                result = await self.mcp_server._list_knowledge_bases()
            else:
                return create_jsonrpc_error_response(
                    ErrorCode.METHOD_NOT_FOUND,
                    f"Tool not implemented: {tool_name}",
                    message.id
                )
            
            return self._format_tool_result(result, message.id)
            
        except Exception as e:
            logger.error(f"Tool execution failed {tool_name}: {e}", exc_info=True)
            return create_jsonrpc_error_response(
                ErrorCode.INTERNAL_ERROR,
                f"Tool execution failed: {str(e)}",
                message.id
            )
    
    def _format_tool_result(self, result: Any, request_id: Optional[Any]) -> MCPMessage:
        """Format tool execution result according to MCP standard"""
        content = []
        
        if hasattr(result, 'content') and result.content:
            for item in result.content:
                if hasattr(item, 'text'):
                    content.append({"type": "text", "text": item.text})
                elif isinstance(item, dict):
                    content.append(item)
                else:
                    content.append({"type": "text", "text": str(item)})
        else:
            content.append({
                "type": "text",
                "text": str(result) if result else "Tool executed successfully"
            })
        
        return MCPMessage(
            id=request_id,
            result={
                "content": content,
                "isError": False
            }
        )
    
    async def _handle_ping(self, message: MCPMessage) -> MCPMessage:
        """Handle ping request"""
        return MCPMessage(id=message.id, result={})
    
    async def _handle_notification(self, message: MCPMessage) -> None:
        """Handle notification (no response)"""
        logger.info(f"Received notification: {message.method}")
        return None

async def handle_mcp_request(message: MCPMessage) -> Optional[MCPMessage]:
    """Handle MCP JSON-RPC request according to standard protocol"""
    try:
        mcp_server = get_mcp_server()
        handler = MCPRequestHandler(mcp_server)
        return await handler.handle_request(message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error handling MCP request: {e}", exc_info=True)
        return create_jsonrpc_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Unexpected error: {str(e)}",
            message.id if hasattr(message, 'id') else None
        )

class SSEEventGenerator:
    """Optimized SSE event generator for MCP streaming"""
    
    def __init__(self, session_id: str, keepalive_interval: float = MCPConfig.SSE_KEEPALIVE_INTERVAL):
        self.session_id = session_id
        self.keepalive_interval = keepalive_interval
        self.message_count = 0
        self._running = True
    
    async def generate_events(self) -> AsyncGenerator[Dict[str, str], None]:
        """Generate SSE events with optimized error handling"""
        try:
            logger.info(f"MCP SSE stream established: {self.session_id}")
            
            # Send initial connection event
            yield self._create_event("connected", {
                "sessionId": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "protocol": "MCP",
                "version": "2024-11-05"
            })
            
            # Keep stream alive with periodic pings
            while self._running:
                try:
                    await asyncio.sleep(self.keepalive_interval)
                    
                    if not self._running:
                        break
                    
                    self.message_count += 1
                    yield self._create_event("ping", {
                        "timestamp": datetime.now().isoformat(),
                        "messageCount": self.message_count
                    })
                    
                except asyncio.CancelledError:
                    logger.info(f"MCP SSE stream cancelled: {self.session_id}")
                    break
                except Exception as e:
                    logger.error(f"Error in MCP SSE stream: {e}")
                    yield self._create_event("error", {
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                    break
                    
        except Exception as e:
            logger.error(f"Fatal error in MCP SSE generator: {e}", exc_info=True)
            yield self._create_event("error", {
                "error": f"Fatal SSE error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        finally:
            logger.info(f"MCP SSE stream closed: {self.session_id}")
            self._running = False
    
    def _create_event(self, event_type: str, data: Dict[str, Any]) -> Dict[str, str]:
        """Create standardized SSE event"""
        return {
            "event": event_type,
            "data": json.dumps(data)
        }
    
    def stop(self):
        """Stop the event generator"""
        self._running = False

class JSONRPCValidator:
    """JSON-RPC request validation utilities"""
    
    @staticmethod
    def validate_jsonrpc_version(data: Dict[str, Any]) -> Optional[JSONResponse]:
        """Validate JSON-RPC version field"""
        if data.get("jsonrpc") != MCPConfig.PROTOCOL_VERSION:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": MCPConfig.PROTOCOL_VERSION,
                    "error": create_error_response(
                        ErrorCode.INVALID_REQUEST,
                        f"Invalid Request: jsonrpc field must be '{MCPConfig.PROTOCOL_VERSION}'"
                    )
                }
            )
        return None
    
    @staticmethod
    def validate_request_structure(data: Any) -> Optional[JSONResponse]:
        """Validate basic request structure"""
        if not isinstance(data, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": MCPConfig.PROTOCOL_VERSION,
                    "error": create_error_response(
                        ErrorCode.INVALID_REQUEST,
                        "Invalid Request: Message must be a JSON object"
                    )
                }
            )
        return None
    
    @staticmethod
    def create_parse_error_response() -> JSONResponse:
        """Create parse error response"""
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": MCPConfig.PROTOCOL_VERSION,
                "error": create_error_response(ErrorCode.PARSE_ERROR, "Parse error: Invalid JSON")
            }
        )

class MCPResponseBuilder:
    """Standardized response builder for MCP endpoints"""
    
    @staticmethod
    def create_mcp_unavailable_response() -> JSONResponse:
        """Create MCP unavailable response"""
        return JSONResponse(
            status_code=503,
            content={
                "jsonrpc": MCPConfig.PROTOCOL_VERSION,
                "error": create_error_response(ErrorCode.INTERNAL_ERROR, "MCP server not available")
            }
        )
    
    @staticmethod
    def create_method_not_allowed_response() -> JSONResponse:
        """Create method not allowed response"""
        return JSONResponse(
            status_code=405,
            content={
                "jsonrpc": MCPConfig.PROTOCOL_VERSION,
                "error": create_error_response(ErrorCode.METHOD_NOT_FOUND, "Method not allowed")
            }
        )
    
    @staticmethod
    def create_sse_info_response() -> JSONResponse:
        """Create SSE endpoint info response"""
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
    
    @staticmethod
    def create_accepted_response() -> JSONResponse:
        """Create accepted response for notifications"""
        return JSONResponse(
            status_code=202,
            content={"status": "accepted"},
            headers={
                "Content-Type": "application/json",
                "Cache-Control": "no-cache"
            }
        )
    
    @staticmethod
    def create_success_response(content: Dict[str, Any], session_id: Optional[str] = None) -> JSONResponse:
        """Create success response with optional session ID"""
        response = JSONResponse(
            status_code=200,
            content=content,
            headers={
                "Content-Type": "application/json",
                "Cache-Control": "no-cache"
            }
        )
        
        if session_id:
            response.headers["Mcp-Session-Id"] = session_id
        
        return response

# =============================================================================
# === Standard MCP Streamable HTTP Endpoint ===
# =============================================================================

@mcp.api_route("", methods=["GET", "POST"])
async def mcp_streamable_http_endpoint(
    request: Request,
    accept: Optional[str] = Header(None)
):
    """
    Optimized MCP Streamable HTTP endpoint implementing 2025-03-26 specification
    
    Features:
    - Modular request handling with improved error responses
    - Optimized SSE streaming with proper cleanup
    - Standardized JSON-RPC validation
    - Enhanced performance and error handling
    
    Reference: https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/
    """
    try:
        if not MCP_AVAILABLE:
            logger.warning("MCP server requested but dependencies not available")
            return MCPResponseBuilder.create_mcp_unavailable_response()
        
        if request.method == "GET":
            return await _handle_get_request(request)
        
        elif request.method == "POST":
            return await _handle_post_request(request)
        
        else:
            return MCPResponseBuilder.create_method_not_allowed_response()
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Critical error in MCP endpoint: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": MCPConfig.PROTOCOL_VERSION,
                "error": create_error_response(ErrorCode.INTERNAL_ERROR, f"Critical error: {str(e)}")
            }
        )

async def _handle_get_request(request: Request) -> JSONResponse:
    """Handle GET requests for MCP endpoint"""
    accept_header = request.headers.get("accept", "")
    
    if "text/event-stream" in accept_header:
        logger.info("MCP GET request - upgrading to SSE stream")
        session_id = str(uuid.uuid4())
        
        event_generator = SSEEventGenerator(session_id)
        response = EventSourceResponse(event_generator.generate_events())
        response.headers["Mcp-Session-Id"] = session_id
        response.headers["Cache-Control"] = "no-cache"
        
        return response
    
    else:
        logger.info("MCP GET request - returning endpoint info")
        return MCPResponseBuilder.create_sse_info_response()

async def _handle_post_request(request: Request) -> JSONResponse:
    """Handle POST requests for MCP endpoint"""
    try:
        message_data = await request.json()
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in MCP POST request")
        return JSONRPCValidator.create_parse_error_response()
    
    # Validate request structure
    validation_error = JSONRPCValidator.validate_request_structure(message_data)
    if validation_error:
        return validation_error
    
    validation_error = JSONRPCValidator.validate_jsonrpc_version(message_data)
    if validation_error:
        return validation_error
    
    # Create and validate MCP message
    try:
        mcp_message = MCPMessage(**message_data)
    except Exception as e:
        logger.warning(f"Invalid MCP message format: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": MCPConfig.PROTOCOL_VERSION,
                "error": create_error_response(ErrorCode.INVALID_REQUEST, f"Invalid Request: {str(e)}")
            }
        )
    
    # Process the request
    logger.info(f"Processing MCP request: {mcp_message.method}")
    response = await handle_mcp_request(mcp_message)
    
    # Handle notifications (no response)
    if response is None:
        return MCPResponseBuilder.create_accepted_response()
    
    # Generate session ID for initialization
    session_id = None
    if mcp_message.method == "initialize" and hasattr(response, 'result'):
        session_id = str(uuid.uuid4())
        logger.info(f"MCP session initialized: {session_id}")
    
    return MCPResponseBuilder.create_success_response(
        response.dict(exclude_none=True),
        session_id
    )

class MCPServerInfo:
    """MCP server information provider"""
    
    @staticmethod
    def get_server_capabilities() -> Dict[str, Any]:
        """Get server capabilities"""
        return {
            "tools": [tool.name for tool in MCP_TOOLS],
            "logging": True,
            "prompts": False,
            "resources": False
        }
    
    @staticmethod
    def get_endpoints() -> Dict[str, str]:
        """Get available endpoints"""
        return {
            "main": "/api/mcp",
            "status": "/api/mcp/status", 
            "info": "/api/mcp/info"
        }
    
    @staticmethod
    def get_compliance_info() -> Dict[str, Any]:
        """Get compliance information"""
        return {
            "standard": "MCP Specification 2024-11-05",
            "transport": "Streamable HTTP",
            "json_rpc": "2.0",
            "features": ["tools", "logging", "streaming"]
        }

# =============================================================================
# === MCP Server Information Endpoints (for debugging/discovery) ===
# =============================================================================

@mcp.get("/status")
async def get_mcp_status():
    """
    Get MCP server status and capabilities
    
    Returns:
        dict: Server status information including:
            - status: "running" or "error"
            - protocol: Protocol name and version
            - transport: Transport method
            - available: Whether server is available
            - capabilities: Supported features
            - endpoints: Available endpoints
            - compliance: Specification compliance info
    
    Example:
        {
            "status": "running",
            "protocol": "Model Context Protocol (MCP)",
            "version": "2024-11-05",
            "transport": "Streamable HTTP",
            "available": true,
            "capabilities": {
                "tools": ["query_knowledge_base", "list_knowledge_bases"],
                "logging": true,
                "prompts": false,
                "resources": false
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
    """
    try:
        if not MCP_AVAILABLE:
            logger.warning("MCP status requested but dependencies not available")
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
                "name": MCPConfig.SERVER_NAME,
                "version": MCPConfig.SERVER_VERSION
            },
            "available": True,
            "capabilities": MCPServerInfo.get_server_capabilities(),
            "endpoints": MCPServerInfo.get_endpoints(),
            "compliance": MCPServerInfo.get_compliance_info()
        }
    except Exception as e:
        logger.error(f"Error getting MCP status: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "protocol": "MCP",
            "transport": "streamable-http",
            "available": False
        }

@mcp.get("/info")
async def get_mcp_info():
    """
    Get detailed MCP server information and usage documentation
    
    Returns:
        dict: Comprehensive server information including:
            - name: Server name
            - description: Server purpose
            - protocol: Protocol details
            - implementation: Technical implementation info
            - usage: Connection and usage instructions
            - tools: Available tools with parameters
            - compliance: Specification compliance
    
    This endpoint provides detailed documentation for developers
    integrating with the MCP server.
    """
    tools_info = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": list(tool.input_schema.get("properties", {}).keys())
        }
        for tool in MCP_TOOLS
    ]
    
    return {
        "name": MCPConfig.SERVER_NAME,
        "description": "Standard MCP server providing access to knowledge base tools",
        "protocol": "Model Context Protocol (MCP)",
        "version": "2024-11-05",
        "transport": "Streamable HTTP",
        "specification": "https://modelcontextprotocol.io/specification/2024-11-05/",
        "implementation": {
            "language": "Python",
            "framework": "FastAPI",
            "server": "Yuxi-Know Knowledge Base",
            "version": MCPConfig.SERVER_VERSION
        },
        "usage": {
            "connection": "Single endpoint at /api/mcp with GET (SSE) and POST (JSON-RPC)",
            "initialization": "Send initialize JSON-RPC request",
            "tools": {
                "list": "Use tools/list method",
                "call": "Use tools/call method with name and arguments"
            },
            "streaming": "Use GET with Accept: text/event-stream for SSE"
        },
        "tools": tools_info,
        "compliance": {
            "standard": "MCP Specification 2024-11-05",
            "transport": "Streamable HTTP (official)",
            "json_rpc": "2.0 (strict)",
            "features": ["tools", "logging", "streaming"]
        }
    }