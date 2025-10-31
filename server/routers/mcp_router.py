import asyncio
import json
import uuid
import traceback
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, Request, HTTPException
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

# Global MCP server instance
_mcp_server_instance = None

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


# =============================================================================
# === MCP Standard Endpoints ===
# =============================================================================

@mcp.get("/status")
async def get_mcp_status():
    """Get MCP server status"""
    try:
        if not MCP_AVAILABLE:
            return {
                "status": "error",
                "error": "MCP dependencies not available",
                "protocol": "MCP",
                "transport": "http"
            }
        
        mcp_server = get_mcp_server()
        return {
            "status": "running",
            "protocol": "MCP",
            "transport": "http",
            "version": "1.0.0",
            "server_info": {
                "name": "Yuxi-Know Knowledge Base MCP Server",
                "version": "1.0.0"
            },
            "available": True
        }
    except Exception as e:
        logger.error(f"Error getting MCP status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "protocol": "MCP",
            "transport": "http",
            "available": False
        }


@mcp.post("/tools/list")
async def list_mcp_tools():
    """List available MCP tools (Standard MCP endpoint)"""
    try:
        if not MCP_AVAILABLE:
            raise HTTPException(status_code=500, detail="MCP not available")
        
        mcp_server = get_mcp_server()
        
        # Get tools from the MCP server
        tools = []
        
        # Add basic query tool
        tools.append({
            "name": "query_knowledge_base",
            "description": "Query all available knowledge bases with intelligent routing",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query_text": {"type": "string", "description": "The query text to search for"},
                    "db_id": {"type": "string", "description": "Specific database ID to query (optional)"},
                    "mode": {"type": "string", "enum": ["local", "global", "hybrid", "naive", "mix"], "default": "mix"},
                    "top_k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100}
                },
                "required": ["query_text"]
            }
        })
        
        # Add list knowledge bases tool
        tools.append({
            "name": "list_knowledge_bases",
            "description": "List all available knowledge bases and their information",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        })
        
        return {
            "tools": tools
        }
        
    except Exception as e:
        logger.error(f"Error listing MCP tools: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")


@mcp.post("/tools/call")
async def call_mcp_tool(request: Request):
    """Call an MCP tool (Standard MCP endpoint)"""
    try:
        if not MCP_AVAILABLE:
            raise HTTPException(status_code=500, detail="MCP not available")
        
        body = await request.json()
        tool_name = body.get("name")
        arguments = body.get("arguments", {})
        
        if not tool_name:
            raise HTTPException(status_code=400, detail="Tool name is required")
        
        mcp_server = get_mcp_server()
        
        # Route to appropriate tool
        if tool_name == "query_knowledge_base":
            result = await mcp_server._query_knowledge_base(arguments)
        elif tool_name == "list_knowledge_bases":
            result = await mcp_server._list_knowledge_bases()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
        
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
        
        return {
            "content": content
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calling MCP tool: {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")


# =============================================================================
# === Legacy Compatibility Endpoints ===
# =============================================================================

@mcp.get("/legacy/info")
async def get_legacy_info():
    """Information about MCP implementation"""
    return {
        "message": "Yuxi-Know MCP Server Implementation",
        "protocol": "Model Context Protocol (MCP)",
        "transport": "HTTP",
        "endpoints": {
            "status": "GET /mcp/status",
            "list_tools": "POST /mcp/tools/list",
            "call_tool": "POST /mcp/tools/call"
        },
        "tools": [
            "query_knowledge_base - Query knowledge bases",
            "list_knowledge_bases - List available knowledge bases"
        ],
        "documentation": "https://modelcontextprotocol.io/"
    }


# =============================================================================
# === Direct Query Endpoints (for compatibility) ===
# =============================================================================

@mcp.post("/query")
async def query_knowledge_base_direct(request: Request):
    """Direct knowledge base query endpoint"""
    try:
        if not MCP_AVAILABLE:
            raise HTTPException(status_code=500, detail="MCP not available")
        
        body = await request.json()
        query_text = body.get("query_text")
        
        if not query_text:
            raise HTTPException(status_code=400, detail="query_text is required")
        
        arguments = {
            "query_text": query_text,
            "db_id": body.get("db_id"),
            "mode": body.get("mode", "mix"),
            "top_k": body.get("top_k", 10)
        }
        
        mcp_server = get_mcp_server()
        result = await mcp_server._query_knowledge_base(arguments)
        
        # Extract content
        content = ""
        if hasattr(result, 'content') and result.content:
            for item in result.content:
                if hasattr(item, 'text'):
                    content += item.text + "\n"
                else:
                    content += str(item) + "\n"
        
        return {
            "success": True,
            "content": content.strip()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in direct query: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.get("/knowledge-bases")
async def list_knowledge_bases_direct():
    """Direct knowledge bases listing endpoint"""
    try:
        if not MCP_AVAILABLE:
            raise HTTPException(status_code=500, detail="MCP not available")
        
        mcp_server = get_mcp_server()
        result = await mcp_server._list_knowledge_bases()
        
        # Extract content
        content = ""
        if hasattr(result, 'content') and result.content:
            for item in result.content:
                if hasattr(item, 'text'):
                    content += item.text + "\n"
                else:
                    content += str(item) + "\n"
        
        # Try to parse as JSON
        try:
            kb_data = json.loads(content.strip())
            return kb_data
        except:
            return {"raw_content": content.strip()}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing knowledge bases: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list knowledge bases: {str(e)}")