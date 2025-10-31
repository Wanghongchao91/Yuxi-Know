import asyncio
import json
import traceback
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

from src.utils import logger

# Import MCP server functionality
try:
    from src.mcp_servers.knowledge_base_server import KnowledgeBaseServer
    from src.knowledge.manager import knowledge_base
except ImportError as e:
    logger.error(f"Failed to import MCP server modules: {e}")
    KnowledgeBaseServer = None

mcp = APIRouter(prefix="/mcp", tags=["mcp"])

# Pydantic models for request/response
class MCPToolListResponse(BaseModel):
    """Response model for MCP tool listing"""
    tools: List[Dict[str, Any]]
    count: int

class MCPToolCallRequest(BaseModel):
    """Request model for MCP tool calls"""
    tool_name: str = Field(description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(default={}, description="Arguments for the tool")

class MCPToolCallResponse(BaseModel):
    """Response model for MCP tool calls"""
    success: bool
    content: str
    error: Optional[str] = None

class KnowledgeQueryRequest(BaseModel):
    """Request model for knowledge base queries"""
    query_text: str = Field(description="The query text to search for")
    db_id: Optional[str] = Field(default=None, description="Specific database ID to query (optional)")
    mode: Optional[str] = Field(default="mix", description="Query mode (local, global, hybrid, naive, mix)")
    top_k: Optional[int] = Field(default=10, description="Maximum number of results to return")

# Global MCP server instance
_mcp_server_instance = None

def get_mcp_server():
    """Get or create MCP server instance"""
    global _mcp_server_instance
    if _mcp_server_instance is None:
        if KnowledgeBaseServer is None:
            raise HTTPException(status_code=500, detail="MCP server not available")
        _mcp_server_instance = KnowledgeBaseServer()
    return _mcp_server_instance

# =============================================================================
# === MCP 服务管理分组 ===
# =============================================================================

@mcp.get("/tools", response_model=MCPToolListResponse)
async def list_mcp_tools():
    """列出所有可用的MCP工具"""
    try:
        server = get_mcp_server()
        
        # 模拟调用list_tools方法
        tools = []
        
        # 获取知识库信息
        try:
            databases = knowledge_base.get_databases()
            retrievers = knowledge_base.get_retrievers()
            
            # 通用查询工具
            tools.append({
                "name": "query_knowledge_base",
                "description": "Query all available knowledge bases with intelligent routing",
                "input_schema": {
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
            
            # 为每个知识库创建专用工具
            for db_id, retriever_info in retrievers.items():
                safe_name = retriever_info["name"].replace(" ", "_").replace("-", "_")[:30]
                tool_name = f"query_{safe_name}_{db_id}"
                
                tools.append({
                    "name": tool_name,
                    "description": f"Query {retriever_info['name']} knowledge base. {retriever_info.get('description', '')}",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query_text": {"type": "string", "description": "The query text to search for"},
                            "mode": {"type": "string", "enum": ["local", "global", "hybrid", "naive", "mix"], "default": "mix"},
                            "top_k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100}
                        },
                        "required": ["query_text"]
                    }
                })
            
            # 知识库列表工具
            tools.append({
                "name": "list_knowledge_bases",
                "description": "List all available knowledge bases and their information",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting knowledge base info: {e}")
            # 返回基本工具
            tools = [{
                "name": "query_knowledge_base",
                "description": "Query knowledge bases (error occurred during initialization)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query_text": {"type": "string", "description": "The query text"}
                    },
                    "required": ["query_text"]
                }
            }]
        
        return MCPToolListResponse(tools=tools, count=len(tools))
        
    except Exception as e:
        logger.error(f"Error listing MCP tools: {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to list MCP tools: {str(e)}")

@mcp.post("/tools/call", response_model=MCPToolCallResponse)
async def call_mcp_tool(request: MCPToolCallRequest):
    """调用指定的MCP工具"""
    try:
        server = get_mcp_server()
        
        # 根据工具名称路由到相应的处理方法
        if request.tool_name == "query_knowledge_base":
            result = await server._query_knowledge_base(request.arguments)
        elif request.tool_name == "list_knowledge_bases":
            result = await server._list_knowledge_bases()
        elif request.tool_name.startswith("query_"):
            # 从工具名称中提取数据库ID
            parts = request.tool_name.split("_")
            if len(parts) >= 2:
                db_id = parts[-1]
                arguments = request.arguments.copy()
                arguments["db_id"] = db_id
                result = await server._query_knowledge_base(arguments)
            else:
                raise HTTPException(status_code=400, detail=f"Invalid tool name format: {request.tool_name}")
        else:
            raise HTTPException(status_code=404, detail=f"Unknown tool: {request.tool_name}")
        
        # 提取结果内容
        if hasattr(result, 'content') and result.content:
            content = result.content[0].text if result.content else "No content"
            success = not getattr(result, 'isError', False)
            error = None if success else content
        else:
            content = str(result)
            success = True
            error = None
        
        return MCPToolCallResponse(
            success=success,
            content=content,
            error=error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calling MCP tool {request.tool_name}: {e}, {traceback.format_exc()}")
        return MCPToolCallResponse(
            success=False,
            content="",
            error=f"Error executing tool: {str(e)}"
        )

# =============================================================================
# === 知识库查询快捷接口 ===
# =============================================================================

@mcp.post("/query", response_model=MCPToolCallResponse)
async def query_knowledge_base_direct(request: KnowledgeQueryRequest):
    """直接查询知识库的快捷接口"""
    try:
        server = get_mcp_server()
        
        arguments = {
            "query_text": request.query_text,
            "db_id": request.db_id,
            "mode": request.mode,
            "top_k": request.top_k
        }
        
        result = await server._query_knowledge_base(arguments)
        
        # 提取结果内容
        if hasattr(result, 'content') and result.content:
            content = result.content[0].text if result.content else "No content"
            success = not getattr(result, 'isError', False)
            error = None if success else content
        else:
            content = str(result)
            success = True
            error = None
        
        return MCPToolCallResponse(
            success=success,
            content=content,
            error=error
        )
        
    except Exception as e:
        logger.error(f"Error in direct knowledge base query: {e}, {traceback.format_exc()}")
        return MCPToolCallResponse(
            success=False,
            content="",
            error=f"Error querying knowledge base: {str(e)}"
        )

@mcp.get("/knowledge-bases")
async def list_knowledge_bases_direct():
    """直接列出知识库的快捷接口"""
    try:
        server = get_mcp_server()
        result = await server._list_knowledge_bases()
        
        # 提取结果内容
        if hasattr(result, 'content') and result.content:
            content = result.content[0].text if result.content else "No content"
            success = not getattr(result, 'isError', False)
        else:
            content = str(result)
            success = True
        
        if success:
            return {"success": True, "content": content}
        else:
            raise HTTPException(status_code=500, detail=content)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing knowledge bases: {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error listing knowledge bases: {str(e)}")

# =============================================================================
# === MCP 服务器状态 ===
# =============================================================================

@mcp.get("/status")
async def get_mcp_server_status():
    """获取MCP服务器状态"""
    try:
        server = get_mcp_server()
        
        # 检查知识库连接
        try:
            databases = knowledge_base.get_databases()
            retrievers = knowledge_base.get_retrievers()
            kb_status = "connected"
            kb_count = len(retrievers)
        except Exception as e:
            kb_status = f"error: {str(e)}"
            kb_count = 0
        
        return {
            "status": "running",
            "server_type": "knowledge_base_mcp_server",
            "knowledge_base_status": kb_status,
            "knowledge_base_count": kb_count,
            "available": KnowledgeBaseServer is not None
        }
        
    except Exception as e:
        logger.error(f"Error getting MCP server status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "available": False
        }