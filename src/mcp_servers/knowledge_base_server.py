#!/usr/bin/env python3
"""
Knowledge Base MCP Server

This MCP server exposes knowledge base retrieval functionality from Yuxi-Know
to other applications via the Model Context Protocol.
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
)
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import knowledge base functionality
try:
    import os
    import sys
    
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.knowledge.manager import knowledge_base
    from src.utils import logger as app_logger
except ImportError as e:
    logger.error(f"Failed to import knowledge base modules: {e}")
    sys.exit(1)


class KnowledgeQueryModel(BaseModel):
    """Model for knowledge base query parameters"""
    query_text: str = Field(description="The query text to search for")
    db_id: Optional[str] = Field(default=None, description="Specific database ID to query (optional)")
    mode: Optional[str] = Field(default="mix", description="Query mode (local, global, hybrid, naive, mix)")
    top_k: Optional[int] = Field(default=10, description="Maximum number of results to return")


class KnowledgeBaseServer:
    """MCP Server for Knowledge Base operations"""
    
    def __init__(self):
        self.server = Server("knowledge-base-server")
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available knowledge base tools"""
            tools = []
            
            try:
                # Get all available databases
                databases = knowledge_base.get_databases()
                retrievers = knowledge_base.get_retrievers()
                
                # Create a general query tool
                tools.append(Tool(
                    name="query_knowledge_base",
                    description="Query all available knowledge bases with intelligent routing",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query_text": {
                                "type": "string",
                                "description": "The query text to search for"
                            },
                            "db_id": {
                                "type": "string",
                                "description": "Specific database ID to query (optional)",
                                "default": None
                            },
                            "mode": {
                                "type": "string",
                                "description": "Query mode",
                                "enum": ["local", "global", "hybrid", "naive", "mix"],
                                "default": "mix"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 100
                            }
                        },
                        "required": ["query_text"]
                    }
                ))
                
                # Create specific tools for each database
                for db_id, retriever_info in retrievers.items():
                    safe_name = retriever_info["name"].replace(" ", "_").replace("-", "_")[:30]
                    tool_name = f"query_{safe_name}_{db_id}"
                    
                    tools.append(Tool(
                        name=tool_name,
                        description=f"Query {retriever_info['name']} knowledge base. {retriever_info.get('description', '')}",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query_text": {
                                    "type": "string",
                                    "description": "The query text to search for"
                                },
                                "mode": {
                                    "type": "string",
                                    "description": "Query mode",
                                    "enum": ["local", "global", "hybrid", "naive", "mix"],
                                    "default": "mix"
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Maximum number of results to return",
                                    "default": 10,
                                    "minimum": 1,
                                    "maximum": 100
                                }
                            },
                            "required": ["query_text"]
                        }
                    ))
                
                # Add database listing tool
                tools.append(Tool(
                    name="list_knowledge_bases",
                    description="List all available knowledge bases and their information",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ))
                
                logger.info(f"Listed {len(tools)} knowledge base tools")
                
            except Exception as e:
                logger.error(f"Error listing tools: {e}")
                # Return at least the basic query tool even if database listing fails
                tools = [Tool(
                    name="query_knowledge_base",
                    description="Query knowledge bases (error occurred during initialization)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query_text": {"type": "string", "description": "The query text"}
                        },
                        "required": ["query_text"]
                    }
                )]
            
            return ListToolsResult(tools=tools)
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls"""
            try:
                logger.info(f"Calling tool: {name} with arguments: {arguments}")
                
                if name == "query_knowledge_base":
                    return await self._query_knowledge_base(arguments)
                elif name == "list_knowledge_bases":
                    return await self._list_knowledge_bases()
                elif name.startswith("query_"):
                    # Extract database ID from tool name
                    parts = name.split("_")
                    if len(parts) >= 2:
                        db_id = parts[-1]  # Last part should be the db_id
                        arguments["db_id"] = db_id
                        return await self._query_knowledge_base(arguments)
                
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Unknown tool: {name}"
                    )],
                    isError=True
                )
                
            except Exception as e:
                logger.error(f"Error calling tool {name}: {e}")
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error executing tool {name}: {str(e)}"
                    )],
                    isError=True
                )
    
    async def _query_knowledge_base(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Query knowledge base with given arguments"""
        try:
            query_text = arguments.get("query_text", "")
            db_id = arguments.get("db_id")
            mode = arguments.get("mode", "mix")
            top_k = arguments.get("top_k", 10)
            
            if not query_text:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text="Error: query_text is required"
                    )],
                    isError=True
                )
            
            # Prepare query parameters
            query_params = {
                "mode": mode,
                "top_k": top_k
            }
            
            # Execute query
            if db_id:
                # Query specific database
                result = await knowledge_base.aquery(query_text, db_id=db_id, **query_params)
                source_info = f"Database: {db_id}"
            else:
                # Query all databases and aggregate results
                retrievers = knowledge_base.get_retrievers()
                all_results = []
                
                for current_db_id, retriever_info in retrievers.items():
                    try:
                        db_result = await knowledge_base.aquery(query_text, db_id=current_db_id, **query_params)
                        if db_result:
                            # Add source information to each result
                            if isinstance(db_result, list):
                                for item in db_result:
                                    if isinstance(item, dict):
                                        item["source_db"] = current_db_id
                                        item["source_name"] = retriever_info["name"]
                                all_results.extend(db_result if isinstance(db_result, list) else [db_result])
                            else:
                                all_results.append({
                                    "content": db_result,
                                    "source_db": current_db_id,
                                    "source_name": retriever_info["name"]
                                })
                    except Exception as e:
                        logger.warning(f"Error querying database {current_db_id}: {e}")
                        continue
                
                result = all_results[:top_k]  # Limit total results
                source_info = f"Searched {len(retrievers)} databases"
            
            # Format result
            if isinstance(result, list) and len(result) > 0:
                formatted_results = []
                for i, item in enumerate(result, 1):
                    if isinstance(item, dict):
                        content = item.get("content", str(item))
                        source = item.get("source_name", item.get("source_db", "Unknown"))
                        score = item.get("score", "N/A")
                        formatted_results.append(f"Result {i} (Source: {source}, Score: {score}):\n{content}")
                    else:
                        formatted_results.append(f"Result {i}:\n{str(item)}")
                
                response_text = f"Query: {query_text}\n{source_info}\nFound {len(result)} results:\n\n" + "\n\n---\n\n".join(formatted_results)
            else:
                response_text = f"Query: {query_text}\n{source_info}\nNo results found."
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=response_text
                )]
            )
            
        except Exception as e:
            logger.error(f"Error in _query_knowledge_base: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error querying knowledge base: {str(e)}"
                )],
                isError=True
            )
    
    async def _list_knowledge_bases(self) -> CallToolResult:
        """List all available knowledge bases"""
        try:
            databases = knowledge_base.get_databases()
            retrievers = knowledge_base.get_retrievers()
            
            db_list = []
            for db_id, retriever_info in retrievers.items():
                db_info = {
                    "id": db_id,
                    "name": retriever_info["name"],
                    "description": retriever_info.get("description", "No description available"),
                    "metadata": retriever_info.get("metadata", {})
                }
                db_list.append(db_info)
            
            response_text = f"Available Knowledge Bases ({len(db_list)} total):\n\n"
            for db in db_list:
                response_text += f"• {db['name']} (ID: {db['id']})\n"
                response_text += f"  Description: {db['description']}\n"
                if db['metadata']:
                    response_text += f"  Metadata: {json.dumps(db['metadata'], indent=2)}\n"
                response_text += "\n"
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=response_text
                )]
            )
            
        except Exception as e:
            logger.error(f"Error in _list_knowledge_bases: {e}")
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error listing knowledge bases: {str(e)}"
                )],
                isError=True
            )
    
    async def run(self):
        """Run the MCP server"""
        logger.info("Starting Knowledge Base MCP Server...")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())


async def main():
    """Main entry point"""
    server = KnowledgeBaseServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())