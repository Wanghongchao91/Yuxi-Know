"""
Knowledge Base MCP Server

This module implements a Model Context Protocol (MCP) server that exposes
knowledge base retrieval functionality. It provides tools for querying
knowledge bases and listing available databases.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    Tool,
)

from src.knowledge.manager import knowledge_base
from src.utils import logger


class KnowledgeBaseServer:
    """MCP Server for Knowledge Base operations"""
    
    def __init__(self):
        self.server = Server("knowledge-base-server")
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            tools = []
            
            try:
                # Get knowledge base information
                retrievers = knowledge_base.get_retrievers()
                
                # General query tool
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
                                "description": "Specific database ID to query (optional)"
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["local", "global", "hybrid", "naive", "mix"],
                                "default": "mix",
                                "description": "Query mode"
                            },
                            "top_k": {
                                "type": "integer",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 100,
                                "description": "Maximum number of results to return"
                            }
                        },
                        "required": ["query_text"]
                    }
                ))
                
                # Create specific tools for each knowledge base
                for db_id, retriever_info in retrievers.items():
                    # Create safe tool name
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
                                    "enum": ["local", "global", "hybrid", "naive", "mix"],
                                    "default": "mix",
                                    "description": "Query mode"
                                },
                                "top_k": {
                                    "type": "integer",
                                    "default": 10,
                                    "minimum": 1,
                                    "maximum": 100,
                                    "description": "Maximum number of results to return"
                                }
                            },
                            "required": ["query_text"]
                        }
                    ))
                
                # List knowledge bases tool
                tools.append(Tool(
                    name="list_knowledge_bases",
                    description="List all available knowledge bases and their information",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ))
                
            except Exception as e:
                logger.error(f"Error creating tools: {e}")
                # Return basic tool on error
                tools = [Tool(
                    name="query_knowledge_base",
                    description="Query knowledge bases (error occurred during initialization)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query_text": {
                                "type": "string",
                                "description": "The query text"
                            }
                        },
                        "required": ["query_text"]
                    }
                )]
            
            return tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            try:
                if name == "query_knowledge_base":
                    result = await self._query_knowledge_base(arguments)
                elif name == "list_knowledge_bases":
                    result = await self._list_knowledge_bases()
                elif name.startswith("query_"):
                    # Extract database ID from tool name
                    parts = name.split("_")
                    if len(parts) >= 2:
                        db_id = parts[-1]
                        arguments = arguments.copy()
                        arguments["db_id"] = db_id
                        result = await self._query_knowledge_base(arguments)
                    else:
                        raise ValueError(f"Invalid tool name format: {name}")
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return result
                
            except Exception as e:
                logger.error(f"Error calling tool {name}: {e}")
                return [TextContent(
                    type="text",
                    text=f"Error executing tool: {str(e)}"
                )]
    
    async def _query_knowledge_base(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Query knowledge base with given arguments"""
        try:
            query_text = arguments.get("query_text", "")
            db_id = arguments.get("db_id")
            mode = arguments.get("mode", "mix")
            top_k = arguments.get("top_k", 10)
            
            if not query_text:
                return [TextContent(
                    type="text",
                    text="Error: query_text is required"
                )]
            
            # Perform the query
            if db_id:
                # Query specific database
                retriever = knowledge_base.get_retriever(db_id)
                if not retriever:
                    return [TextContent(
                        type="text",
                        text=f"Error: Database {db_id} not found"
                    )]
                
                results = await asyncio.to_thread(
                    retriever.query,
                    query_text,
                    mode=mode,
                    top_k=top_k
                )
            else:
                # Query all databases
                results = await asyncio.to_thread(
                    knowledge_base.query,
                    query_text,
                    mode=mode,
                    top_k=top_k
                )
            
            # Format results
            if results:
                formatted_results = []
                for i, result in enumerate(results[:top_k], 1):
                    if hasattr(result, 'page_content') and hasattr(result, 'metadata'):
                        content = result.page_content
                        metadata = result.metadata
                        source = metadata.get('source', 'Unknown')
                        formatted_results.append(f"{i}. Source: {source}\nContent: {content}\n")
                    else:
                        formatted_results.append(f"{i}. {str(result)}\n")
                
                response_text = f"Found {len(results)} results for query: '{query_text}'\n\n" + "\n".join(formatted_results)
            else:
                response_text = f"No results found for query: '{query_text}'"
            
            return [TextContent(type="text", text=response_text)]
            
        except Exception as e:
            logger.error(f"Error in _query_knowledge_base: {e}")
            return [TextContent(
                type="text",
                text=f"Error querying knowledge base: {str(e)}"
            )]
    
    async def _list_knowledge_bases(self) -> List[TextContent]:
        """List all available knowledge bases"""
        try:
            databases = knowledge_base.get_databases()
            retrievers = knowledge_base.get_retrievers()
            
            if not retrievers:
                return [TextContent(
                    type="text",
                    text="No knowledge bases available"
                )]
            
            kb_list = []
            for db_id, retriever_info in retrievers.items():
                name = retriever_info.get("name", "Unknown")
                description = retriever_info.get("description", "No description")
                kb_type = retriever_info.get("type", "Unknown")
                
                kb_list.append(f"- ID: {db_id}")
                kb_list.append(f"  Name: {name}")
                kb_list.append(f"  Type: {kb_type}")
                kb_list.append(f"  Description: {description}")
                kb_list.append("")
            
            response_text = f"Available Knowledge Bases ({len(retrievers)}):\n\n" + "\n".join(kb_list)
            return [TextContent(type="text", text=response_text)]
            
        except Exception as e:
            logger.error(f"Error in _list_knowledge_bases: {e}")
            return [TextContent(
                type="text",
                text=f"Error listing knowledge bases: {str(e)}"
            )]
    
    async def run_stdio(self):
        """Run the server with stdio transport"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="knowledge-base-server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities={}
                    )
                )
            )


async def main():
    """Main entry point for the MCP server"""
    server = KnowledgeBaseServer()
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())