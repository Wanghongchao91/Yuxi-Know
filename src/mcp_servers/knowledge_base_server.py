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
    
    from src.knowledge import knowledge_base
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
        """Query knowledge base with given arguments - supports batch queries and reranking"""
        try:
            # Support both single query text and array of query texts
            query_text_input = arguments.get("query_text", "")
            db_id_input = arguments.get("db_id")
            mode = arguments.get("mode", "mix")
            top_k = arguments.get("top_k", 10)
            enable_rerank = arguments.get("enable_rerank", True)
            rerank_model = arguments.get("rerank_model", "bge-reranker-v2-m3")
            
            # Normalize query texts to list
            if isinstance(query_text_input, str):
                query_texts = [query_text_input]
            elif isinstance(query_text_input, list):
                query_texts = query_text_input
            else:
                query_texts = [str(query_text_input)]
            
            if not query_texts or not query_texts[0]:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text="Error: query_text is required"
                    )],
                    isError=True
                )
            
            # Normalize db_ids to list
            if db_id_input is None:
                db_ids = None  # Query all databases
            elif isinstance(db_id_input, str):
                db_ids = [db_id_input]
            elif isinstance(db_id_input, list) and db_id_input:
                db_ids = db_id_input
            else:
                db_ids = None  # Query all databases
            
            # Prepare query parameters
            query_params = {
                "mode": mode,
                "top_k": top_k
            }
            
            # Collect all results from all queries
            all_results = []
            query_metadata = {
                "total_queries": len(query_texts),
                "databases_queried": [],
                "rerank_enabled": enable_rerank,
                "rerank_model": rerank_model if enable_rerank else None
            }
            
            # Execute queries for each query text
            for query_idx, query_text in enumerate(query_texts):
                if db_ids:
                    # Query specific databases
                    for db_id in db_ids:
                        try:
                            result = await knowledge_base.aquery(query_text, db_id=db_id, **query_params)
                            if result and db_id not in query_metadata["databases_queried"]:
                                query_metadata["databases_queried"].append(db_id)
                            
                            # Process and add source info
                            processed_results = self._process_query_result(result, db_id, f"Database: {db_id}")
                            for item in processed_results:
                                item["query_index"] = query_idx
                                item["original_query"] = query_text
                            all_results.extend(processed_results)
                            
                        except Exception as e:
                            logger.warning(f"Error querying database {db_id} with query '{query_text}': {e}")
                            continue
                else:
                    # Query all databases
                    retrievers = knowledge_base.get_retrievers()
                    query_metadata["databases_queried"] = list(retrievers.keys())
                    
                    for current_db_id, retriever_info in retrievers.items():
                        try:
                            db_result = await knowledge_base.aquery(query_text, db_id=current_db_id, **query_params)
                            
                            # Process and add source info
                            processed_results = self._process_query_result(
                                db_result, current_db_id, retriever_info.get("name", current_db_id)
                            )
                            for item in processed_results:
                                item["query_index"] = query_idx
                                item["original_query"] = query_text
                            all_results.extend(processed_results)
                            
                        except Exception as e:
                            logger.warning(f"Error querying database {current_db_id} with query '{query_text}': {e}")
                            continue
            
            # Apply reranking if enabled and we have multiple results
            if enable_rerank and len(all_results) > 1:
                try:
                    from src.models.rerank import get_reranker
                    reranker = get_reranker(rerank_model)
                    
                    # Prepare reranking pairs: (query, document) for each result
                    rerank_pairs = []
                    for result in all_results:
                        query = result.get("original_query", "")
                        content = result.get("content", "")
                        if isinstance(content, dict):
                            content = json.dumps(content, ensure_ascii=False)
                        elif not isinstance(content, str):
                            content = str(content)
                        rerank_pairs.append([query, content])
                    
                    # Get relevance scores
                    if rerank_pairs:
                        scores = await reranker.acompute_score(rerank_pairs, normalize=True)
                        
                        # Add scores to results and sort by relevance
                        for idx, (result, score) in enumerate(zip(all_results, scores)):
                            result["relevance_score"] = score
                        
                        # Sort by relevance score (descending)
                        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                        
                        logger.info(f"Reranked {len(all_results)} results with model {rerank_model}")
                        
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}. Using original results order.")
            
            # Limit results to top_k
            final_results = all_results[:top_k]
            
            # Add metadata to results
            result_data = {
                "results": final_results,
                "metadata": query_metadata,
                "total_results": len(final_results),
                "reranked": enable_rerank
            }
            
            # Ensure result is JSON-serializable
            try:
                result_text = json.dumps(result_data, ensure_ascii=False, indent=2)
            except (TypeError, ValueError) as json_error:
                logger.warning(f"Result could not be JSON serialized: {json_error}")
                # Fallback to basic structure
                fallback_data = {
                    "results": [{"text": str(result), "type": "text"} for result in final_results],
                    "metadata": {"error": "JSON serialization failed, using fallback"}
                }
                result_text = json.dumps(fallback_data, ensure_ascii=False)
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=result_text
                )]
            )
            
        except Exception as e:
            logger.error(f"Error in _query_knowledge_base: {e}", exc_info=True)
            error_msg = f"Error querying knowledge base: {str(e)}"
            
            # Provide specific guidance based on error type
            if "Expecting value" in str(e):
                error_msg += "\n\nThis error usually means the knowledge base returned empty or invalid JSON results. Try:\n"
                error_msg += "1. Check if the database ID is correct using list_knowledge_bases first\n"
                error_msg += "2. Verify the database exists and contains data\n"
                error_msg += "3. Try a different search term or query mode (local, global, hybrid, naive, mix)\n"
                error_msg += "4. Check if the database was properly initialized"
            elif "not found" in str(e).lower():
                error_msg += "\n\nDatabase not found. Please call list_knowledge_bases to get available database IDs."
            elif "empty" in str(e).lower() or "none" in str(e).lower():
                error_msg += "\n\nQuery returned no results. Try a different search term or check if the database contains data."
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=error_msg
                )],
                isError=True
            )
    
    def _process_query_result(self, result: Any, db_id: str, source_name: str) -> List[Dict[str, Any]]:
        """Process query result and add metadata"""
        processed_results = []
        
        if result is None or result == "":
            return processed_results
        
        if isinstance(result, str):
            if result.strip() == "":
                return processed_results
            else:
                # Try to parse as JSON
                try:
                    parsed_result = json.loads(result)
                    result = parsed_result
                except json.JSONDecodeError:
                    # If not valid JSON, treat as text content
                    processed_results.append({
                        "content": result,
                        "type": "text",
                        "source_db": db_id,
                        "source_name": source_name
                    })
                    return processed_results
        
        # Handle list results
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    item["source_db"] = db_id
                    item["source_name"] = source_name
                    processed_results.append(item)
                else:
                    processed_results.append({
                        "content": str(item),
                        "type": "text",
                        "source_db": db_id,
                        "source_name": source_name
                    })
        # Handle single dict result
        elif isinstance(result, dict):
            result["source_db"] = db_id
            result["source_name"] = source_name
            processed_results.append(result)
        else:
            # Handle other types
            processed_results.append({
                "content": str(result),
                "type": "text",
                "source_db": db_id,
                "source_name": source_name
            })
        
        return processed_results
    
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