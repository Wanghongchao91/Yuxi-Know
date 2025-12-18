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
import time
from collections import OrderedDict
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


class RerankConfig:
    """Internal configuration for reranking parameters"""
    DEFAULT_ENABLE_RERANK = True
    DEFAULT_RERANK_MODEL = "bge-reranker-v2-m3"
    DEFAULT_RERANK_STRATEGY = "auto"
    DEFAULT_KG_WEIGHT = None  # Auto calculate
    DEFAULT_DIVERSITY_BOOST = True


class KnowledgeBaseServer:
    """MCP Server for Knowledge Base operations"""
    
    def __init__(self):
        self.server = Server("knowledge-base-server")
        self._cache_store: OrderedDict = OrderedDict()
        self._cache_lock = asyncio.Lock()
        self._inflight: Dict[str, asyncio.Task] = {}
        self._cache_ttl_seconds = 300
        self._max_cache_size = 1000
        self._batch_semaphore = asyncio.Semaphore(8)
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
                    description="必须先调用 list_knowledge_bases 获取 db_id；单库查询（db_id 必填），一次仅查询一个知识库；如需查询多个库请多次调用。",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query_text": {
                                "type": "string",
                                "description": "查询文本（支持单条或批量）"
                            },
                            "db_id": {
                                "type": "string",
                                "description": "必填。要查询的知识库 ID（先通过 list_knowledge_bases 获取）"
                            },
                            "mode": {
                                "type": "string",
                                "description": "查询模式",
                                "enum": ["local", "global", "hybrid", "naive", "mix"]
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "返回数量上限",
                                "minimum": 1,
                                "maximum": 100
                            }
                        },
                        "required": ["query_text", "db_id"]
                    }
                ))
                
                # Create specific tools for each database
                for db_id, retriever_info in retrievers.items():
                    safe_name = retriever_info["name"].replace(" ", "_").replace("-", "_")[:30]
                    tool_name = f"query_{safe_name}_{db_id}"
                    tools.append(Tool(
                        name=tool_name,
                        description=f"查询 {retriever_info['name']} 知识库（工具名已包含 db_id，无需传参）。{retriever_info.get('description', '')}",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query_text": {
                                    "type": "string",
                                    "description": "查询文本"
                                },
                                "mode": {
                                    "type": "string",
                                    "description": "查询模式",
                                    "enum": ["local", "global", "hybrid", "naive", "mix"]
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "返回数量上限",
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
        try:
            query_text_input = arguments.get("query_text", "")
            db_id = arguments.get("db_id")
            meta_args = {k: v for k, v in arguments.items() if k not in {"query_text", "db_id"}}

            if isinstance(query_text_input, list):
                texts: List[str] = [t for t in query_text_input if isinstance(t, str) and t.strip()]
                if not texts:
                    return CallToolResult(
                        content=[TextContent(type="text", text="错误：缺少必填参数 query_text")],
                        isError=True
                    )
                results: List[Any] = []
                async def _run_one(q: str) -> Any:
                    return await self._cached_query_one(q.strip(), db_id, meta_args)
                tasks = []
                for q in texts:
                    await self._batch_semaphore.acquire()
                    tasks.append(asyncio.create_task(self._run_with_semaphore(_run_one, q)))
                aggregated = await asyncio.gather(*tasks, return_exceptions=True)
                for r in aggregated:
                    if isinstance(r, Exception):
                        results.append({"error": str(r)})
                    else:
                        results.append(r)
                text_out = json.dumps({"results": results}, ensure_ascii=False, indent=2)
                return CallToolResult(content=[TextContent(type="text", text=text_out)])

            if not isinstance(query_text_input, str) or not query_text_input.strip():
                return CallToolResult(
                    content=[TextContent(type="text", text="错误：缺少必填参数 query_text")],
                    isError=True
                )

            if not isinstance(db_id, str) or not db_id.strip():
                return CallToolResult(
                    content=[TextContent(type="text", text="错误：缺少必填参数 db_id（请先调用 list_knowledge_bases 获取）")],
                    isError=True
                )

            raw_result = await self._cached_query_one(query_text_input.strip(), db_id, meta_args)

            if raw_result is None or raw_result == "":
                return CallToolResult(
                    content=[TextContent(type="text", text="无结果")],
                    isError=True
                )

            if isinstance(raw_result, (dict, list)):
                text_out = json.dumps(raw_result, ensure_ascii=False, indent=2)
            else:
                text_out = str(raw_result)

            return CallToolResult(content=[TextContent(type="text", text=text_out)])

        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"查询知识库出错: {str(e)}")],
                isError=True
            )

    async def _run_with_semaphore(self, func, q: str) -> Any:
        try:
            return await func(q)
        finally:
            self._batch_semaphore.release()

    def _make_cache_key(self, query: str, db_id: str, meta: Dict[str, Any]) -> str:
        qn = query.strip().lower()
        mid = (db_id or "").strip()
        clean_meta = {k: v for k, v in meta.items() if v is not None}
        if "top_k" in clean_meta:
            try:
                clean_meta["top_k"] = max(1, min(int(clean_meta["top_k"]), 100))
            except Exception:
                clean_meta["top_k"] = 10
        meta_str = json.dumps(clean_meta, sort_keys=True, ensure_ascii=False)
        return f"{mid}|{meta_str}|{qn}"

    async def _cached_query_one(self, query: str, db_id: str, meta: Dict[str, Any]) -> Any:
        tk = max(1, min(int((meta or {}).get("top_k", 10)), 100))
        meta_norm = dict(meta or {})
        meta_norm["top_k"] = tk
        key = self._make_cache_key(query, db_id, meta_norm)
        now = time.monotonic()
        async with self._cache_lock:
            if key in self._cache_store:
                expiry, value = self._cache_store[key]
                if expiry > now:
                    self._cache_store.move_to_end(key)
                    return value
                else:
                    del self._cache_store[key]
            existing = self._inflight.get(key)
            if existing:
                fut = existing
            else:
                fut = asyncio.create_task(knowledge_base.aquery(query, db_id=db_id, **meta_norm))
                self._inflight[key] = fut
        try:
            result = await fut
        finally:
            async with self._cache_lock:
                self._inflight.pop(key, None)
        async with self._cache_lock:
            self._cache_store[key] = (now + self._cache_ttl_seconds, result)
            if len(self._cache_store) > self._max_cache_size:
                self._cache_store.popitem(last=False)
        return result
    
    def _process_query_result(self, result: Any, db_id: str, source_name: str) -> List[Dict[str, Any]]:
        """Process query result and add metadata with knowledge base type detection"""
        processed_results = []
        
        if result is None or result == "":
            return processed_results
        
        # Detect knowledge base type from source name or db_id
        kb_type = self._detect_kb_type(db_id, source_name)
        
        if isinstance(result, str):
            if result.strip() == "":
                return processed_results
            else:
                # Handle LightRAG knowledge graph output
                if kb_type == "lightrag" and self._is_knowledge_graph_output(result):
                    return self._process_lightrag_kg_output(result, db_id, source_name)
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
                            "source_name": source_name,
                            "kb_type": kb_type
                        })
                        return processed_results
        
        # Handle list results (ChromaDB/Milvus vector results)
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    item["source_db"] = db_id
                    item["source_name"] = source_name
                    item["kb_type"] = kb_type
                    # Add data type based on content structure
                    item["data_type"] = self._detect_data_type(item)
                    processed_results.append(item)
                else:
                    processed_results.append({
                        "content": str(item),
                        "type": "text",
                        "source_db": db_id,
                        "source_name": source_name,
                        "kb_type": kb_type,
                        "data_type": "text"
                    })
        # Handle single dict result
        elif isinstance(result, dict):
            result["source_db"] = db_id
            result["source_name"] = source_name
            result["kb_type"] = kb_type
            result["data_type"] = self._detect_data_type(result)
            processed_results.append(result)
        else:
            # Handle other types
            processed_results.append({
                "content": str(result),
                "type": "text",
                "source_db": db_id,
                "source_name": source_name,
                "kb_type": kb_type,
                "data_type": "text"
            })
        
        return processed_results
    
    def _detect_kb_type(self, db_id: str, source_name: str) -> str:
        try:
            retrievers = knowledge_base.get_retrievers()
            info = retrievers.get(db_id)
            if info:
                meta = info.get("metadata", {})
                kb_type = meta.get("kb_type") or info.get("kb_type") or ""
                if isinstance(kb_type, str) and kb_type:
                    return kb_type.lower()
        except Exception:
            pass
        if "lightrag" in source_name.lower() or "lightrag" in db_id.lower():
            return "lightrag"
        if "chroma" in source_name.lower() or "chroma" in db_id.lower():
            return "chroma"
        if "milvus" in source_name.lower() or "milvus" in db_id.lower():
            return "milvus"
        return "unknown"
    
    def _detect_data_type(self, result: Dict[str, Any]) -> str:
        """Detect data type based on content structure"""
        content = result.get("content", {})
        if isinstance(content, dict):
            if "entity" in content or "entity1" in content or "entity2" in content:
                return "knowledge_graph"
            elif "chunks" in content or "documents" in content:
                return "vector_result"
            else:
                return "structured"
        elif isinstance(content, str):
            if self._is_knowledge_graph_output(content):
                return "knowledge_graph"
            else:
                return "text"
        else:
            return "unknown"
    
    def _is_knowledge_graph_output(self, text: str) -> bool:
        """Check if text contains knowledge graph data patterns"""
        kg_patterns = [
            "entity:", "entities:", "relationship:", "relationships:",
            "知识图谱", "实体", "关系", "[knowledge graph data]",
            "knowledge graph data (entity)",
            "knowledge graph data (relationship)",
            "document chunks",
            "参考文档",
            "```json"
        ]
        text_lower = text.lower()
        return any(pattern.lower() in text_lower for pattern in kg_patterns)
    
    def _process_lightrag_kg_output(self, kg_text: str, db_id: str, source_name: str) -> List[Dict[str, Any]]:
        """Process LightRAG knowledge graph output into structured results"""
        processed_results = []
        
        # Try to parse as JSON first
        try:
            parsed_data = json.loads(kg_text)
            if isinstance(parsed_data, dict):
                # Handle structured knowledge graph data
                entities = parsed_data.get("entities", [])
                relationships = parsed_data.get("relationships", [])
                
                # Process entities
                for entity in entities:
                    if isinstance(entity, dict):
                        processed_results.append({
                            "content": entity,
                            "type": "entity",
                            "source_db": db_id,
                            "source_name": source_name,
                            "kb_type": "lightrag",
                            "data_type": "knowledge_graph"
                        })
                
                # Process relationships
                for rel in relationships:
                    if isinstance(rel, dict):
                        processed_results.append({
                            "content": rel,
                            "type": "relationship",
                            "source_db": db_id,
                            "source_name": source_name,
                            "kb_type": "lightrag",
                            "data_type": "knowledge_graph"
                        })
                
                return processed_results
        except json.JSONDecodeError:
            pass
        
        # Parse LightRAG's formatted text output
        # The format is: "Knowledge Graph Data (Entity):" followed by JSON objects
        # and "Knowledge Graph Data (Relationship):" followed by JSON objects
        
        lines = kg_text.strip().split('\n')
        current_section = None
        json_buffer = ""
        in_json = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect section headers
            if "Knowledge Graph Data (Entity)" in line or "实体" in line.lower():
                current_section = "entity"
                continue
            elif "Knowledge Graph Data (Relationship)" in line or "关系" in line.lower():
                current_section = "relationship"
                continue
            elif "Document Chunks" in line or "参考" in line:
                current_section = "chunks"
                continue
            
            # Parse JSON objects from the formatted output
            if current_section in ["entity", "relationship"]:
                # Check if this line starts a JSON object
                if line.startswith('{"'):
                    # single-line JSON object
                    if line.endswith('}'):
                        try:
                            parsed_obj = json.loads(line)
                            if current_section == "entity" and "entity" in parsed_obj:
                                processed_results.append({
                                    "content": parsed_obj,
                                    "type": "entity",
                                    "source_db": db_id,
                                    "source_name": source_name,
                                    "kb_type": "lightrag",
                                    "data_type": "knowledge_graph"
                                })
                            elif current_section == "relationship" and ("entity1" in parsed_obj or "entity2" in parsed_obj):
                                processed_results.append({
                                    "content": parsed_obj,
                                    "type": "relationship",
                                    "source_db": db_id,
                                    "source_name": source_name,
                                    "kb_type": "lightrag",
                                    "data_type": "knowledge_graph"
                                })
                        except json.JSONDecodeError:
                            pass
                        continue
                    in_json = True
                    json_buffer = line
                elif in_json:
                    json_buffer += line
                    # Check if we've completed a JSON object
                    if line.endswith('}'):
                        try:
                            parsed_obj = json.loads(json_buffer)
                            if current_section == "entity" and "entity" in parsed_obj:
                                processed_results.append({
                                    "content": parsed_obj,
                                    "type": "entity",
                                    "source_db": db_id,
                                    "source_name": source_name,
                                    "kb_type": "lightrag",
                                    "data_type": "knowledge_graph"
                                })
                            elif current_section == "relationship" and ("entity1" in parsed_obj or "entity2" in parsed_obj):
                                processed_results.append({
                                    "content": parsed_obj,
                                    "type": "relationship",
                                    "source_db": db_id,
                                    "source_name": source_name,
                                    "kb_type": "lightrag",
                                    "data_type": "knowledge_graph"
                                })
                            json_buffer = ""
                            in_json = False
                        except json.JSONDecodeError:
                            # Continue building the JSON string
                            continue
            
            # Parse document chunks
            elif current_section == "chunks":
                # Robust JSON capture for chunk entries (single-line or multi-line)
                if line.startswith('{'):
                    # Try single-line first
                    if line.endswith('}'):
                        try:
                            parsed_obj = json.loads(line)
                            processed_results.append({
                                "content": parsed_obj,
                                "type": "chunk",
                                "source_db": db_id,
                                "source_name": source_name,
                                "kb_type": "lightrag",
                                "data_type": "document_chunk"
                            })
                        except json.JSONDecodeError:
                            pass
                        continue
                    # Start buffering multi-line JSON
                    in_json = True
                    json_buffer = line
                elif in_json:
                    json_buffer += line
                    if line.endswith('}'):
                        try:
                            parsed_obj = json.loads(json_buffer)
                            processed_results.append({
                                "content": parsed_obj,
                                "type": "chunk",
                                "source_db": db_id,
                                "source_name": source_name,
                                "kb_type": "lightrag",
                                "data_type": "document_chunk"
                            })
                            json_buffer = ""
                            in_json = False
                        except json.JSONDecodeError:
                            continue
        
        # If we found parsed results, return them
        if processed_results:
            return processed_results
        
        # Fallback: parse as formatted text (original logic)
        lines = kg_text.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect section headers
            if any(header in line.lower() for header in ["entity", "实体"]):
                current_section = "entity"
                continue
            elif any(header in line.lower() for header in ["relationship", "关系"]):
                current_section = "relationship"
                continue
            
            # Parse entity/relationship content
            if current_section and (":" in line or "->" in line):
                if current_section == "entity":
                    # Parse entity format: "Name: Type - Description"
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        entity_name = parts[0].strip()
                        entity_info = parts[1].strip()
                        
                        # Extract type and description
                        if " - " in entity_info:
                            type_desc = entity_info.split(" - ", 1)
                            entity_type = type_desc[0].strip()
                            description = type_desc[1].strip() if len(type_desc) > 1 else ""
                        else:
                            entity_type = "Unknown"
                            description = entity_info
                        
                        processed_results.append({
                            "content": {
                                "entity": entity_name,
                                "type": entity_type,
                                "description": description
                            },
                            "rendered_text": f"Entity: {entity_name}" + (f" (Type: {entity_type})" if entity_type else "") + (f" - {description}" if description else ""),
                            "type": "entity",
                            "source_db": db_id,
                            "source_name": source_name,
                            "kb_type": "lightrag",
                            "data_type": "knowledge_graph"
                        })
                
                elif current_section == "relationship":
                    # Parse relationship format: "Entity1 -> Entity2: Description"
                    if "->" in line:
                        parts = line.split("->", 1)
                        if len(parts) == 2:
                            entity1 = parts[0].strip()
                            remaining = parts[1].strip()
                            
                            # Extract entity2 and description
                            if ":" in remaining:
                                entity2_desc = remaining.split(":", 1)
                                entity2 = entity2_desc[0].strip()
                                description = entity2_desc[1].strip() if len(entity2_desc) > 1 else ""
                            else:
                                entity2 = remaining
                                description = ""
                            
                            processed_results.append({
                                "content": {
                                    "entity1": entity1,
                                    "entity2": entity2,
                                    "description": description
                                },
                                "rendered_text": f"Relationship: {entity1} -> {entity2}" + (f" - {description}" if description else ""),
                                "type": "relationship",
                                "source_db": db_id,
                                "source_name": source_name,
                                "kb_type": "lightrag",
                                "data_type": "knowledge_graph"
                            })
        
        # If no structured data was found, treat as plain text
        if not processed_results:
            processed_results.append({
                "content": kg_text,
                "type": "text",
                "source_db": db_id,
                "source_name": source_name,
                "kb_type": "lightrag",
                "data_type": "text"
            })
        
        return processed_results
    
    def _score_entity_relevance(self, entity_data: Dict[str, Any], query: str) -> float:
        """Score entity relevance for knowledge graph data with enhanced scoring"""
        score = 0.0
        query_lower = query.lower()
        
        # Entity name matching (highest weight)
        entity_name = entity_data.get("entity", "")
        entity_name_lower = entity_name.lower()
        
        # Exact match gets highest score
        if query_lower == entity_name_lower:
            score += 2.0
        # Partial match at beginning
        elif entity_name_lower.startswith(query_lower):
            score += 1.5
        # Contains query
        elif query_lower in entity_name_lower:
            score += 1.0
        
        # Type matching (medium weight)
        entity_type = entity_data.get("type", "")
        if query_lower in entity_type.lower():
            score += 0.3
        
        # Description content matching (lower weight but cumulative)
        description = entity_data.get("description", "")
        if "<SEP>" in description:
            descriptions = description.split("<SEP>")
            desc_matches = 0
            for desc in descriptions:
                if query_lower in desc.lower():
                    desc_matches += 1
            # Cap at 0.5 for description matches
            score += min(0.1 * desc_matches, 0.5)
        else:
            if query_lower in description.lower():
                score += 0.2
        
        # Bonus for high-quality content
        if len(entity_name) > 2 and len(description) > 20:
            score += 0.1
        
        return score
    
    def _calculate_kg_weight(self, query_texts: List[str], content: Dict[str, Any], data_type: str) -> float:
        """Calculate knowledge graph weight based on query intent and data type"""
        if not query_texts:
            return 0.3  # 默认权重
        
        # 合并所有查询文本进行分析
        combined_query = " ".join(query_texts).lower()
        
        # 定义知识图谱友好的关键词
        kg_friendly_keywords = {
            "entity": ["实体", "概念", "定义", "是什么", "介绍", "entity", "concept", "definition"],
            "relationship": ["关系", "联系", "关联", "连接", "之间", "relationship", "relation", "connection"],
            "graph": ["图谱", "图", "网络", "结构", "graph", "network", "structure"]
        }
        
        # 分析查询意图
        kg_intent_score = 0.0
        for category, keywords in kg_friendly_keywords.items():
            for keyword in keywords:
                if keyword in combined_query:
                    kg_intent_score += 0.2
        
        # 基于数据类型的权重调整
        if data_type == "entity":
            base_weight = 0.4
        elif data_type == "relationship":
            base_weight = 0.5
        else:
            base_weight = 0.3
        
        # 最终权重 = 基础权重 + 查询意图分数（最高不超过0.8）
        final_weight = min(base_weight + kg_intent_score, 0.8)
        
        # 根据内容质量微调
        if data_type == "entity":
            entity_name = content.get("entity", "")
            description = content.get("description", "")
            if len(entity_name) > 2 and len(description) > 10:
                final_weight += 0.1  # 内容质量好的增加权重
        
        return min(final_weight, 0.9)  # 确保不超过0.9
    
    def _merge_mixed_results(self, text_results: List[Dict[str, Any]], kg_results: List[Dict[str, Any]], 
                           query_texts: List[str] = None, kg_weight: float = None, 
                           diversity_boost: bool = True) -> List[Dict[str, Any]]:
        """Merge and balance text and knowledge graph results with knowledge base type awareness"""
        
        # Enhanced score normalization that considers knowledge base type
        def normalize_scores(results: List[Dict[str, Any]], score_key: str = "relevance_score") -> List[Dict[str, Any]]:
            if not results:
                return results
            
            scores = [r.get(score_key, 0) for r in results]
            if not scores or max(scores) == min(scores):
                # 如果没有分数或分数相同，赋予默认分数（考虑知识库类型）
                for i, result in enumerate(results):
                    kb_type = result.get("kb_type", "unknown")
                    # Different base scores for different KB types
                    if kb_type == "chroma":
                        base_score = 0.6
                    elif kb_type == "milvus":
                        base_score = 0.65
                    elif kb_type == "lightrag":
                        base_score = 0.7  # Knowledge graph gets slightly higher base
                    else:
                        base_score = 0.5
                    
                    result[score_key] = base_score + (i * 0.05)  # 简单的递减分数
                return results
            
            # 归一化到0-1范围，考虑知识库类型权重
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score if max_score != min_score else 1
            
            for result in results:
                original_score = result.get(score_key, 0)
                normalized_score = (original_score - min_score) / score_range
                
                # Apply knowledge base type boost
                kb_type = result.get("kb_type", "unknown")
                if kb_type == "lightrag":
                    normalized_score *= 1.1  # Slight boost for knowledge graph data
                elif kb_type == "milvus":
                    normalized_score *= 1.05  # Small boost for Milvus
                
                result[score_key] = max(0.1, min(1.0, normalized_score))  # 确保在合理范围内
            
            return results
        
        # 分别归一化两类结果的分数
        text_results = normalize_scores(text_results, "relevance_score")
        kg_results = normalize_scores(kg_results, "relevance_score")
        
        # 为知识图谱结果添加额外的专业分数权重
        if kg_weight is None:
            # 自动计算权重，考虑查询意图和知识库类型
            for result in kg_results:
                content = result.get("content", {})
                kb_type = result.get("kb_type", "unknown")
                if isinstance(content, dict):
                    original_query = result.get("original_query", "")
                    
                    # Base KG weight calculation
                    if "entity" in content:
                        entity_score = self._score_entity_relevance(content, original_query)
                        auto_kg_weight = self._calculate_kg_weight(query_texts, content, "entity")
                        
                        # Adjust weight based on knowledge base type
                        if kb_type == "lightrag":
                            auto_kg_weight *= 1.2  # Boost for LightRAG entities
                        
                        result["relevance_score"] = result.get("relevance_score", 0) * (1 - auto_kg_weight) + entity_score * auto_kg_weight
                        
                    elif "entity1" in content:
                        rel_score = self._score_relationship_relevance(content, original_query)
                        auto_kg_weight = self._calculate_kg_weight(query_texts, content, "relationship")
                        
                        # Adjust weight based on knowledge base type
                        if kb_type == "lightrag":
                            auto_kg_weight *= 1.15  # Boost for LightRAG relationships
                        
                        result["relevance_score"] = result.get("relevance_score", 0) * (1 - auto_kg_weight) + rel_score * auto_kg_weight
        else:
            # 使用用户指定的权重
            for result in kg_results:
                content = result.get("content", {})
                if isinstance(content, dict):
                    original_query = result.get("original_query", "")
                    if "entity" in content:
                        entity_score = self._score_entity_relevance(content, original_query)
                        result["relevance_score"] = result.get("relevance_score", 0) * (1 - kg_weight) + entity_score * kg_weight
                    elif "entity1" in content:
                        rel_score = self._score_relationship_relevance(content, original_query)
                        result["relevance_score"] = result.get("relevance_score", 0) * (1 - kg_weight) + rel_score * kg_weight
        
        # 合并所有结果
        all_results = text_results + kg_results
        
        # 按分数排序
        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # 应用多样性策略：避免同类结果过度集中，考虑知识库类型
        if diversity_boost:
            final_results = []
            max_consecutive_same_type = 1
            max_consecutive_same_kb = 3    # 最多连续3个同知识库结果
            
            for result in all_results:
                data_type = result.get("data_type", "text")
                kb_type = result.get("kb_type", "unknown")
                
                # 检查是否需要强制插入不同类型结果以保持多样性
                if len(final_results) >= max_consecutive_same_type:
                    last_types = [r.get("data_type", "text") for r in final_results[-max_consecutive_same_type:]]
                    if len(set(last_types)) == 1:  # 最近都是同类型
                        # 寻找不同类型的高分结果
                        opposite_type = "knowledge_graph" if data_type == "text" else "text"
                        opposite_candidates = [r for r in all_results[len(final_results):] 
                                               if r.get("data_type", "text") == opposite_type]
                        
                        if opposite_candidates:
                            # 插入一个相反类型的最佳结果
                            best_opposite = max(opposite_candidates, key=lambda x: x.get("relevance_score", 0))
                            final_results.append(best_opposite)
                            all_results.remove(best_opposite)
                
                # 检查知识库类型多样性
                if len(final_results) >= max_consecutive_same_kb:
                    last_kb_types = [r.get("kb_type", "unknown") for r in final_results[-max_consecutive_same_kb:]]
                    if len(set(last_kb_types)) == 1:  # 最近都是同知识库
                        # 寻找不同知识库的高分结果
                        current_kb_type = kb_type
                        different_kb_candidates = [r for r in all_results[len(final_results):] 
                                                 if r.get("kb_type", "unknown") != current_kb_type]
                        
                        if different_kb_candidates:
                            # 插入一个不同知识库的最佳结果
                            best_different_kb = max(different_kb_candidates, key=lambda x: x.get("relevance_score", 0))
                            final_results.append(best_different_kb)
                            all_results.remove(best_different_kb)
                
                final_results.append(result)
                
                # 限制总数量，保持平衡
                if len(final_results) >= len(all_results):
                    break
            
            all_results = final_results
        
        # 添加混合结果标识
        for result in all_results:
            content = result.get("content", {})
            if isinstance(content, dict) and ("entity" in content or "entity1" in content):
                result["data_type"] = "knowledge_graph"
            else:
                result["data_type"] = "text"
        
        # 添加统计信息
        kb_type_stats = {}
        for result in all_results:
            kb_type = result.get("kb_type", "unknown")
            kb_type_stats[kb_type] = kb_type_stats.get(kb_type, 0) + 1
        
        logger.info(f"Merged {len(text_results)} text results and {len(kg_results)} KG results. KB types: {kb_type_stats}")
        
        return all_results
    
    async def _rerank_text_results(self, results: List[Dict[str, Any]], rerank_model: str) -> List[Dict[str, Any]]:
        """Apply specialized text reranking to results based on knowledge base type"""
        try:
            from src.models.rerank import get_reranker
            reranker = get_reranker(rerank_model)
            
            # Prepare reranking pairs: (query, document) for each result
            rerank_pairs = []
            for result in results:
                query = result.get("original_query", "")
                content = result.get("content", "")
                kb_type = result.get("kb_type", "unknown")
                
                # Extract text content based on knowledge base type
                if kb_type == "chroma":
                    # ChromaDB returns chunks with content and metadata
                    if isinstance(content, dict):
                        doc_text = content.get("content", "")
                        metadata = content.get("metadata", {})
                        # Include metadata context for better reranking
                        if metadata:
                            doc_text = f"{doc_text}\n[Metadata: {json.dumps(metadata, ensure_ascii=False)}]"
                    else:
                        doc_text = str(content)
                elif kb_type == "milvus":
                    # Milvus returns structured content with metadata
                    if isinstance(content, dict):
                        doc_text = content.get("content", "")
                        metadata = content.get("metadata", {})
                        # Include source and other metadata
                        if metadata.get("source"):
                            doc_text = f"Source: {metadata['source']}\n{doc_text}"
                    else:
                        doc_text = str(content)
                else:
                    # Default text extraction
                    if isinstance(content, dict):
                        if "reference_id" in content and "content" in content:
                            doc_text = content.get("content", "")
                        else:
                            doc_text = json.dumps(content, ensure_ascii=False)
                    elif not isinstance(content, str):
                        doc_text = str(content)
                    else:
                        doc_text = content
                
                rerank_pairs.append([query, doc_text])
            
            # Get relevance scores
            if rerank_pairs:
                scores = await reranker.acompute_score(rerank_pairs, normalize=True)
                
                # Add scores to results and sort by relevance
                for idx, (result, score) in enumerate(zip(results, scores)):
                    result["relevance_score"] = score
                    
                    # Preserve original scores for different knowledge base types
                    kb_type = result.get("kb_type", "unknown")
                    if kb_type == "chroma":
                        # Preserve original ChromaDB similarity score
                        if isinstance(result.get("content"), dict):
                            original_score = result["content"].get("score", 0.0)
                            result["original_chroma_score"] = original_score
                    elif kb_type == "milvus":
                        # Preserve original Milvus score
                        if isinstance(result.get("content"), dict):
                            original_score = result["content"].get("score", 0.0)
                            result["original_milvus_score"] = original_score
                
                # Sort by relevance score (descending)
                results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                
                logger.info(f"Reranked {len(results)} text results with model {rerank_model} (KB types: {set(r.get('kb_type', 'unknown') for r in results)})")
            
            return results
            
        except Exception as e:
            logger.warning(f"Text reranking failed: {e}. Using original results order.")
            return results
    
    async def _rerank_knowledge_graph_results(self, results: List[Dict[str, Any]], rerank_model: str) -> List[Dict[str, Any]]:
        """Apply knowledge graph specific reranking to results with LightRAG optimization"""
        try:
            from src.models.rerank import get_reranker
            reranker = get_reranker(rerank_model)
            
            # Prepare reranking pairs: (query, document) for each result
            rerank_pairs = []
            for result in results:
                query = result.get("original_query", "")
                content = result.get("content", {})
                
                if isinstance(content, dict):
                    # Check if this is knowledge graph data
                    if "entity" in content:
                        # Entity data - enhanced format for LightRAG
                        entity_name = content.get('entity', '')
                        entity_type = content.get('type', '')
                        description = content.get('description', '')
                        
                        # Create comprehensive entity description for reranking
                        content = f"Entity: {entity_name}"
                        if entity_type:
                            content += f" (Type: {entity_type})"
                        if description:
                            content += f" - {description}"
                            
                    elif "entity1" in content and "entity2" in content:
                        # Relationship data - enhanced format for LightRAG
                        entity1 = content.get('entity1', '')
                        entity2 = content.get('entity2', '')
                        description = content.get('description', '')
                        
                        # Create comprehensive relationship description
                        content = f"Relationship: {entity1} -> {entity2}"
                        if description:
                            content += f" - {description}"
                            
                    else:
                        # Regular dict data
                        content = json.dumps(content, ensure_ascii=False)
                elif not isinstance(content, str):
                    content = str(content)
                
                rerank_pairs.append([query, content])
            
            # Get relevance scores
            if rerank_pairs:
                scores = await reranker.acompute_score(rerank_pairs, normalize=True)
                
                # Add scores to results and sort by relevance
                for idx, (result, score) in enumerate(zip(results, scores)):
                    result["relevance_score"] = score
                    
                    # For knowledge graph data, also compute specialized scores
                    content = result.get("content", {})
                    if isinstance(content, dict):
                        original_query = result.get("original_query", "")
                        if "entity" in content:
                            # Entity data - add specialized entity relevance score
                            entity_score = self._score_entity_relevance(content, original_query)
                            result["entity_relevance_score"] = entity_score
                            
                            # Combine rerank score with entity score for final ranking
                            combined_score = score * 0.7 + entity_score * 0.3
                            result["combined_relevance_score"] = combined_score
                            
                        elif "entity1" in content and "entity2" in content:
                            # Relationship data - add specialized relationship relevance score
                            rel_score = self._score_relationship_relevance(content, original_query)
                            result["relationship_relevance_score"] = rel_score
                            
                            # Combine rerank score with relationship score
                            combined_score = score * 0.6 + rel_score * 0.4
                            result["combined_relevance_score"] = combined_score
                
                # Sort by combined relevance score if available, otherwise by rerank score
                results.sort(key=lambda x: x.get("combined_relevance_score", x.get("relevance_score", 0)), reverse=True)
                
                logger.info(f"Reranked {len(results)} knowledge graph results with model {rerank_model}")
            
            return results
            
        except Exception as e:
            logger.warning(f"Knowledge graph reranking failed: {e}. Using original results order.")
            return results
    
    def _score_relationship_relevance(self, rel_data: Dict[str, Any], query: str) -> float:
        """Score relationship relevance for knowledge graph data with enhanced scoring"""
        score = 0.0
        query_lower = query.lower()
        
        # Entity1 and entity2 name matching (higher weight for exact matches)
        for entity_field in ["entity1", "entity2"]:
            entity_name = rel_data.get(entity_field, "")
            entity_name_lower = entity_name.lower()
            
            # Exact match gets highest score
            if query_lower == entity_name_lower:
                score += 1.5
            # Partial match at beginning
            elif entity_name_lower.startswith(query_lower):
                score += 1.0
            # Contains query
            elif query_lower in entity_name_lower:
                score += 0.6
        
        # Relationship description matching
        description = rel_data.get("description", "")
        if "<SEP>" in description:
            descriptions = description.split("<SEP>")
            desc_matches = 0
            for desc in descriptions:
                if query_lower in desc.lower():
                    desc_matches += 1
            # Cap at 0.8 for description matches
            score += min(0.15 * desc_matches, 0.8)
        else:
            if query_lower in description.lower():
                score += 0.5
        
        # Bonus for high-quality relationships
        if len(description) > 10:
            score += 0.1
        
        return score
    
    def _process_query_result(self, result: Any, db_id: str, source_name: str) -> List[Dict[str, Any]]:
        """Process query result and add metadata with knowledge base type detection"""
        processed_results = []
        
        if result is None or result == "":
            return processed_results
        
        # Detect knowledge base type from source name or db_id
        kb_type = self._detect_kb_type(db_id, source_name)
        
        if isinstance(result, str):
            if result.strip() == "":
                return processed_results
            else:
                # Handle LightRAG knowledge graph output
                if kb_type == "lightrag" and self._is_knowledge_graph_output(result):
                    return self._process_lightrag_kg_output(result, db_id, source_name)
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
                            "source_name": source_name,
                            "kb_type": kb_type
                        })
                        return processed_results
        
        # Handle list results (ChromaDB/Milvus vector results)
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    item["source_db"] = db_id
                    item["source_name"] = source_name
                    item["kb_type"] = kb_type
                    # Add data type based on content structure
                    item["data_type"] = self._detect_data_type(item)
                    processed_results.append(item)
                else:
                    processed_results.append({
                        "content": str(item),
                        "type": "text",
                        "source_db": db_id,
                        "source_name": source_name,
                        "kb_type": kb_type,
                        "data_type": "text"
                    })
        # Handle single dict result
        elif isinstance(result, dict):
            # If LightRAG packed output as {"content": "formatted text"}, parse it
            if kb_type == "lightrag":
                content_val = result.get("content")
                if isinstance(content_val, str) and self._is_knowledge_graph_output(content_val):
                    return self._process_lightrag_kg_output(content_val, db_id, source_name)
            result["source_db"] = db_id
            result["source_name"] = source_name
            result["kb_type"] = kb_type
            result["data_type"] = self._detect_data_type(result)
            processed_results.append(result)
        else:
            # Handle other types
            processed_results.append({
                "content": str(result),
                "type": "text",
                "source_db": db_id,
                "source_name": source_name,
                "kb_type": kb_type,
                "data_type": "text"
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
            
            response_text = f"可用知识库（共 {len(db_list)} 个）：\n\n"
            for db in db_list:
                response_text += f"• {db['name']}（ID: {db['id']}）\n"
                response_text += f"  描述: {db['description']}\n"
                if db['metadata']:
                    response_text += f"  元数据: {json.dumps(db['metadata'], ensure_ascii=False, indent=2)}\n"
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
