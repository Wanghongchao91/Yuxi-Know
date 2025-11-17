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
                    description="Query all available knowledge bases with intelligent routing and automatic reranking. Results are automatically optimized for relevance using advanced reranking algorithms.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query_text": {
                                "type": "string",
                                "description": "The query text to search for. Can be a single string or array of strings for batch queries"
                            },
                            "db_id": {
                                "type": "string",
                                "description": "Specific database ID to query (optional). Use list_knowledge_bases to get available db_id values"
                            },
                            "mode": {
                                "type": "string",
                                "description": "Query mode",
                                "enum": ["local", "global", "hybrid", "naive", "mix"]
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
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
                                    "enum": ["local", "global", "hybrid", "naive", "mix"]
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Maximum number of results to return",
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
            # 使用内部默认配置，用户不需要看到这些参数
            enable_rerank = arguments.get("enable_rerank", RerankConfig.DEFAULT_ENABLE_RERANK)
            rerank_model = arguments.get("rerank_model", RerankConfig.DEFAULT_RERANK_MODEL)
            rerank_strategy = arguments.get("rerank_strategy", RerankConfig.DEFAULT_RERANK_STRATEGY)
            kg_weight = arguments.get("kg_weight", RerankConfig.DEFAULT_KG_WEIGHT)
            diversity_boost = arguments.get("diversity_boost", RerankConfig.DEFAULT_DIVERSITY_BOOST)
            
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
                "mixed_data_detected": False  # 将在后续处理中更新
                # 内部参数不暴露给用户：rerank_model, rerank_strategy, kg_weight, diversity_boost
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
                    # Check if this is knowledge graph data and apply specialized reranking
                    if rerank_strategy == "auto":
                        # 分析结果类型分布
                        text_results = []
                        kg_results = []
                        
                        for result in all_results:
                            content = result.get("content", {})
                            if isinstance(content, dict) and ("entity" in content or "entity1" in content):
                                kg_results.append(result)
                            else:
                                text_results.append(result)
                        
                        # 混合重排策略
                        if kg_results and text_results:
                            # 既有知识图谱又有文档数据，采用混合重排
                            logger.info(f"Mixed data detected: {len(kg_results)} KG results, {len(text_results)} text results")
                            query_metadata["mixed_data_detected"] = True
                            
                            # 分别重排两类数据
                            if text_results:
                                text_results = await self._rerank_text_results(text_results, rerank_model)
                            if kg_results:
                                kg_results = await self._rerank_knowledge_graph_results(kg_results, rerank_model)
                            
                            # 合并结果，使用归一化分数确保公平比较
                            all_results = self._merge_mixed_results(text_results, kg_results, query_texts, 
                                                                   kg_weight=kg_weight, diversity_boost=diversity_boost)
                            
                        elif kg_results:
                            # 只有知识图谱数据
                            all_results = await self._rerank_knowledge_graph_results(all_results, rerank_model)
                        else:
                            # 只有文本数据
                            all_results = await self._rerank_text_results(all_results, rerank_model)
                    elif rerank_strategy == "knowledge_graph":
                        # Force knowledge graph reranking
                        all_results = await self._rerank_knowledge_graph_results(all_results, rerank_model)
                    elif rerank_strategy == "mixed":
                        # 强制混合重排，不管数据类型如何
                        query_metadata["mixed_data_detected"] = True  # 标记为混合数据
                        text_results = []
                        kg_results = []
                        
                        for result in all_results:
                            content = result.get("content", {})
                            if isinstance(content, dict) and ("entity" in content or "entity1" in content):
                                kg_results.append(result)
                            else:
                                text_results.append(result)
                        
                        # 分别重排两类数据
                        if text_results:
                            text_results = await self._rerank_text_results(text_results, rerank_model)
                        if kg_results:
                            kg_results = await self._rerank_knowledge_graph_results(kg_results, rerank_model)
                        
                        # 合并结果
                        all_results = self._merge_mixed_results(text_results, kg_results, query_texts, 
                                                               kg_weight=kg_weight, diversity_boost=diversity_boost)
                    else:
                        # Standard text reranking
                        all_results = await self._rerank_text_results(all_results, rerank_model)
                        
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}. Using original results order.")
            
            # Limit results to top_k
            final_results = all_results[:top_k]
            
            # Add metadata to results (简化用户可见的元数据)
            simplified_metadata = {
                "total_queries": query_metadata["total_queries"],
                "databases_queried": query_metadata["databases_queried"],
                "total_results": len(final_results),
                "data_types": list(set(r.get("data_type", "text") for r in final_results)) if final_results else ["text"]
            }
            
            # 只在混合数据时显示额外信息
            if query_metadata.get("mixed_data_detected", False):
                simplified_metadata["mixed_data_detected"] = True
            
            result_data = {
                "results": final_results,
                "metadata": simplified_metadata
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
    
    def _score_entity_relevance(self, entity_data: Dict[str, Any], query: str) -> float:
        """Score entity relevance for knowledge graph data"""
        score = 0.0
        query_lower = query.lower()
        
        # Entity name matching
        entity_name = entity_data.get("entity", "")
        if query_lower in entity_name.lower():
            score += 1.0
        
        # Type matching
        entity_type = entity_data.get("type", "")
        if query_lower in entity_type.lower():
            score += 0.5
        
        # Description content matching
        description = entity_data.get("description", "")
        if "<SEP>" in description:
            descriptions = description.split("<SEP>")
            for desc in descriptions:
                if query_lower in desc.lower():
                    score += 0.3
        else:
            if query_lower in description.lower():
                score += 0.3
        
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
        """Merge and balance text and knowledge graph results"""
        # 确保两类结果都有合理的分数分布
        def normalize_scores(results: List[Dict[str, Any]], score_key: str = "relevance_score") -> List[Dict[str, Any]]:
            if not results:
                return results
            
            scores = [r.get(score_key, 0) for r in results]
            if not scores or max(scores) == min(scores):
                # 如果没有分数或分数相同，赋予默认分数
                for i, result in enumerate(results):
                    result[score_key] = 0.5 + (i * 0.1)  # 简单的递减分数
                return results
            
            # 归一化到0-1范围
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score if max_score != min_score else 1
            
            for result in results:
                original_score = result.get(score_key, 0)
                normalized_score = (original_score - min_score) / score_range
                result[score_key] = max(0.1, normalized_score)  # 确保最小分数不为0
            
            return results
        
        # 分别归一化两类结果的分数
        text_results = normalize_scores(text_results, "relevance_score")
        kg_results = normalize_scores(kg_results, "relevance_score")
        
        # 为知识图谱结果添加额外的专业分数权重
        if kg_weight is None:
            # 自动计算权重
            for result in kg_results:
                content = result.get("content", {})
                if isinstance(content, dict):
                    original_query = result.get("original_query", "")
                    if "entity" in content:
                        entity_score = self._score_entity_relevance(content, original_query)
                        # 实体结果给予额外权重，考虑查询意图
                        auto_kg_weight = self._calculate_kg_weight(query_texts, content, "entity")
                        result["relevance_score"] = result.get("relevance_score", 0) * (1 - auto_kg_weight) + entity_score * auto_kg_weight
                    elif "entity1" in content:
                        rel_score = self._score_relationship_relevance(content, original_query)
                        # 关系结果给予额外权重，考虑查询意图
                        auto_kg_weight = self._calculate_kg_weight(query_texts, content, "relationship")
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
        
        # 应用多样性策略：避免同类结果过度集中
        if diversity_boost:
            final_results = []
            max_consecutive_same_type = 3  # 最多连续3个同类型结果
            
            for result in all_results:
                data_type = result.get("data_type", "text")
                
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
                            # 继续处理当前结果
                
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
        
        logger.info(f"Merged {len(text_results)} text results and {len(kg_results)} KG results")
        
        return all_results
    
    async def _rerank_text_results(self, results: List[Dict[str, Any]], rerank_model: str) -> List[Dict[str, Any]]:
        """Apply standard text reranking to results"""
        try:
            from src.models.rerank import get_reranker
            reranker = get_reranker(rerank_model)
            
            # Prepare reranking pairs: (query, document) for each result
            rerank_pairs = []
            for result in results:
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
                for idx, (result, score) in enumerate(zip(results, scores)):
                    result["relevance_score"] = score
                
                # Sort by relevance score (descending)
                results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                
                logger.info(f"Reranked {len(results)} text results with model {rerank_model}")
            
            return results
            
        except Exception as e:
            logger.warning(f"Text reranking failed: {e}. Using original results order.")
            return results
    
    async def _rerank_knowledge_graph_results(self, results: List[Dict[str, Any]], rerank_model: str) -> List[Dict[str, Any]]:
        """Apply knowledge graph specific reranking to results"""
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
                        # Entity data
                        content = f"Entity: {content.get('entity', '')} ({content.get('type', '')}) - {content.get('description', '')}"
                    elif "entity1" in content:
                        # Relationship data
                        content = f"Relationship: {content.get('entity1', '')} -> {content.get('entity2', '')} - {content.get('description', '')}"
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
                        elif "entity1" in content:
                            # Relationship data - add specialized relationship relevance score
                            rel_score = self._score_relationship_relevance(content, original_query)
                            result["relationship_relevance_score"] = rel_score
                
                # Sort by relevance score (descending)
                results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                
                logger.info(f"Reranked {len(results)} knowledge graph results with model {rerank_model}")
            
            return results
            
        except Exception as e:
            logger.warning(f"Knowledge graph reranking failed: {e}. Using original results order.")
            return results
    
    def _score_relationship_relevance(self, rel_data: Dict[str, Any], query: str) -> float:
        """Score relationship relevance for knowledge graph data"""
        score = 0.0
        query_lower = query.lower()
        
        # Entity1 and entity2 name matching
        for entity_field in ["entity1", "entity2"]:
            entity_name = rel_data.get(entity_field, "")
            if query_lower in entity_name.lower():
                score += 0.8
        
        # Relationship description matching
        description = rel_data.get("description", "")
        if query_lower in description.lower():
            score += 0.6
        
        return score
    
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