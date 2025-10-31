from fastapi import APIRouter

from server.routers.auth_router import auth
from server.routers.chat_router import chat
from server.routers.dashboard_router import dashboard
from server.routers.graph_router import graph
from server.routers.knowledge_router import knowledge
from server.routers.mcp_router import mcp
from server.routers.system_router import system
from server.routers.task_router import tasks

router = APIRouter()

# 注册路由结构
router.include_router(system)  # /api/system/*
router.include_router(auth)  # /api/auth/*
router.include_router(chat)  # /api/chat/*
router.include_router(dashboard)  # /api/dashboard/*
router.include_router(knowledge)  # /api/knowledge/*
router.include_router(graph)  # /api/graph/*
router.include_router(tasks)  # /api/tasks/*
router.include_router(mcp)  # /api/mcp/*
