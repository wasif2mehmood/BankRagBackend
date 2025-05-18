from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, START, END
from utils import (
    ChatMessage, 
    ChatState,
    safety_guardrail_node, 
    memory_check_node, 
    memory_response_node, 
    rag_response_node,
    safety_response_node,
    safety_router,
    memory_router
)

# Build the graph
def create_chat_graph():
    """Create a LangGraph flow for the chat system"""
    # Create a new graph
    builder = StateGraph(ChatState)
    
    # Add nodes
    builder.add_node("safety_check", safety_guardrail_node)
    builder.add_node("memory_check", memory_check_node)
    builder.add_node("memory_response", memory_response_node)
    builder.add_node("rag_response", rag_response_node)
    builder.add_node("safety_response", safety_response_node)
    
    # Set the entry point
    builder.add_edge(START, "safety_check")
    
    # Add conditional edges
    builder.add_conditional_edges(
        "safety_check",
        safety_router,
        {
            "memory_check": "memory_check",
            "safety_response": "safety_response"
        }
    )
    
    builder.add_conditional_edges(
        "memory_check",
        memory_router,
        {
            "memory_response": "memory_response",
            "rag_response": "rag_response"
        }
    )
    
    # Set terminal nodes
    builder.add_edge("memory_response", END)
    builder.add_edge("rag_response", END)
    builder.add_edge("safety_response", END)
    
    return builder

# Export the graph builder but NOT the compiled graph
graph_builder = create_chat_graph()