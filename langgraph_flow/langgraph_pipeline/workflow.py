from langgraph.graph import StateGraph
from langgraph_flow.agents.load_data import load_data, DataState
from langgraph_flow.agents.detect_patterns import detect_patterns
from langgraph_flow.agents.visualize_data import visualize
from langgraph_flow.agents.generate_rapport import generate_report

def build_pipeline():
    builder = StateGraph(DataState)
    builder.add_node("Load Data", load_data)
    builder.add_node("Visualize", visualize)
    builder.add_node("Detect", detect_patterns)
    builder.add_node("generate_report", generate_report)

    builder.set_entry_point("Load Data")
    builder.add_edge("Load Data", "Visualize")
    builder.add_edge("Visualize", "Detect")
    builder.add_edge("Detect", "generate_report")

    return builder.compile()
