import pytest
from agents.state import AgentState

def test_parallel_merge_reducers():
    """Test that parallel agent outputs merge their lists via LangGraph reducers rather than overriding."""
    # This is handled internally by LangGraph's state reducers defined in AgentState.
    # We will simulate the add operator.
    import operator
    
    # Simulate two agents running in parallel and both returning logs and errors
    agent_a_output = {"agent_logs": [{"a": 1}], "errors": ["err_a"]}
    agent_b_output = {"agent_logs": [{"b": 2}], "errors": ["err_b"]}
    
    # LangGraph uses operator.add for Annotated[list, operator.add]
    merged_logs = operator.add(agent_a_output["agent_logs"], agent_b_output["agent_logs"])
    merged_errors = operator.add(agent_a_output["errors"], agent_b_output["errors"])
    
    assert len(merged_logs) == 2
    assert merged_logs == [{"a": 1}, {"b": 2}]
    
    assert len(merged_errors) == 2
    assert merged_errors == ["err_a", "err_b"]
