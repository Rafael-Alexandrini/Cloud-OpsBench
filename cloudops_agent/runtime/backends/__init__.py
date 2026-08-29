from .react_single_agent import ReactSingleAgentBackend
from .two_agent_demo import TwoAgentBackend

BACKENDS = {
    "react_single_agent": ReactSingleAgentBackend,
    "two_agent_demo": TwoAgentBackend,
}

__all__ = ["ReactSingleAgentBackend", "TwoAgentBackend", "BACKENDS"]
