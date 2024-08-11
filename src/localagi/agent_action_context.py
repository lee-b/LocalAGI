from argparse import Namespace
from dataclasses import dataclass
from langchain.vectorstores import Chroma

from .config import Config


@dataclass
class AgentActionContext:
    args : Namespace = None
    vector_db: Chroma = None
    config : Config = None


__all__ = [ 'AgentActionContext' ]
