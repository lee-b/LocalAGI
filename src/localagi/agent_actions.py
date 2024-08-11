### Agent capabilities
### These functions are called by the agent to perform actions
###

import json
import os
import uuid
from logging import getLogger
from copy import copy
from dataclasses import dataclass
from typing import Dict, List

from duckduckgo_search import DDGS

from .agent_action_context import AgentActionContext


logger = getLogger(__name__)


def save(ctx: AgentActionContext, memory, agent_actions={}, localagi=None):
    q = json.loads(memory)

    logger.info(">>> saving to memories: ") 
    logger.info(q["content"])

    ctx.vector_db.add_texts([q["content"]],[{"id": str(uuid.uuid4())}])
    ctx.vector_db.persist()

    return f"The object was saved permanently to memory."


def search_memory(ctx: AgentActionContext, query, agent_actions={}, localagi=None):
    q = json.loads(query)
    docs = ctx.vector_db.similarity_search(q["reasoning"])
    text_res="Memories found in the database:\n"
    for doc in docs:
        text_res+="- "+doc.page_content+"\n"

    #if args.postprocess:
    #    return post_process(text_res)
    #return text_res
    return localagi.post_process(text_res)


# write file to disk with content
def save_file(ctx: AgentActionContext, arg, agent_actions={}, localagi=None):
    arg = json.loads(arg)
    filename = arg["filename"]
    content = arg["content"]
    # create persistent dir if does not exist
    if not os.path.exists(ctx.config.PERSISTENT_DIR):
        os.makedirs(ctx.config.PERSISTENT_DIR)
    # write the file in the directory specified
    filename = os.path.join(ctx.config.PERSISTENT_DIR, filename)
    with open(filename, 'w') as f:
        f.write(content)
    return f"File {filename} saved successfully."


def ddg(ctx: AgentActionContext, query: str, num_results: int, backend: str = "api") -> List[Dict[str, str]]:
    """Run query through DuckDuckGo and return metadata.

    Args:
        query: The query to search for.
        num_results: The number of results to return.

    Returns:
        A list of dictionaries with the following keys:
            snippet - The description of the result.
            title - The title of the result.
            link - The link to the result.
    """

    with ctx.config.DDGS() as ddgs:
        results = ddgs.text(
            query,
            backend=backend,
        )
        if results is None:
            return [{"Result": "No good DuckDuckGo Search Result was found"}]

        def to_metadata(result: Dict) -> Dict[str, str]:
            if backend == "news":
                return {
                    "date": result["date"],
                    "title": result["title"],
                    "snippet": result["body"],
                    "source": result["source"],
                    "link": result["url"],
                }
            return {
                "snippet": result["body"],
                "title": result["title"],
                "link": result["href"],
            }

        formatted_results = []
        for i, res in enumerate(results, 1):
            if res is not None:
                formatted_results.append(to_metadata(res))
            if len(formatted_results) == num_results:
                break
    return formatted_results


## Search on duckduckgo
def search_duckduckgo(ctx: AgentActionContext, args, a, agent_actions={}, localagi=None):
    a = json.loads(a)
    list=ddg(a["query"], args.search_results)

    text_res=""   
    for doc in list:
        text_res+=f"""{doc["link"]}: {doc["title"]} {doc["snippet"]}\n"""  

    #if args.postprocess:
    #    return post_process(text_res)
    return text_res
    #l = json.dumps(list)
    #return l

### End Agent capabilities
###

raw_agent_actions = {
    "search_internet": {
        "function": search_duckduckgo,
        "plannable": True,
        "description": 'For searching the internet with a query, the assistant replies with the action "search_internet" and the query to search.',
        "signature": {
            "name": "search_internet",
            "description": """For searching internet.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "information to save"
                    },
                },
            }
        },
    },
    "save_file": {
        "function": save_file,
        "plannable": True,
        "description": 'The assistant replies with the action "save_file", the filename and content to save for writing a file to disk permanently. This can be used to store the result of complex actions locally.',
        "signature": {
            "name": "save_file",
            "description": """For saving a file to disk with content.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "information to save"
                    },
                    "content": {
                        "type": "string",
                        "description": "information to save"
                    },
                },
            }
        },
    },
    "save_memory": {
        "function": save,
        "plannable": True,
        "description": 'The assistant replies with the action "save_memory" and the string to remember or store an information that thinks it is relevant permanently.',
        "signature": {
            "name": "save_memory",
            "description": """Save or store informations into memory.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "information to save"
                    },
                },
                "required": ["content"]
            }
        },
    },
    "search_memory": {
        "function": search_memory,
        "plannable": True,
        "description": 'The assistant replies with the action "search_memory" for searching between its memories with a query term.',
        "signature": {
            "name": "search_memory",
            "description": """Search in memory""",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "reasoning behind the intent"
                    },
                },
                "required": ["reasoning"]
            }
        }, 
    },
}


def inject_context_to_agent(ctx: AgentActionContext, agent):
    curried_agent = copy(agent)
    curried_agent["function"] = lambda *varargs, **kwargs: agent["function"](ctx, *varargs, **kwargs)
    return curried_agent


def get_agent_actions(ctx: AgentActionContext):
    ### Agent action definitions
    global raw_agent_actions
    curried_agent_actions = [ inject_context_to_agent(ctx, aa) for aa in raw_agent_actions ]
    return curried_agent_actions


__all__ = ['get_agent_actions']
