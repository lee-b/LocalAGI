import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

from ascii_magic import AsciiArt
from loguru import logger


from .agent_action_context import AgentActionContext
from .agent_actions import get_agent_actions
from .config import Config
from .constants import Constants
from .localagi import LocalAGI
from .vector_db import build_chroma_db_client


logger = logging.getLogger(__name__)


def my_filter(record):
    return record["level"].no >= logger.level(LOG_LEVEL).no

# Function to create images with LocalAI
def display_avatar(cfg, agi, input_text=None, model=None):
    input_text |= cfg.STABLEDIFFUSION_PROMPT
    model |= cfg.STABLEDIFFUSION_MODEL

    image_url = agi.get_avatar(input_text, model)
    # convert the image to ascii art
    my_art = AsciiArt.from_url(image_url)
    my_art.to_terminal()


## This function is called to ask the user if does agree on the action to take and execute
def ask_user_confirmation(action_name, action_parameters):
    logger.info("==> Ask user confirmation")
    logger.info("==> action_name: {action_name}", action_name=action_name)
    logger.info("==> action_parameters: {action_parameters}", action_parameters=action_parameters)
    # Ask via stdin
    logger.info("==> Do you want to execute the action? (y/n)")
    user_input = input()
    if user_input == "y":
        logger.info("==> Executing action")
        return True
    else:
        logger.info("==> Skipping action")
        return False


def parse_args(args):
    parser = argparse.ArgumentParser(description='LocalAGI')

    # System prompt
    parser.add_argument('--system-prompt', dest='system_prompt', action='store',
                        help='System prompt to use')
    # Batch mode
    parser.add_argument('--prompt', dest='prompt', action='store', default=False,
                        help='Prompt mode')
    # Interactive mode
    parser.add_argument('--interactive', dest='interactive', action='store_true', default=False,
                        help='Interactive mode. Can be used with --prompt to start an interactive session')
    # skip avatar creation
    parser.add_argument('--skip-avatar', dest='skip_avatar', action='store_true', default=False,
                        help='Skip avatar creation') 
    # Reevaluate
    parser.add_argument('--re-evaluate', dest='re_evaluate', action='store_true', default=False,
                        help='Reevaluate if another action is needed or we have completed the user request')
    # Postprocess
    parser.add_argument('--postprocess', dest='postprocess', action='store_true', default=False,
                        help='Postprocess the reasoning')
    # Subtask context
    parser.add_argument('--subtask-context', dest='subtaskContext', action='store_true', default=False,
                        help='Include context in subtasks')

    # Search results number
    parser.add_argument('--search-results', dest='search_results', type=int, action='store', default=2,
                        help='Number of search results to return')
    # Plan message
    parser.add_argument('--plan-message', dest='plan_message', action='store', 
                        help="What message to use during planning",
    )                   

    # TTS api base
    parser.add_argument('--tts-api-base', dest='tts_api_base', action='store', default=DEFAULT_API_BASE,
                        help='TTS api base')
    # LocalAI api base
    parser.add_argument('--localai-api-base', dest='localai_api_base', action='store', default=DEFAULT_API_BASE,
                        help='LocalAI api base')
    # Images api base
    parser.add_argument('--images-api-base', dest='images_api_base', action='store', default=DEFAULT_API_BASE,
                        help='Images api base')
    # Embeddings api base
    parser.add_argument('--embeddings-api-base', dest='embeddings_api_base', action='store', default=DEFAULT_API_BASE,
                        help='Embeddings api base')
    # Functions model
    parser.add_argument('--functions-model', dest='functions_model', action='store', default="functions",
                        help='Functions model')
    # Embeddings model
    parser.add_argument('--embeddings-model', dest='embeddings_model', action='store', default="all-MiniLM-L6-v2",
                        help='Embeddings model')
    # LLM model
    parser.add_argument('--llm-model', dest='llm_model', action='store', default="gpt-4",
                        help='LLM model')
    # Voice model
    parser.add_argument('--tts-model', dest='tts_model', action='store', default="en-us-kathleen-low.onnx",
                        help='TTS model')
    # Stable diffusion model
    parser.add_argument('--stablediffusion-model', dest='stablediffusion_model', action='store', default="stablediffusion",
                        help='Stable diffusion model')
    # Stable diffusion prompt
    parser.add_argument('--stablediffusion-prompt', dest='stablediffusion_prompt', action='store', default=DEFAULT_PROMPT,
                        help='Stable diffusion prompt')
    # Force action
    parser.add_argument('--force-action', dest='force_action', action='store', default="",
                        help='Force an action')
    # Debug mode
    parser.add_argument('--debug', dest='debug', action='store_true', default=False,
                        help='Debug mode')
    # Critic mode
    parser.add_argument('--critic', dest='critic', action='store_true', default=False,
                        help='Enable critic')

    # Parse arguments
    args = parser.parse_args()

    return args


def init_logging(args):
    global LOG_LEVEL

    # Set log level
    LOG_LEVEL = "INFO"

    logger.remove()
    logger.add(sys.stderr, filter=my_filter)

    if args.debug:
        LOG_LEVEL = "DEBUG"

    logger.debug("Debug mode on")


def run_agent(cfg: Config, agent_actions):
    conversation_history = []

    # Create a LocalAGI instance
    logger.info("Creating LocalAGI instance")

    localagi = LocalAGI(
        agent_actions=agent_actions,
        llm_model=cfg.LLM_MODEL,
        tts_model=cfg.VOICE_MODEL,
        tts_api_base=cfg.TTS_API_BASE,
        functions_model=cfg.FUNCTIONS_MODEL,
        api_base=cfg.LOCALAI_API_BASE,
        stablediffusion_api_base=cfg.IMAGE_API_BASE,
        stablediffusion_model=cfg.STABLEDIFFUSION_MODEL,
        force_action=cfg.args.force_action,
        plan_message=cfg.args.plan_message,
    )

    # Set a system prompt if SYSTEM_PROMPT is set
    if cfg.SYSTEM_PROMPT != "":
        conversation_history.append({
            "role": "system",
            "content": cfg.SYSTEM_PROMPT,
        })

    logger.info("Welcome to LocalAGI")

    # Skip avatar creation if --skip-avatar is set
    if not cfg.args.skip_avatar:
        logger.info("Creating avatar, please wait...")
        display_avatar(cfg, localagi)

    actions = " ".join(
        (f"'{action}'" for action in agent_actions)
    )

    logger.info("LocalAGI internally can do the following actions:{actions}", actions=actions)

    if not cfg.args.prompt:
        logger.info(">>> Interactive mode <<<")
    else:
        logger.info(">>> Prompt mode <<<")
        logger.info(cfg.args.prompt)

    # IF in prompt mode just evaluate, otherwise loop
    if cfg.args.prompt:
        conversation_history=localagi.evaluate(
            cfg.args.prompt, 
            conversation_history, 
            critic=cfg.args.critic,
            re_evaluate=cfg.args.re_evaluate, 
            # Enable to lower context usage but increases LLM calls
            postprocess=cfg.args.postprocess,
            subtaskContext=cfg.args.subtaskContext,
            )
        localagi.tts_play(conversation_history[-1]["content"])

    if not cfg.args.prompt or cfg.args.interactive:
        # TODO: process functions also considering the conversation history? conversation history + input
        logger.info(">>> Ready! What can I do for you? ( try with: plan a roadtrip to San Francisco ) <<<")

        while True:
            user_input = input(">>> ")
            # we are going to use the args to change the evaluation behavior
            conversation_history=localagi.evaluate(
                user_input,
                conversation_history,
                critic=cfg.args.critic,
                re_evaluate=cfg.args.re_evaluate,
                # Enable to lower context usage but increases LLM calls
                postprocess=cfg.args.postprocess,
                subtaskContext=cfg.args.subtaskContext,
            )
            localagi.tts_play(conversation_history[-1]["content"])


def main():
    args = parse_args(sys.argv)
    init_logging(args)

    config = Config.from_environ_and_args(os.environ, args)

    vector_db = build_chroma_db_client(config)

    agent_action_ctx = AgentActionContext(
        args=args,
        vector_db=vector_db,
        config=config,
    )

    agent_actions = get_agent_actions(agent_action_ctx)

    run_agent(args, agent_actions)
