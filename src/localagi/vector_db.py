import sys

import pysqlite3
from langchain.embeddings import LocalAIEmbeddings
from langchain.vectorstores import Chroma

from .config import Config


def build_chroma_db_client(config: Config) -> Chroma:
    # these three lines swap the stdlib sqlite3 lib with
    # the pysqlite3 package for chroma

    # TODO: would the following work, less invasively?
    #
    # sqlite3 = sys.modules['sqlite3']
    # sys.modules['sqlite3'] = pysqlite3
    # try:
    #     chroma_client = Chroma(collection_name="memories", persist_directory="db", embedding_function=embeddings)
    # finally:
    #     sys.modules['sqlite3'] = sqlite3

    sys.modules['sqlite3'] = pysqlite3

    embeddings = LocalAIEmbeddings(
        model=config.EMBEDDINGS_MODEL,
        openai_api_base=config.EMBEDDINGS_API_BASE
    )

    chroma_client = Chroma(
        collection_name="memories",
        persist_directory="db",
        embedding_function=embeddings,
    )

    return chroma_client
