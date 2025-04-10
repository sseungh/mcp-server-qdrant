import abc
import json
import logging
from typing import Any, List

from mcp.server.fastmcp import Context, FastMCP

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP, abc.ABC):
    """
    An MCP server for Qdrant using a single collection specified in
    the configuration.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: EmbeddingProviderSettings,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings
        self.embedding_provider_settings = embedding_provider_settings

        self.embedding_provider = create_embedding_provider(embedding_provider_settings)
        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    @abc.abstractmethod
    async def find(self, *args, **kwargs) -> Any:
        """
        An abstract method for finding memories in Qdrant.
        Each implementation should define its own set of parameters,
        so they are visible in the MCP client.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def store(self, *args, **kwargs) -> Any:
        """
        An abstract method for storing memories in Qdrant.
        Each implementation should define its own set of parameters,
        so they are visible in the MCP client.
        """
        raise NotImplementedError

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def setup_tools(self):
        """
        Register the tools in the server.
        """
        self.add_tool(
            self.find,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database
            self.add_tool(
                self.store,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )


class MultiCollectionQdrantMCPServer(QdrantMCPServer):
    """
    An MCP server for Qdrant accepting collection name as a tool parameter,
    so multiple collections can be used for different kinds of memories.
    """

    async def find(
        self,
        ctx: Context,
        query: str,
        collection_name: str,
    ) -> List[str]:
        """
        Find memories in Qdrant.
        :param ctx: The context for the request.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :return: A list of entries found.
        """
        await ctx.debug(f"Finding results for query '{query}' in {collection_name}")

        entries = await self.qdrant_connector.search(
            query,
            collection_name=collection_name,
            limit=self.qdrant_settings.search_limit,
        )
        if not entries:
            return [f"No information found for the query '{query}'"]
        content = [
            f"Results for the query '{query}'",
        ]
        for entry in entries:
            content.append(self.format_entry(entry))
        return content

    async def store(
        self,
        ctx: Context,
        information: str,
        collection_name: str,
        # The `metadata` parameter is defined as non-optional, but it can be None.
        # If we set it to be optional, some of the MCP clients, like Cursor, cannot
        # handle the optional parameter correctly.
        metadata: Metadata = None,  # type: ignore
    ) -> str:
        """
        Store some information in Qdrant.
        :param ctx: The context for the request.
        :param information: The information to store.
        :param metadata: JSON metadata to store with the information, optional.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        :return: A message indicating that the information was stored.
        """
        await ctx.debug(f"Storing information {information} in Qdrant")

        entry = Entry(content=information, metadata=metadata)
        await self.qdrant_connector.store(entry, collection_name=collection_name)
        return f"Remembered: {information} in collection {collection_name}"


class DefaultCollectionQdrantMCPServer(QdrantMCPServer):
    """
    An MCP server for Qdrant using a single collection specified in
    the configuration.
    """

    async def find(
        self,
        ctx: Context,
        query: str,
    ) -> List[str]:
        """
        Find memories in Qdrant using the default collection.
        :param ctx: The context for the request.
        :param query: The query to use for the search.
        :return: A list of entries found.
        """
        await ctx.debug(f"Finding results for query {query}")

        entries = await self.qdrant_connector.search(
            query,
            collection_name=self.qdrant_settings.collection_name,
            limit=self.qdrant_settings.search_limit,
        )
        if not entries:
            return [f"No information found for the query '{query}'"]
        content = [
            f"Results for the query '{query}'",
        ]
        for entry in entries:
            content.append(self.format_entry(entry))
        return content

    async def store(
        self,
        ctx: Context,
        information: str,
        metadata: Metadata = None,  # type: ignore
    ) -> str:
        """
        Store some information in Qdrant in a default collection.
        :param ctx: The context for the request.
        :param information: The information to store.
        :param metadata: JSON metadata to store with the information, optional.
        :return: A message indicating that the information was stored.
        """
        await ctx.debug(f"Storing information {information} in Qdrant")

        entry = Entry(content=information, metadata=metadata)
        await self.qdrant_connector.store(
            entry, collection_name=self.qdrant_settings.collection_name
        )
        return f"Remembered: {information}"


def create_mcp_server(
    tool_settings: ToolSettings,
    qdrant_settings: QdrantSettings,
    embedding_provider_settings: EmbeddingProviderSettings,
) -> QdrantMCPServer:
    """
    Create an instance of the appropriate MCP server based on the configuration.
    """
    if qdrant_settings.collection_name:
        return DefaultCollectionQdrantMCPServer(
            tool_settings,
            qdrant_settings,
            embedding_provider_settings,
        )

    return MultiCollectionQdrantMCPServer(
        tool_settings,
        qdrant_settings,
        embedding_provider_settings,
    )
