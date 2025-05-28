from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    FilterableField,
    QdrantSettings,
    ToolSettings,
)

qdrant_settings = QdrantSettings(
    filterable_fields=[
        FilterableField(
            name="color",
            field_type="keyword",
            condition="==",
            description="The color of the object",
        ),
    ],
)

mcp = QdrantMCPServer(
    tool_settings=ToolSettings(),
    qdrant_settings=qdrant_settings,
    embedding_provider_settings=EmbeddingProviderSettings(),
)
