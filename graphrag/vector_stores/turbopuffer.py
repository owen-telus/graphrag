
"""TurboPuffer vector storage implementation."""

import os
from typing import Any

import turbopuffer as tpuf

from graphrag.model.types import TextEmbedder
from graphrag.vector_stores.base import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)


class TurboPufferVectorStore(BaseVectorStore):
    """TurboPuffer vector storage implementation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def connect(self, **kwargs: Any) -> Any:
        """Connect to the vector storage."""
        tpuf.api_base_url = kwargs.get("api_base") or os.environ.get(
            "TURBOPUFFER_API_BASE_URL"
        )
        tpuf.api_key = kwargs.get("api_key") or os.environ.get(
            "TURBOPUFFER_API_KEY"
        )

        if not tpuf.api_base_url or not tpuf.api_key:
            error_msg = "TURBOPUFFER_API_BASE_URL and TURBOPUFFER_API_KEY must be set in either kwargs or environment variables"
            raise ValueError(error_msg)
        self.document_collection = tpuf.Namespace(self.collection_name)

    def load_documents(
        self, documents: list[VectorStoreDocument], overwrite: bool = False
    ) -> None:
        """Load documents into vector storage."""
        for document in documents:
            if "text" in document.attributes:
                error_msg = (
                    "text is a reserved attribute name and cannot be used in attributes"
                )
                raise ValueError(error_msg)
        data = [
            tpuf.VectorRow(
                id=document.id,
                vector=document.vector,
                attributes={
                    "text": document.text,
                    **document.attributes,
                },
            )
            for document in documents
            if document.vector is not None
        ]

        if overwrite and self.document_collection.exists():
            self.document_collection.delete_all()
            self.document_collection.upsert(
                data,
                distance_metric="cosine_distance",
            )
        else:
            self.document_collection.upsert(
                data,
                distance_metric="cosine_distance",
            )

    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform a vector-based similarity search."""
        docs = self.document_collection.query(
            vector=query_embedding,
            top_k=k,
            include_attributes=True,
            include_vectors=True,
            filters=self.query_filter,
        )

        results = []
        for doc in docs:
            # Get text from attributes
            text_value = doc.attributes.get("text") if doc.attributes else None
            text = str(text_value) if text_value is not None else None

            # Copy attributes and remove text since it's handled separately
            attributes = doc.attributes.copy() if doc.attributes else {}
            if "text" in attributes:
                del attributes["text"]

            # Create document
            document = VectorStoreDocument(
                id=doc.id,
                text=text,
                vector=doc.vector,
                attributes=attributes,
            )

            # Calculate similarity score from distance
            # cosine_distance is between 0 (identical) and 2 (opposite)
            # Convert to similarity score between 0 and 1
            score = 1 - (doc.dist / 2) if doc.dist is not None else 0

            results.append(VectorStoreSearchResult(document=document, score=score))

        return results

    def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform a similarity search using a given input text."""
        query_embedding = text_embedder(text)
        if query_embedding:
            return self.similarity_search_by_vector(query_embedding, k)
        return []

    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
        """Build a query filter to filter documents by id."""
        if len(include_ids) == 0:
            self.query_filter = None
        else:
            self.query_filter = ("id", "In", include_ids)
        return self.query_filter

    def search_by_id(self, id: str) -> VectorStoreDocument:
        """Search for a document by id."""
        query_filter = ("id", "Eq", id)
        result = self.document_collection.query(
            filters=query_filter,
            top_k=1,  # should only be 1 result for a given id
            include_attributes=True,
            include_vectors=True,
        )

        if result and len(result) > 0:
            row = result[0]
            # text is stored in attributes, ensure it's str or None
            text_value = row.attributes.get("text") if row.attributes else None
            text = str(text_value) if text_value is not None else None
            # Remove text from attributes since it's handled separately
            attributes = row.attributes.copy() if row.attributes else {}
            if "text" in attributes:
                del attributes["text"]
            return VectorStoreDocument(
                id=row.id,
                text=text,
                vector=row.vector,
                attributes=attributes,
            )
        return VectorStoreDocument(id=id, text=None, vector=None)
