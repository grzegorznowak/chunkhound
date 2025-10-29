"""Clustering service for grouping sources in map-reduce synthesis.

Uses K-means clustering on embeddings to group files into token-bounded clusters
for parallel synthesis operations.
"""

import math
from dataclasses import dataclass

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans  # type: ignore[import-not-found]

from chunkhound.interfaces.embedding_provider import EmbeddingProvider
from chunkhound.interfaces.llm_provider import LLMProvider


@dataclass
class ClusterGroup:
    """A cluster of files for synthesis."""

    cluster_id: int
    file_paths: list[str]
    files_content: dict[str, str]  # file_path -> content
    total_tokens: int


class ClusteringService:
    """Service for clustering files into token-bounded groups."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        max_tokens_per_cluster: int = 30000,
    ):
        """Initialize clustering service.

        Args:
            embedding_provider: Provider for generating embeddings
            llm_provider: Provider for token estimation
            max_tokens_per_cluster: Maximum tokens allowed per cluster
        """
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider
        self._max_tokens_per_cluster = max_tokens_per_cluster

    async def cluster_files(
        self, files: dict[str, str]
    ) -> tuple[list[ClusterGroup], dict[str, int]]:
        """Cluster files into token-bounded groups.

        Args:
            files: Dictionary mapping file_path -> file_content

        Returns:
            Tuple of (cluster_groups, metadata) where metadata contains:
                - num_clusters: Number of clusters created
                - total_files: Total number of files
                - total_tokens: Total tokens across all files
                - avg_tokens_per_cluster: Average tokens per cluster

        Raises:
            ValueError: If files dict is empty
        """
        if not files:
            raise ValueError("Cannot cluster empty files dictionary")

        # Calculate total tokens
        total_tokens = sum(
            self._llm_provider.estimate_tokens(content) for content in files.values()
        )

        # Compute optimal k based on token budget
        k = self._compute_optimal_k(total_tokens, len(files))

        logger.info(
            f"Clustering {len(files)} files ({total_tokens:,} tokens) into {k} clusters "
            f"(max {self._max_tokens_per_cluster:,} tokens/cluster)"
        )

        # Special case: single cluster (fallback to single-pass)
        if k == 1:
            logger.info("Only 1 cluster needed - will use single-pass synthesis")
            cluster_group = ClusterGroup(
                cluster_id=0,
                file_paths=list(files.keys()),
                files_content=files,
                total_tokens=total_tokens,
            )
            metadata = {
                "num_clusters": 1,
                "total_files": len(files),
                "total_tokens": total_tokens,
                "avg_tokens_per_cluster": total_tokens,
            }
            return [cluster_group], metadata

        # Generate embeddings for each file
        file_paths = list(files.keys())
        file_contents = [files[fp] for fp in file_paths]

        logger.debug(f"Generating embeddings for {len(file_contents)} files")
        embeddings = await self._embedding_provider.embed(file_contents)

        # Run K-means clustering
        logger.debug(f"Running K-means with k={k}")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(np.array(embeddings))

        # Group files by cluster
        clusters: dict[int, list[str]] = {}
        for file_path, cluster_id in zip(file_paths, labels):
            clusters.setdefault(int(cluster_id), []).append(file_path)

        # Build cluster groups with token counts
        cluster_groups: list[ClusterGroup] = []
        for cluster_id, cluster_file_paths in sorted(clusters.items()):
            cluster_files_content = {fp: files[fp] for fp in cluster_file_paths}
            cluster_tokens = sum(
                self._llm_provider.estimate_tokens(content)
                for content in cluster_files_content.values()
            )

            cluster_group = ClusterGroup(
                cluster_id=cluster_id,
                file_paths=cluster_file_paths,
                files_content=cluster_files_content,
                total_tokens=cluster_tokens,
            )
            cluster_groups.append(cluster_group)

            logger.debug(
                f"Cluster {cluster_id}: {len(cluster_file_paths)} files, "
                f"{cluster_tokens:,} tokens"
            )

        avg_tokens = total_tokens / len(cluster_groups) if cluster_groups else 0
        metadata = {
            "num_clusters": len(cluster_groups),
            "total_files": len(files),
            "total_tokens": total_tokens,
            "avg_tokens_per_cluster": int(avg_tokens),
        }

        logger.info(
            f"Created {len(cluster_groups)} clusters, "
            f"avg {int(avg_tokens):,} tokens/cluster"
        )

        return cluster_groups, metadata

    def _compute_optimal_k(self, total_tokens: int, num_files: int) -> int:
        """Compute optimal number of clusters based on token budget.

        Args:
            total_tokens: Total tokens across all files
            num_files: Number of files to cluster

        Returns:
            Optimal k value (minimum 1, maximum num_files)
        """
        # Calculate minimum k needed to stay under token budget
        min_k = math.ceil(total_tokens / self._max_tokens_per_cluster)

        # k cannot exceed number of files
        k = min(min_k, num_files)

        # k must be at least 1
        k = max(1, k)

        return k
