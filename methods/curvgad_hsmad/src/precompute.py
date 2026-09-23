"""Auditable, label-free curvature preprocessing for frozen HSMAD graphs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy.sparse import csr_matrix, load_npz, save_npz


@dataclass(frozen=True)
class CurvatureArtifact:
    matrix: csr_matrix
    matrix_path: Path
    metadata_path: Path
    matrix_sha256: str
    cache_hit: bool


def _sha256_bytes(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def graph_topology_sha256(graph) -> str:
    """Hash node count and the complete directed edge multiset; never inspect ndata."""
    src, dst = graph.edges(order="eid")
    edges = np.stack(
        (src.detach().cpu().numpy().astype("<i8"), dst.detach().cpu().numpy().astype("<i8")),
        axis=1,
    )
    if len(edges):
        edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    return _sha256_bytes(np.asarray([graph.num_nodes()], dtype="<i8").tobytes(), edges.tobytes())


def cache_key(graph_sha256: str, *, method: str, alpha: float, implementation_sha256: str) -> str:
    payload = json.dumps(
        {
            "alpha": float(alpha),
            "graph_sha256": graph_sha256,
            "implementation_sha256": implementation_sha256,
            "method": method,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _matrix_sha256(matrix: csr_matrix) -> str:
    canonical = matrix.tocsr()
    canonical.sort_indices()
    return _sha256_bytes(
        np.asarray(canonical.shape, dtype="<i8").tobytes(),
        canonical.indptr.astype("<i8", copy=False).tobytes(),
        canonical.indices.astype("<i8", copy=False).tobytes(),
        canonical.data.astype("<f8", copy=False).tobytes(),
    )


def _simple_undirected_graph(graph) -> nx.Graph:
    src, dst = graph.edges(order="eid")
    result = nx.Graph()
    result.add_nodes_from(range(graph.num_nodes()))
    for u, v in zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist()):
        if u != v:
            result.add_edge(int(u), int(v))
    return result


def compute_or_load_exact_curvature(
    graph,
    cache_dir: Path,
    *,
    alpha: float = 0.5,
    method: str = "exact_orc",
) -> CurvatureArtifact:
    """Compute official exact ORC once and cache it by topology, code, and alpha."""
    if method != "exact_orc":
        raise ValueError("only method='exact_orc' is allowed by this candidate protocol")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    graph_sha256 = graph_topology_sha256(graph)
    implementation_sha256 = _implementation_sha256()
    key = cache_key(
        graph_sha256,
        method=method,
        alpha=alpha,
        implementation_sha256=implementation_sha256,
    )
    matrix_path = cache_dir / f"{key}.npz"
    metadata_path = cache_dir / f"{key}.json"
    if matrix_path.exists() and metadata_path.exists():
        matrix = load_npz(matrix_path).tocsr()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        matrix_sha256 = _matrix_sha256(matrix)
        if metadata.get("matrix_sha256") != matrix_sha256:
            raise RuntimeError("cached curvature matrix SHA256 mismatch")
        return CurvatureArtifact(matrix, matrix_path, metadata_path, matrix_sha256, True)

    simple_graph = _simple_undirected_graph(graph)
    orc = OllivierRicci(simple_graph, alpha=alpha, verbose="ERROR", proc=1)
    orc.compute_ricci_curvature()
    rows, columns, values = [], [], []
    for u, v, edge_data in orc.G.edges(data=True):
        rows.append(int(u))
        columns.append(int(v))
        values.append(float(edge_data["ricciCurvature"]))
    matrix = csr_matrix((values, (rows, columns)), shape=(graph.num_nodes(), graph.num_nodes()), dtype=np.float64)
    matrix_sha256 = _matrix_sha256(matrix)
    save_npz(matrix_path, matrix)
    metadata = {
        "alpha": float(alpha),
        "cache_key": key,
        "graph_sha256": graph_sha256,
        "implementation_sha256": implementation_sha256,
        "label_or_mask_access": "none",
        "matrix_sha256": matrix_sha256,
        "method": method,
        "num_nodes": int(graph.num_nodes()),
        "simple_undirected_edges_without_self_loops": int(simple_graph.number_of_edges()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return CurvatureArtifact(matrix, matrix_path, metadata_path, matrix_sha256, False)
