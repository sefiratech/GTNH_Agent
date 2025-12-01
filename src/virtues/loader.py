# load virtues.yaml into VirtueConfig
# src/virtues/loader.py

from pathlib import Path
from typing import Dict, Any

import yaml

from .schema import (
    VirtueNode,
    VirtueEdge,
    VirtueContext,
    DerivedVirtueSpec,
    VirtueConfig,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
VIRTUES_PATH = CONFIG_DIR / "virtues.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_virtue_config(path: Path | None = None) -> VirtueConfig:
    raw = _load_yaml(path or VIRTUES_PATH)

    # nodes
    nodes: Dict[str, VirtueNode] = {}
    for node_id, data in raw["sefirot_nodes"].items():
        nodes[node_id] = VirtueNode(
            id=node_id,
            label=data.get("label", node_id),
            role=data.get("role", ""),
            tier=data.get("tier", ""),
            pillar=data.get("pillar", ""),
        )

    # edges
    edges = [
        VirtueEdge(source=e["from"], target=e["to"])
        for e in raw.get("edges", [])
    ]

    # contexts
    contexts: Dict[str, VirtueContext] = {}
    for ctx_id, ctx_data in raw.get("contexts", {}).items():
        contexts[ctx_id] = VirtueContext(
            id=ctx_id,
            description=ctx_data.get("description", ""),
            node_weights=ctx_data.get("node_weights", {}),
            hard_constraints=ctx_data.get("hard_constraints", []),
        )

    # derived virtues
    derived: Dict[str, DerivedVirtueSpec] = {}
    for dv_id, dv_data in raw.get("derived_virtues", {}).items():
        derived[dv_id] = DerivedVirtueSpec(
            id=dv_id,
            from_nodes=dv_data.get("from_nodes", {}),
            description=dv_data.get("description", ""),
        )

    features = raw.get("features", {})

    return VirtueConfig(
        nodes=nodes,
        edges=edges,
        features=features,
        contexts=contexts,
        derived_virtues=derived,
    )
