# src/virtues/sanity.py

from .loader import load_virtue_config


def validate_virtue_config() -> None:
    """
    Run basic sanity checks on virtues.yaml.
    Raise AssertionError or custom exceptions on failures.
    """
    cfg = load_virtue_config()

    assert cfg.nodes, "No virtue nodes defined"
    assert cfg.contexts, "No contexts defined"

    node_ids = set(cfg.nodes.keys())

    # edges reference valid nodes
    for e in cfg.edges:
        assert e.source in node_ids, f"Edge source {e.source} not a node"
        assert e.target in node_ids, f"Edge target {e.target} not a node"

    # contexts have reasonable weights
    for ctx_id, ctx in cfg.contexts.items():
        total_weight = sum(ctx.node_weights.values())
        assert total_weight > 0.0, f"Context {ctx_id} has zero total weight"

    # derived virtues reference valid nodes
    for dv_id, dv in cfg.derived_virtues.items():
        for node_id in dv.from_nodes.keys():
            assert node_id in node_ids, f"Derived virtue {dv_id} references unknown node {node_id}"
