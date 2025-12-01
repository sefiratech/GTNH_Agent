# src/semantics/ingest/__init__.py
"""
Runtime-assisted ingestion utilities for GTNH semantics.

These tools are *offline* helpers that:
  - read raw dumps from a running GTNH instance
  - normalize them into:
      * gtnh_items.generated.yaml
      * gtnh_blocks.generated.yaml
      * gtnh_recipes.generated.json
for use by SemanticsDB at runtime.

They are not imported in hot paths; they exist for build/maintenance workflows.
"""
