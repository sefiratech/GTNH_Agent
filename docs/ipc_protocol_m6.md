Right now:

Python assumes normalized events:

time_update, position_update, chunk_data, spawn_entity, destroy_entities, set_slot, window_items, dimension_change

And action packet types:

move_step, block_dig, block_place, use_item, interact

What’s missing is the actual spec for the Forge mod:

Define a small JSON schema:

Incoming → { "type": "chunk_data", "payload": { ... } }

Outgoing → { "type": "move_step", "payload": { "x":..., "y":..., "z":..., "radius":... } }
