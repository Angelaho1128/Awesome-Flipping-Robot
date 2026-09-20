"""Preserve platform-specific arrow codes; masking to 8 bits loses arrows."""
def decode_key(key):
    if key in (63232,2490368,65362):return 'nudge_up'
    if key in (63233,2621440,65364):return 'nudge_down'
    return None
