"""
Helper constants related to colors, imported in ./bar_chart_builder.py and 
./violin_chart_builder.py
"""

DARK_RAINBOW =  ['ff4d4d', 'ff8000', 'ffcc00', '80ff66', '33eeff', '4d91ff', '674dff', 'ff66ff', '966919', '1e7b1e', '009961']
LIGHT_RAINBOW = ['ffadad', 'ffb366', 'ffe066', 'caffbf', '9bf6ff', 'a0c4ff', 'bdb2ff', 'ffc6ff', 'da9825', '2db92d', '00cc81']
MID_RAINBOW =   ['ff8080', 'ff9933', 'ffd633', 'aaff99', '66f2ff', '80b0ff', '9380ff', 'ff99ff', 'ae791e', '28a428', '00b371']
#                 red       orange    yellow    green     cyan      bleu      purple    pink      golden    forest    jade

COLOR_PERMUTATION = [1, 0, 2, 3, 10, 4, 5, 6, 7]

PALETTES = {}
PALETTES['DARK'] = [DARK_RAINBOW[i] for i in COLOR_PERMUTATION]
PALETTES['LIGHT'] = [LIGHT_RAINBOW[i] for i in COLOR_PERMUTATION]
PALETTES['MID'] = [MID_RAINBOW[i] for i in COLOR_PERMUTATION]

DARK_PALETTE = MODEL2NAME = {
    "command_r": DARK_RAINBOW[1],
    "command_r+": DARK_RAINBOW[0],
    "gpt-4o": DARK_RAINBOW[2],
    "llama": DARK_RAINBOW[3],
    "silma": DARK_RAINBOW[10],
    'acegpt': DARK_RAINBOW[4],
    "command_r_base": DARK_RAINBOW[5],
    "command_r+_base": DARK_RAINBOW[6],
    "llama-base": DARK_RAINBOW[7],
}
LIGHT_PALETTE = MODEL2NAME = {
    "command_r": LIGHT_RAINBOW[1],
    "command_r+": LIGHT_RAINBOW[0],
    "gpt-4o": LIGHT_RAINBOW[2],
    "llama": LIGHT_RAINBOW[3],
    "silma": LIGHT_RAINBOW[10],
    'acegpt': LIGHT_RAINBOW[4],
    "command_r_base": LIGHT_RAINBOW[5],
    "command_r+_base": LIGHT_RAINBOW[6],
    "llama-base": LIGHT_RAINBOW[7],
}
MID_PALETTE = MODEL2NAME = {
    "command_r": MID_RAINBOW[1],
    "command_r+": MID_RAINBOW[0],
    "gpt-4o": MID_RAINBOW[2],
    "llama": MID_RAINBOW[3],
    "silma": MID_RAINBOW[10],
    'acegpt': MID_RAINBOW[4],
    "command_r_base": MID_RAINBOW[5],
    "command_r+_base": MID_RAINBOW[6],
    "llama-base": MID_RAINBOW[7],
}
