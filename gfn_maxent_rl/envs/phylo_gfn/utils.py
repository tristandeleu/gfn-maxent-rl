from gfn_maxent_rl.envs.phylo_gfn.trees import Leaf, RootedTree


CHARACTERS_MAPS = {
    'DNA': {
        'A': 0b1000, 'C': 0b0100, 'G': 0b0010, 'T': 0b0001,
        'N': 0b1111, '?': 0b1111
    },
    'RNA': {
        'A': 0b1000, 'C': 0b0100, 'G': 0b0010, 'U': 0b0001,
        'N': 0b1111, '?': 0b1111
    },
    'DNA_WITH_GAP': {
        'A': 0b10000, 'C': 0b01000, 'G': 0b00100, 'T': 0b00010,
        '-': 0b00001, 'N': 0b11110, '?': 0b11110
    },
    'RNA_WITH_GAP': {
        'A': 0b10000, 'C': 0b01000, 'G': 0b00100, 'U': 0b00010,
        '-': 0b00001, 'N': 0b11110, '?': 0b11110
    }
}


def get_tree_type(tree):
    if tree is None:
        return 0
    elif isinstance(tree, Leaf):
        return 1
    elif isinstance(tree, RootedTree):
        return 2
    else:
        raise ValueError(f'Unknown type: {type(tree)}')
