import numpy as np
import haiku as hk
import jax
import hydra

from functools import singledispatch


def class_fullname(obj):
    # https://stackoverflow.com/a/2020083
    klass = obj.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__ # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__

@singledispatch
def to_dict(obj, prefix):
    raise TypeError(f'Object "{prefix}" of unknown type: {type(obj)}')

@to_dict.register(np.ndarray)
@to_dict.register(jax.Array)
def _(obj, prefix):
    return {
        f'{prefix}/type': 'ndarray',
        f'{prefix}/value': np.asarray(obj)
    }

@to_dict.register(dict)
def _(obj, prefix):
    data = {f'{prefix}/type': 'haiku_dict'}
    tree = hk.data_structures.to_haiku_dict(obj)
    for module_name, name, value in hk.data_structures.traverse(tree):
        data[f'{prefix}/value/{module_name}/{name}'] = value
    return data

@to_dict.register(tuple)
def _(obj, prefix):
    if not hasattr(obj, '_asdict'):
        data = {f'{prefix}/type': 'tuple'}
        items = [(str(i), value) for (i, value) in enumerate(obj)]
    else:
        data = {
            f'{prefix}/type': 'namedtuple',
            f'{prefix}/class': class_fullname(obj)
        }
        items = list(obj._asdict().items())

    for field, value in items:
        field_data = to_dict(value, field)
        data.update({f'{prefix}/value/{k}': v for (k, v) in field_data.items()})

    return data

def save(filename, **trees):
    data = {}
    for key, value in trees.items():
        data.update(to_dict(value, key))
    np.savez(filename, **data)


def to_haiku_dict(dictionary):
    def _flatten(dictionary):
        if all(isinstance(value, (jax.Array, np.ndarray))
                for value in dictionary.values()):
            return dictionary
        else:
            data = {}
            for key, value in dictionary.items():
                data.update({f'{key}/{k}': v for (k, v)
                    in _flatten(value).items()})
            return data

    output = {}
    for key, value in _flatten(dictionary).items():
        module_name, _, name = key.rpartition('/')
        if module_name not in output:
            output[module_name] = {}
        output[module_name][name] = value
    
    return hk.data_structures.to_haiku_dict(output)

def from_dict(dictionary):
    assert ('type' in dictionary)
    obj_type = dictionary['type'].item()

    if obj_type == 'ndarray':
        return dictionary['value']
    
    elif obj_type == 'haiku_dict':
        value = dictionary.get('value', {})
        return to_haiku_dict(value)
    
    elif obj_type == 'tuple':
        data = {int(key): from_dict(value) for (key, value)
            in dictionary['value'].items()}
        return tuple(data[i] for i in range(len(data)))

    elif obj_type == 'namedtuple':
        klass = hydra.utils.get_class(dictionary['class'].item())
        data = {key: from_dict(value) for (key, value)
            in dictionary['value'].items()}
        return klass(**data)

    else:
        raise TypeError(f'Unknown type: {obj_type}')


def load(filename, **kwargs):
    data = {}
    f = open(filename, 'rb') if isinstance(filename, str) else filename
    results = np.load(f, **kwargs)

    # Unflatten the dictionary: https://stackoverflow.com/a/6037657
    for key in results.files:
        parts = key.split('/')
        d = data
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = results[key]

    for key, dictionary in data.items():
        data[key] = from_dict(dictionary)

    return data
