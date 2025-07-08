from dataclasses import is_dataclass, fields, dataclass
from pathlib import Path
import msgspec

def dataclass_to_dict(input_object: dataclass)->dict:
    """Convert a dataclass instance (and any nested dataclasses) to a dictionary.
    
    Args:
        input_object: A dataclass instance or a structure containing dataclasses
        
    Returns:
        A dictionary representation of the dataclass with nested dataclasses also converted
    """
    
    # Base case: None
    if input_object is None:
        return None
        
    # Recursive case: dataclass instance
    if is_dataclass(input_object):
        result = {}
        for field in fields(input_object):
            field_value = getattr(input_object, field.name)
            result[field.name] = dataclass_to_dict(field_value)
        return result
        
    # Recursive case: list/tuple containing possibly nested dataclasses
    if isinstance(input_object, (list, tuple)):
        return type(input_object)(dataclass_to_dict(item) for item in input_object)
        
    # Recursive case: dictionary containing possibly nested dataclasses
    if isinstance(input_object, dict):
        return {key: dataclass_to_dict(value) for key, value in input_object.items()}
        
    # Special case: Path objects
    if isinstance(input_object, Path):
        return str(input_object)
        
    if isinstance(input_object, msgspec.Struct):
        return {f: getattr(input_object, f) for f in input_object.__struct_fields__}

    # Base case: return the object itself (int, float, str, etc.)
    return input_object


def flatten_dict(d, parent_key='', sep='.'):
    out = {}
    for k in d:
        v = d[k]
        if isinstance(v, dict):
            new_parent_key = f"{parent_key}{sep if parent_key else ''}{k}"
            out.update(flatten_dict(v, parent_key=new_parent_key, sep=sep))
        else:
            out[f"{parent_key}{sep if parent_key else ''}{k}"] = d[k]
    return out
