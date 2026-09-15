from enum import IntEnum

class ActivationType(IntEnum):
    InvalidType = 0
    Identity = 1
    Gelu = 2
    Relu = 3
    Silu = 4
    Swiglu = 5
    Geglu = 6
    SwigluBias = 7
    Relu2 = 8

def is_gated_activation(activation_type: ActivationType) -> bool:
    return activation_type in [ActivationType.Swiglu, ActivationType.SwigluBias, ActivationType.Geglu]
