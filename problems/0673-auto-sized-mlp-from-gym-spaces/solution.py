# much harder than easy...

def auto_mlp_from_spaces(obs_space: dict, act_space: dict, hidden_layers: list) -> dict:
    """
    Automatically determine MLP architecture from environment space definitions.

    Args:
        obs_space: dict with 'type' and type-specific keys defining the observation space
        act_space: dict with 'type' and type-specific keys defining the action space
        hidden_layers: list of ints specifying hidden layer widths

    Returns:
        dict with keys:
          'input_dim': int
          'output_dim': int
          'layer_shapes': list of ((weight_rows, weight_cols), (bias_dim,)) tuples
          'total_params': int
    """
    def calculate_dim(space: dict) -> int:
        space_type = space.get('type')
        if space_type == 'Box':
            shape = space['shape']
            dim = 1
            for d in shape:
                dim *= d
            return dim
        elif space_type in ('Discrete', 'MultiBinary'):
            return space['n']
        else:
            raise ValueError(f"Unsupported space type: {space_type}")
    input_dim = calculate_dim(obs_space)
    output_dim = calculate_dim(act_space)
    dims = [input_dim] + hidden_layers + [output_dim]
    layer_shapes = [((dims[i], dims[i+1]), (dims[i+1],)) for i in range(len(dims) - 1)]
    total_params = sum(in_f * out_f + out_f for (in_f, out_f), _ in layer_shapes)

    return {
        'input_dim': input_dim, 
        'output_dim': output_dim, 
        'layer_shapes': layer_shapes, 
        'total_params': total_params
    }