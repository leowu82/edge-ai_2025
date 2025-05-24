from hqq.core.quantize import BaseQuantizeConfig

# TODO: Make your own quant config for DeiT-S
def get_quant_config_deit(model):
    # Print the model architecture
    # print(model)
    
    quant_config = {}
    
    n_blocks = len(model.blocks)
    q8_64_config = BaseQuantizeConfig(nbits=8, group_size=64)
    q8_32_config = BaseQuantizeConfig(nbits=8, group_size=32)
    q4_64_config = BaseQuantizeConfig(nbits=4, group_size=64)
    q4_32_config = BaseQuantizeConfig(nbits=4, group_size=32)
    q3_64_config = BaseQuantizeConfig(nbits=3, group_size=64)
    q3_32_config = BaseQuantizeConfig(nbits=3, group_size=32)
    q2_64_config = BaseQuantizeConfig(nbits=2, group_size=64)
    q2_32_config = BaseQuantizeConfig(nbits=2, group_size=32)
    
    for i in range(n_blocks):
        quant_config[f'blocks.{i}.attn.qkv'] = q4_64_config
        quant_config[f'blocks.{i}.attn.proj'] = q4_64_config
        quant_config[f'blocks.{i}.mlp.fc1'] = q4_32_config
        quant_config[f'blocks.{i}.mlp.fc2'] = q4_32_config
        
    return quant_config

# TODO: Make your own quant config for Language Model
def get_quant_config_slm(model):
    # Print the model architecture
    # print(model)
    
    quant_config = {}
    
    n_layers = model.config.num_hidden_layers
    q8_64_config = BaseQuantizeConfig(nbits=8, group_size=64) 
    q8_32_config = BaseQuantizeConfig(nbits=8, group_size=32) 
    q4_64_config = BaseQuantizeConfig(nbits=4, group_size=64) 
    q4_32_config = BaseQuantizeConfig(nbits=4, group_size=32) 
    q2_64_config = BaseQuantizeConfig(nbits=2, group_size=64) 
    q2_32_config = BaseQuantizeConfig(nbits=2, group_size=32) 
    
    for i in range(n_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q4_64_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q4_64_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q4_64_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q8_64_config
        
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q4_64_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q4_64_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q8_64_config
        
    return quant_config