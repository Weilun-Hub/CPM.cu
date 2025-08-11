from .. import C
from .tree_drafter import LLM_with_tree_drafter
import math, torch
from transformers import PretrainedConfig
from ..common.logging import logger

class Eagle3Config(PretrainedConfig):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
    

class LLM_with_eagle3(LLM_with_tree_drafter):
    def __init__(self,
                 eagle_path,
                 base_path,
                 num_iter=6,
                 topk_per_iter=10,
                 tree_size=60,
                 eagle_window_size=0,
                 apply_eagle_quant: bool=False,
                 **kwargs):
        super().__init__(
            "eagle3", eagle_path, base_path,
            tree_size = tree_size,
            **kwargs
        )
        self.apply_eagle_quant = apply_eagle_quant
        
        self.eagle_path = eagle_path
        self.eagle_config = Eagle3Config.from_pretrained(eagle_path)
        self.eagle_config.update({
            "head_dim": self.eagle_config.hidden_size // self.eagle_config.num_attention_heads,
            "rope_base": 10000,
        })

        # For Qwen3, head_dim is explicitly specified in config and may not equal hidden_size // num_attention_heads
        if not hasattr(self.eagle_config, "head_dim"):
            self.eagle_config.head_dim = self.eagle_config.hidden_size // self.eagle_config.num_attention_heads
        else:
            # Qwen3 models have explicit head_dim that might be different
            logger.info(f"Using explicit head_dim from eagle config: {self.eagle_config.head_dim}")
        
        scale_residual = self.config.scale_depth / math.sqrt(self.config.num_hidden_layers + 1)
        
        if apply_eagle_quant and hasattr(self.eagle_config, "quantization_config"):
            self.group_size = self.eagle_config.quantization_config.get('group_size', 0)
        else:
            self.group_size = 0
        assert self.group_size == 128 or self.group_size == 0, "only group_size 128 is supported in quantization mode"
        
        C.init_eagle3_model(
            self.eagle_config.num_hidden_layers,
            self.eagle_config.intermediate_size,
            self.eagle_config.num_attention_heads,
            self.eagle_config.num_key_value_heads,
            self.eagle_config.head_dim,
            self.eagle_config.rms_norm_eps,
            num_iter,
            topk_per_iter,
            self.tree_size,
            self.dtype_int,
            apply_eagle_quant,
            self.group_size,
            eagle_window_size,
            self.eagle_config.draft_vocab_size,
            scale_residual
        )

    def _load(self, name, param, dtype=None, cls=None):
        
        if cls == self.drafter_type:
            if dtype is None:
                dtype = self.dtype
            param = param.contiguous()

            if 'd2t' in name:
                param = param.to(torch.int32)
                C.load_model(f"{cls}.{name}", param.data_ptr())
                return

            if not self.apply_eagle_quant:
                param = param.to(dtype)
            
            if 'embed_tokens' in name:
                return
            
            C.load_model(f"{cls}.{name}", param.data_ptr())
        else:
            super()._load(name, param, dtype)
    
    def load_from_hf(self):

        super().load_from_hf()
        
        inv_freq = 1.0 / (self.eagle_config.rope_base ** (torch.arange(0, self.eagle_config.head_dim, 2).float() / self.eagle_config.head_dim))
        self._load(f"{self.drafter_type}.rotary_emb.inv_freq", inv_freq, dtype=torch.float32)


        
        

