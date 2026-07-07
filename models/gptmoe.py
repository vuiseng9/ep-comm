import torch
import torch.nn as nn
import torch.nn.functional as F
from .attn import Attn
from .moe import MoE
from .rope import RoPE
from .token_shuffler import EPBackend

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, ctx_size, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, ctx_size, embed_dim))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def extra_repr(self):
        return f"L={self.weight.size(1)}, D={self.weight.size(2)}"
    
    def forward(self, x):
        # x: (B, L, E) or (B, L)
        L = x.size(1)
        return self.weight[:, :L, :]
    

class TxBlock(nn.Module):
    def __init__(self, D, Dff, H, G, E, K, EP, rope=None, ep_backend=EPBackend.LOCAL, lbgamma=0.0, lbforce=False):
        super().__init__()
        self.D = D
        self.Dff = Dff
        self.H = H
        self.G = G
        self.E = E
        self.K = K
        self.EP = EP
        self.EPR = E // EP
        self.lbgamma = lbgamma
        self.lbforce = lbforce
        self.attn = Attn(D, H, G, rope)
        self.moe = MoE(E, K, EP, D, Dff, ep_backend, lbgamma, lbforce)

    def extra_repr(self):
        return f"E={self.E}, K={self.K}, EPR={self.EPR}, D={self.D}, Dff={self.Dff}, H={self.H}, G={self.G}"
    
    def forward(self, x, mask=None):
        h = self.attn(x, mask)
        return h + self.moe(h)
    
class GptMoE(nn.Module):
    def __init__(self, L, D, Dff, H, V, S, G, E, K, EP, ep_backend=EPBackend.LOCAL, lbgamma=0.0, lbforce=False):
        super().__init__()
        assert D%H==0, "D must be divisible by H"
        self.D = D # token/embedding/hidden dim
        self.H = H # number of attention head
        self.Dff = Dff # intermediate dim of MLP
        assert H%G==0, "H must be divisible by G"
        self.G = G # group size of gqa, 1 is mha
        assert E%EP==0, "E must be divisible by EP"
        self.E = E # n_experts
        self.K = K # n_activated_experts
        self.EPR = E // EP # number of experts per rank
        self.EP = EP # EP ranks
        self.lbgamma = lbgamma
        self.lbforce = lbforce

        self.V = V # vocab size
        self.S = S # context length
        self.L = L # number of transformer block
        
        self.token_embed = nn.Embedding(V, D)
        self.rope = RoPE(D//H, S)

        self.layers = nn.ModuleList(
            [TxBlock(D, Dff, H, G, E, K, self.EP, self.rope, ep_backend, lbgamma, lbforce) for _ in range(L)])
        
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.token_embed.weight # tie weight 

    def extra_repr(self):
        div = "─"*25
        return f"{div}\n" \
               f"L={self.L}, S={self.S}, V={self.V}\n" \
               f"D={self.D}, Dff={self.Dff}, H={self.H}, G={self.G}\n" \
               f"E={self.E}, K={self.K}, EP={self.EP}, EPR={self.EPR}\n" \
               f"force_lb={int(self.lbforce)}, lbgamma={self.lbgamma}\n" \
               f"tie_word_embeddings={int(self.lm_head.weight.data_ptr() == self.token_embed.weight.data_ptr())}\n" \
               f"{div}"

    @property
    def device(self):
        return self.token_embed.weight.device
                
    def _make_causal_mask(self, length):
        # make an lower-triangular matrix with entries of one 
        # and invert it to get mask position with true value.
        return torch.tril(torch.ones(1, 1, length, length, device=self.device)) == 0

    def forward(self, x, pad_mask=None):
        B, L = x.shape
        assert L <= self.S, "input length longer than model ctx length"

        causal_mask = self._make_causal_mask(L)

        # merge pad mask if provided
        if pad_mask is not None:

            query_mask = pad_mask.unsqueeze(1).unsqueeze(3)  # (B,1,S,1) masks queries
            key_mask   = pad_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,S) masks keys
            # causal_mask: (1, 1, S, S) broadcasted to (B, 1, S, S)
            # (B, 1, S, S) with be broacasted across heads dimension during attention computation
            attn_mask = causal_mask | key_mask | query_mask
        else:
            attn_mask = causal_mask

        h = self.token_embed(x) 

        for layer in self.layers:
            h = layer(h, causal_mask)

        logits = self.lm_head(h)
        return logits
    
    @torch.no_grad()
    def generate(self, tokenizer, prompts: list[str], max_new_tokens: int, T: float = -1.0):
        was_training = self.training
        self.eval()

        if tokenizer is None:
            raise ValueError(f"No tokenizer has been registered, generate function only works with a tokenizer. "
                             "Please register one via instance.register_tokenizer(tokenizer)")
        
        if not isinstance(prompts, list):
            raise TypeError(f"prompts must be a list of strings; use list of 1 element for 1 string")
        elif not isinstance(prompts[0], str):
            raise TypeError(f"prompts[0] is not a string; {type(prompts[0]).__name__}")
        
        # padding required for batch generation
        # on left side because it is causal.
        # right is for training convention and other generic use case.
        # therefore we need to revert at the end after tokenizer use
        orig_pad_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        
        # tokenize the prompts
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        device = next(self.parameters()).device

        ids = encoded['input_ids'].to(device)  # (B, S)
        # [Misnormer] attention_mask means which are real token, not the triangular mask used in attention
        # triangular causal mask is taken care at lower level, the forward function.
        pad_mask = encoded['attention_mask'] == 0 # also just a convention here, 0 means padding in HF tokenizer T_T, we need otherwise
        pad_mask = pad_mask.to(device)  # (B, S)
        
        # restore original padding side
        tokenizer.padding_side = orig_pad_side
                
        # for simplicity, we don't implement kv caching
        for _ in range(max_new_tokens):
            # crop to context length (self.S) from last tokens
            ids_cond = ids[:, -self.S:]
            pad_mask_cond = pad_mask[:, -self.S:]         

            logits = self.forward(ids_cond, pad_mask=pad_mask_cond)

            logits = logits[:, -1, :]   # take the last token's logits (B, V)
            
            # T <= 0: greedy (argmax); T > 0: temperature-scaled sampling
            if T <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
            else:
                # temperature scaling
                logits = logits / T
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)    # (B, 1)

            ids = torch.cat((ids, next_token), dim=1)       # (B, S+1)
            # extend pad_mask with False
            pad_mask = torch.cat((pad_mask, torch.zeros_like(next_token, dtype=torch.bool)), dim=1)
        
        # decode the generated token ids back to text
        generated_texts = tokenizer.batch_decode(ids, skip_special_tokens=True)
        
        if was_training:
            self.train()
        return generated_texts
    
if __name__ == "__main__":
    D = 16
    DFF = 48
    H = 4
    V = 256
    L = 64
    G = 2
    E = 4
    K = 2
    nlayer = 3

    model = GptMoE(D, DFF, H, V, L, E, K, nlayer, G=G)
    print(model)

    x = torch.randint(0, V-1, size=(3, L))
    y = model(x)
    print("x.shape", x.shape)
    print("y.shape", y.shape)

    print("end.")