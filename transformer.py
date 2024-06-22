import mlx.core as mx
import mlx.nn as nn

from typing import List, NamedTuple

class LayerWeights(NamedTuple):
    """
    holds the weight for a single transformer layer.

    - tok
    - w_q_dhk, w_k_dhk, w_v_dhk: Weights for query, key, and value projections
    - w_o_hkd: Weight for output projection in attention
    - w1, w2, w3: Weights for the mlp block
    """
    attn_norm: mx.array 
    ffn_norm: mx.array 
    w_q_dhk: mx.array 
    w_k_dhk: mx.array
    w_v_dhk: mx.array
    w_o_hkd: mx.array
    w1: mx.array
    w2: mx.array
    w3: mx.array

class XfmrWeights(NamedTuple):
    """
    holds all weight for the transformer model

    - tok_embedding: token embedding matrix
    - norm: final layer normalization weights
    - ouput: ouput projection weights
    """
    tok_embeddings: mx.array
    layer_weights: List[LayerWeights]
    norm: mx.array
    output: mx.array

def norm(x, w, eps: float = 1e-6):
    """
    applies layer normalization to the input
    """
    return w * (x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps))


def mlp(x: mx.array, w1: mx.array, w2: mx.array, w3: mx.array) -> mx.array:
    """ 
    mlp or feed forward network of transformer layer

    returns tensor after applying the feed forward network
    """
    return mx.matmul(nn.silu(mx.matmul(x, w1)) * mx.matmul(x, w3), w2)

def attention(input_bld, params):
    """
    compute multi head attension

    returns output tensor after applying attension
    B: batch size
    L: sequence length
    M: memory length 
    D: model dimension
    H: number of attention heads in a layer
    K: size of each attention key or value
    """
    normalized_bld = norm(input_bld, params.attn_norm)
    
    # Replace einsum operations with reshape and matmul
    b, l, d = normalized_bld.shape
    h, k = params.w_q_dhk.shape[-2:]
    
    query_blhk = mx.reshape(mx.matmul(normalized_bld, mx.reshape(params.w_q_dhk, (d, h*k))), (b, l, h, k))
    key_blhk = mx.reshape(mx.matmul(normalized_bld, mx.reshape(params.w_k_dhk, (d, h*k))), (b, l, h, k))
    value_blhk = mx.reshape(mx.matmul(normalized_bld, mx.reshape(params.w_v_dhk, (d, h*k))), (b, l, h, k))
    
    # Compute attention scores
    logits_bhlm = mx.matmul(mx.transpose(query_blhk, (0, 2, 1, 3)), mx.transpose(key_blhk, (0, 2, 3, 1)))
    logits_bhlm = logits_bhlm / mx.sqrt(k)
    
    # Create and apply attention mask
    mask = mx.triu(mx.ones((l, l)), k=1).astype(input_bld.dtype)
    logits_bhlm = logits_bhlm - mx.inf * mask[None, None, :, :]
    
    weights_bhlm = nn.softmax(logits_bhlm, axis=-1)
    
    # Apply attention weights to values
    wtd_values_blhk = mx.matmul(weights_bhlm, mx.transpose(value_blhk, (0, 2, 1, 3)))
    out_bld = mx.reshape(mx.matmul(mx.reshape(wtd_values_blhk, (b, l, h*k)), mx.reshape(params.w_o_hkd, (h*k, d))), (b, l, d))
    
    return out_bld


def transformer(token: mx.array, params: mx.array) -> mx.array:
    """
    applies the full transformer model to the input tokens

    returns logits for the next token prediction
    """

    x = params.tok_embeddings[tokens]
    for layer_weights in params.layer_weights:
        x += attention(x, layer_weights)
        x += mlp(norm(x, layer_weights.ffn_norm), layer_weights.w1, layer_weights.w2, layer_weights.w3)

    x = norm(x, params.norm)
    logits = mx.matmul(x, mx.transpose(params.output))
    return logits

if __name__ == '__main__':
    vocab_size = 320000
    dim = 4096
    hidden_dim = 14336
    n_layers = 1
    n_heads = 32
    head_dim = dim // n_heads

    layer_weights = LayerWeights(
        attn_norm=mx.ones((dim,)),
        ffn_norm=mx.ones((dim,)),
        w_q_dhk=mx.zeros((dim, n_heads, head_dim)),
        w_k_dhk=mx.zeros((dim, n_heads, head_dim)),
        w_v_dhk=mx.zeros((dim, n_heads, head_dim)),
        w_o_hkd=mx.zeros((n_heads, head_dim, dim)),
        w1=mx.zeros((dim, hidden_dim)),
        w2=mx.zeros((hidden_dim, dim)),
        w3=mx.zeros((dim, hidden_dim))
    )
    params = XfmrWeights(tok_embeddings=mx.ones((vocab_size, dim)), layer_weights=[layer_weights], norm=mx.ones((dim,)), output=mx.ones((vocab_size, dim)))
    tokens = mx.array([[123,234,234,345,446]])
    out = transformer(tokens, params)
    print(f'{out.shape=}')
