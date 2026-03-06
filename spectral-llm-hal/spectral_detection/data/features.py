import torch
from typing import Dict

# SPECTRAL / "EIGENVALUE" FEATURE EXTRACTION
@torch.no_grad()
def extract_topk_laplacian_eigs(model: torch.nn.Module, output_sequence_1d: torch.Tensor, k: int = 10) -> Dict[str, torch.Tensor]:
    """
    Compute top-k Laplacian eigenvalues and LLM-Check Attention Scores per head.
    output_sequence_1d: shape [T] (full prompt+generated sequence).
    Returns: Dictionary containing flattened float16 CPU tensors of Attention Eigenvalues, and Attention Scores
    """
    input_ids = output_sequence_1d.unsqueeze(0).to(model.device)  # [1, T]

    out = model(
        input_ids=input_ids,
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )
    
    attns = out.attentions
    if attns is None:
        raise RuntimeError("Attentions are still None even with eager attention.")

    # [L, 1, H, T, T] -> [L, H, T, T]
    A = torch.stack(attns, dim=0).squeeze(1)
    _, _, T_len, _ = A.shape

    col_sums = A.sum(dim=-2)  # [L, H, T]
    denom = (T_len - torch.arange(T_len, device=A.device)).clamp_min(1)  # [T] 
    d_ii = col_sums / denom
    
    a_ii = torch.diagonal(A, dim1=-2, dim2=-1)  # [L, H, T]
    
    # Laplacian Eigenvalues
    eigenvalues = d_ii - a_ii  # [L, H, T]
    k_val = min(int(k), T_len)
    sorted_eigvals, _ = torch.sort(eigenvalues, dim=-1, descending=True)
    top_k = sorted_eigvals[..., :k_val]  # [L, H, k]
    
    # Flatten from [L, H, k] -> [L * H * k]
    feat_laplacian = top_k.flatten().detach().cpu().to(torch.float16)

    # Attention Score (Per Head) defined as the mean of the log-diagonals 
    # 1/m \sum log (aii)
    attn_score_head = torch.log(a_ii.clamp(min=1e-10)).mean(dim=-1)  # [L, H]
    
    # Flatten from [L, H] -> [L * H]
    feat_attn = attn_score_head.flatten().detach().cpu().to(torch.float16)

    # Clean up GPU memory
    del out, A, col_sums, denom, d_ii, a_ii, eigenvalues, sorted_eigvals, top_k, attn_score_head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "laplacian": feat_laplacian,      # Shape: [L * H * k]
        "attention_score": feat_attn      # Shape: [L * H]
    }