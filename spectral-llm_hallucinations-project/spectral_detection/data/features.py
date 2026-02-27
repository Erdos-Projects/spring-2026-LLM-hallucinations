import torch

def extract_laplacian(model, laplacian_features_dict, num_top_eigenvalues, output_sequence, answer_id) -> None:   
    """
    Extract the Laplacian as described in paper:  
    Binkowski et al, Hallucination Detection in LLMs Using Spectral Features of Attention
    """               
    with torch.no_grad():
        forward_out = model(output_sequence.unsqueeze(0), output_attentions=True)
        
        # A is a contiguous 4-dimensional PyTorch tensor of dimensions (L, H, T, T) 
        # where T = number of tokens of the prompt
        A = torch.stack(forward_out.attentions, dim=0).squeeze(1)
        _, _, T_len, _ = A.shape
        
        # Collapse columns, i.e. sum across rows
        col_sums = A.sum(dim=-2) 
        T_minus_i = T_len - torch.arange(T_len, device=A.device)
        d_ii = col_sums / T_minus_i 
        a_ii = torch.diagonal(A, dim1=-2, dim2=-1) 

        # Equation (2) from the paper
        eigenvalues = d_ii - a_ii 
        
        k = min(num_top_eigenvalues, T_len)
        sorted_eigvals, _ = torch.sort(eigenvalues, dim=-1, descending=True)
        top_k_eigvals = sorted_eigvals[..., :k]
        
        # Cast to CPU float16 to match paper optimization and save RAM
        laplacian_tensor = top_k_eigvals.flatten().cpu().to(torch.float16)
        
        # Store in the dictionary 
        laplacian_features_dict[answer_id] = laplacian_tensor
        
        del forward_out, A, col_sums, T_minus_i, d_ii, a_ii, eigenvalues, sorted_eigvals, top_k_eigvals, laplacian_tensor