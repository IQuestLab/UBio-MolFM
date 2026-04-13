import torch
from loguru import logger

def check_tensor_match(
    tensor_ref: torch.Tensor, 
    tensor_my: torch.Tensor, 
    atol: float = 1e-3, 
    rtol: float = 1e-3, 
    show_details: bool = True
) -> bool:
    """
    Checks if the numerical difference between two tensors is within the specified tolerance, 
    and provides a detailed error distribution log.
    
    Args:
        tensor_ref: The reference tensor (e.g., official/standard output).
        tensor_my: The tensor to be tested (e.g., your custom Triton kernel output).
        atol: Absolute tolerance threshold.
        rtol: Relative tolerance threshold.
        show_details: Whether to print a detailed comparison report.
        
    Returns:
        bool: True if the tensors match within the tolerances, False otherwise.
    """
    logger.info("Verifying tensors")
    
    # Ensure both tensors are on the same device before comparison
    if tensor_ref.device != tensor_my.device:
        tensor_my = tensor_my.to(tensor_ref.device)

    # 1. Core comparison logic
    is_match = torch.allclose(tensor_ref, tensor_my, atol=atol, rtol=rtol)
    
    # 2. Print detailed report
    if show_details:
        # Calculate maximum absolute error
        diff = torch.abs(tensor_ref - tensor_my)
        max_abs_error = torch.max(diff).item()
        
        logger.info("========================================")
        if is_match:
            logger.info(
                "Precision test passed | atol={} | rtol={} | max_abs_error={:.6f}",
                atol,
                rtol,
                max_abs_error,
            )
        else:
            logger.error(
                "Precision test failed | atol={} | rtol={} | max_abs_error={:.6f}",
                atol,
                rtol,
                max_abs_error,
            )
            
            # Count the number and ratio of elements exceeding the tolerance
            error_mask = diff > atol
            error_count = error_mask.sum().item()
            total_count = tensor_ref.numel()
            error_ratio = (error_count / total_count) * 100
            
            logger.error(
                "Error distribution | exceeded={} / {} ({:.2f}%)",
                error_count,
                total_count,
                error_ratio,
            )
            
            # Sample the first 5 error locations for kernel debugging
            error_indices = torch.nonzero(error_mask)
            logger.error("Error samples (Index: Reference vs. Yours)")
            for i in range(min(5, error_count)):
                idx = tuple(error_indices[i].tolist())
                val_ref = tensor_ref[idx].item()
                val_my = tensor_my[idx].item()
                val_diff = abs(val_ref - val_my)
                logger.error(
                    "Index {} | ref={:.6f} | my={:.6f} | diff={:.6f}",
                    idx,
                    val_ref,
                    val_my,
                    val_diff,
                )
        logger.info("========================================")

    return is_match
