from pathlib import Path
import torch
import logging

def correct_checkpoint_keys(input_path, output_path=None):
    """
    Load a PyTorch checkpoint file and remove the 'model.' prefix from the keys in the state_dict.
    Then, save the resulting state_dict as a .pth file and remove the original .ckpt file.

    Args:
        input_path (str or Path): Path to the input checkpoint (.ckpt) file.
        output_path (str or Path, optional): Path to save the corrected checkpoint file.
            If None, the corrected file overwrites the original file name with a .pth extension.

    Returns:
        None
    """
    logger = logging.getLogger("correct_checkpoint_keys")
    
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)

    # Check if the input file exists
    if not input_path.exists():
        logger.warning(f"Input file {input_path} does not exist")
        return

    # Load the checkpoint
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

    # Extract the state_dict from the checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Create a new state_dict with corrected keys
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith('model.'):
            new_key = key[len('model.'):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]

    # Determine the .pth output path
    pth_path = output_path.with_suffix('.pth')

    # Save the corrected state_dict as a .pth file
    logger.info(f"Saving corrected checkpoint to {pth_path}")
    torch.save(new_state_dict, pth_path)

    # Remove the original .ckpt file
    if input_path.exists():
        logger.info(f"Removing original file {input_path}")
        input_path.unlink()
    if output_path != input_path and output_path.exists():
        logger.info(f"Removing original output file {output_path}")
        output_path.unlink() 