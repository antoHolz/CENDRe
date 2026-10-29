# Hydra quality of life module.hydraqol.py
import os
import sys
import json
import logging
import datetime
import traceback
from functools import wraps
from pathlib import Path
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from collections.abc import Mapping, Sequence
import shutil

def run_decorator(func):
    @wraps(func)
    def wrapper(cfg, *args, **kwargs):
        logger = logging.getLogger("wrapper")
        start_time = datetime.datetime.now()
        script_name = os.path.basename(sys.argv[0])
        save_dir = Path(cfg.save_dir)
        run_info_path = save_dir / "run_info.json"
        config_path = save_dir / "config.yaml"
        hydra_config_path = save_dir / "hydra_config.yaml"
        procid = os.environ.get("SLURM_PROCID", "0")

        # Determine mode from cfg; defaults to 'base' if not specified
        mode = getattr(cfg, "mode", "base")
        valid_modes = ["base", "force", "force-report", "force-metric", "force-correctness", "clean", "check"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")

        # --------------------------------------------
        # Step 1: Gather current run status information
        # --------------------------------------------
        save_dir_exists = save_dir.exists()
        run_info_exists = run_info_path.exists()
        config_exists = config_path.exists()

        if not save_dir_exists:
            run_status = "not run"
        else:
            if run_info_exists:
                run_status = "completed"
            elif config_exists:
                run_status = "incomplete"
            else:
                run_status = "not run (only folder)"
            
        # log status, mode, save_dir, and procid
        logger.info(f"state: {run_status}")
        logger.info(f"mode: {mode}")
        logger.info(f"save_dir: {save_dir}")
        logger.info(f"proc ID: {procid}")

        # --------------------------------------------
        # Step 2: Mode Handling
        # --------------------------------------------
        if mode == "check":
            # Check mode: only check status and return
            run_info = {
                "state": run_status,
                "mode": mode,
                "func_name": func.__name__,
                "executed_file": script_name,
                "save_dir": str(cfg.save_dir),
                "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_time_seconds": (datetime.datetime.now() - start_time).total_seconds(),
                "result": None
            }
            return run_info

        if mode == "base":
            # If run is completed, skip
            if run_status == "completed":
                logger.info(f"[BASE] Run already completed in {save_dir}. Skipping...")
                run_info = {
                    "state": "skipped",
                    "mode": mode,
                    "func_name": func.__name__,
                    "executed_file": script_name,
                    "save_dir": str(cfg.save_dir),
                    "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "end_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "total_time_seconds": (datetime.datetime.now() - start_time).total_seconds(),
                    "result": None
                }
                logger.info(f"[TASK FINISHED] "+"#"*50)
                for key, value in run_info.items():
                    logger.info(f"{key}: {value}")
                logger.info(f"[TASK FINISHED] "+"#"*50)
                return run_info
            # Otherwise, proceed

        elif mode == "force":
            # Force mode: always remove and re-run
            if save_dir_exists:
                logger.info(f"[FORCE] Removing existing directory {save_dir} and rerunning...")
                shutil.rmtree(save_dir, ignore_errors=True)

        elif mode == "clean":
            # Clean mode: if completed, skip like base; if not completed, remove and re-run
            if run_status == "completed":
                logger.info(f"[CLEAN] Run already completed in {save_dir}. Skipping...")
                run_info = {
                    "state": "skipped",
                    "mode": mode,
                    "func_name": func.__name__,
                    "executed_file": script_name,
                    "save_dir": str(cfg.save_dir),
                    "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "end_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "total_time_seconds": (datetime.datetime.now() - start_time).total_seconds(),
                    "result": None
                }
                logger.info(f"[TASK FINISHED] "+"#"*50)
                for key, value in run_info.items():
                    logger.info(f"{key}: {value}")
                logger.info(f"[TASK FINISHED] "+"#"*50)
                return run_info
            else:
                if save_dir_exists:
                    logger.info(f"[CLEAN] Run not completed in {save_dir}. Removing directory and rerunning...")
                    shutil.rmtree(save_dir, ignore_errors=True)

        # --------------------------------------------
        # Step 3: Setup save_dir and config
        # --------------------------------------------
        if int(procid) == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            # Save the config
            with open(config_path, "w") as f:
                OmegaConf.save(config=cfg, f=f)
            # Save the Hydra config
            #with open(hydra_config_path, "w") as f:
            #    OmegaConf.save(config=HydraConfig.get(), f=f)
            

        # --------------------------------------------
        # Step 4: Run the function
        # --------------------------------------------
        try:
            # Run the function
            result = func(cfg, *args, **kwargs)
            # On success, record new run_info
            run_info = {
                "status": "success",
                "mode": mode,
                "func_name": func.__name__,
                "executed_file": script_name,
                "save_dir": str(cfg.save_dir),
                "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_time_seconds": (datetime.datetime.now() - start_time).total_seconds(),
                "result": result
            }
            if int(procid) == 0:
                with open(run_info_path, "w") as f:
                    json.dump(run_info, f, indent=4)
            logger.info(f"[TASK FINISHED] "+"#"*50)
            for key, value in run_info.items():
                logger.info(f"{key}: {value}")
            logger.info(f"[TASK FINISHED] "+"#"*50)
            return run_info

        except Exception as e:
            # --------------------------------------------
            # Step 5: In case of error, save traceback
            # --------------------------------------------
            error_file = save_dir / f"run_error_{procid}.txt"
            with open(error_file, "w") as f:
                traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
                f.write(traceback_str)
            raise e

    return wrapper





################################################################################################
# Custom OmegaConf resolvers
################################################################################################
def default(val, default=1):
    # Here, you could add more logic to determine if val is "missing"
    return default if val is None else val
OmegaConf.register_new_resolver("default", default)

def get_mp_start_method():
    """
    Returns the appropriate multiprocessing start method based on the operating system.
    Windows: 'spawn'
    Linux: 'fork'
    """
    import platform
    return 'spawn' if platform.system().lower() == 'windows' else 'fork'

OmegaConf.register_new_resolver("mp_start_method", get_mp_start_method)

def math(operator, *args):
    """
    Custom resolver for mathematical operations.
    
    Args:
        operator (str): The mathematical operator to use ('+', '-', '*', '/', '**', etc.)
        *args: The operands for the mathematical operation
        
    Returns:
        The result of applying the operator to the operands
    """
    if not args:
        raise ValueError("At least one operand is required for math operations")
    
    # Convert all arguments to float for consistent handling
    operands = [float(arg) for arg in args]
    
    # Apply the operator
    if operator == '+':
        return sum(operands)
    if operator == '++':
        return int(sum(operands))
    elif operator == '-':
        result = operands[0]
        for operand in operands[1:]:
            result -= operand
        return result
    elif operator == '*':
        result = 1
        for operand in operands:
            result *= operand
        return result
    elif operator == '/':
        result = operands[0]
        for operand in operands[1:]:
            if operand == 0:
                raise ValueError("Division by zero")
            result /= operand
        return result
    elif operator == '**':
        if len(operands) != 2:
            raise ValueError("Power operation requires exactly 2 operands")
        return operands[0] ** operands[1]
    elif operator == '%':
        if len(operands) != 2:
            raise ValueError("Modulo operation requires exactly 2 operands")
        if operands[1] == 0:
            raise ValueError("Modulo by zero")
        return operands[0] % operands[1]
    elif operator == '//':
        if len(operands) != 2:
            raise ValueError("Floor division operation requires exactly 2 operands")
        if operands[1] == 0:
            raise ValueError("Division by zero")
        return operands[0] // operands[1]
    else:
        raise ValueError(f"Unsupported operator: {operator}")

OmegaConf.register_new_resolver("math", math)
OmegaConf.register_new_resolver("add", lambda a, b: int(a) + int(b), use_cache=True)

OmegaConf.register_new_resolver("take_id", lambda a,b: a[b])


_STANDARDIZE_TARGETS = {
    "modules.transforms.hftransforms.HFStandardize2",
    "modules.transforms.hftransforms.HFStandardize1",
    "modules.transforms.tstransforms.TSStandardize",
}

def _contains_standardize(node) -> bool:
    if node is None:
        return False

    # dict-like
    if isinstance(node, Mapping):
        tgt = node.get("_target_")
        if tgt in _STANDARDIZE_TARGETS:
            return True
        return any(_contains_standardize(v) for v in node.values())

    # list/tuple-like (but not strings)
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        return any(_contains_standardize(x) for x in node)

    return False

def requires_standardize(*nodes) -> bool:
    return any(_contains_standardize(n) for n in nodes)

OmegaConf.register_new_resolver("requires_standardize", requires_standardize, replace=True)