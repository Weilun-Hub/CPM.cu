#!/usr/bin/env python3
"""MiniCPM4 Test Generation Launcher"""

import sys
import subprocess
import argparse
from cpmcu.common.args import str2bool
from cpmcu.common.logging import logger


def main():
    """Launch generation with MiniCPM4 configuration"""
    parser = argparse.ArgumentParser(description='MiniCPM4 Generation Configuration')
    group = parser.add_argument_group('MiniCPM4 Configuration')
    
    group.add_argument('--apply-sparse', '--apply_sparse', default=True, type=str2bool, nargs='?', const=True,
                      help='Enable sparse attention (default: True)')
    group.add_argument('--apply-quant', '--apply_quant', default=True, type=str2bool, nargs='?', const=True,
                      help='Enable quantization for base model (default: True)')
    group.add_argument('--apply-eagle', '--apply_eagle', default=True, type=str2bool, nargs='?', const=True,
                      help='Enable Eagle speculative decoding (default: True)')
    group.add_argument('--apply-eagle-quant', '--apply_eagle_quant', default=True, type=str2bool, nargs='?', const=True,
                      help='Enable quantization for Eagle draft model (default: True)')
    group.add_argument('--spec-type', '--spec_type', type=str, default="eagle2", choices=["eagle2", "eagle3"],
                      help='Speculative decoding type (default: eagle2)')
    group.add_argument('--minicpm4-yarn', '--minicpm4_yarn', default=True, type=str2bool, nargs='?', const=True,
                      help='Enable MiniCPM4 YARN for long context support (default: True)')
    
    # Handle help requests
    if '--help' in sys.argv or '-h' in sys.argv:
        subprocess.run([sys.executable, "-m", "cpmcu.cli", "--help"])
        parser.print_help()
        return 0
        
    args, unknown_args = parser.parse_known_args()
    
    # Setup model paths
    model_path = "openbmb/MiniCPM4-8B-marlin-cpmcu" if args.apply_quant else "openbmb/MiniCPM4-8B"
    draft_model_path = "openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu" if args.apply_eagle_quant else "openbmb/MiniCPM4-8B-Eagle-FRSpec"
    frspec_path = "openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu"
    
    # Build command
    cmd_args = [sys.executable, "-m", "cpmcu.cli",
               "--model-path", model_path,
               "--model-type", "minicpm4" if args.apply_sparse else "minicpm"]
    
    if args.apply_eagle:
        cmd_args.extend(["--spec-type", args.spec_type, "--draft-model-path", draft_model_path, "--frspec-path", frspec_path])
    
    # Add YARN configuration
    if args.minicpm4_yarn:
        cmd_args.extend(["--minicpm4-yarn"])
    
    cmd_args.extend(unknown_args)
    
    try:
        return subprocess.run(cmd_args, check=True).returncode
    except (subprocess.CalledProcessError, KeyboardInterrupt) as e:
        if isinstance(e, KeyboardInterrupt):
            logger.warning("\nGeneration interrupted by user")
            return 0
        logger.error(f"Generation failed: {e}")
        return e.returncode


if __name__ == "__main__":
    sys.exit(main()) 