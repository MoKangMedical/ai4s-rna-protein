#!/usr/bin/env python3
"""
Protenix MPS Inference - Apple Silicon M4 GPU
"""
import os
import sys

PROTENIX_DIR = "/Users/apple/Desktop/ai4s-rna-protein/data/raw/all_atom_diffusion_model/Protenix"
os.chdir(PROTENIX_DIR)
sys.path.insert(0, PROTENIX_DIR)
sys.path.insert(0, "/Users/apple/Desktop/ai4s-rna-protein")

# Apply MPS patches BEFORE any Protenix import
import mps_patch

import torch
import logging
import warnings

warnings.filterwarnings("ignore")

# Patch InferenceRunner to support MPS
from runner.inference import InferenceRunner as OriginalRunner

class MPSInferenceRunner(OriginalRunner):
    def init_env(self):
        self.use_cuda = False
        if torch.cuda.device_count() > 0:
            self.use_cuda = True
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("[MPS] Using Apple Silicon GPU")
        else:
            self.device = torch.device("cpu")

import runner.inference
runner.inference.InferenceRunner = MPSInferenceRunner

# Set arguments and run
sys.argv = [
    "run_mps_inference.py",
    "--model_name", "protenix_base_default_v0.5.0",
    "--seeds", "101",
    "--dump_dir", "./output_mps",
    "--input_json_path", "./competition_input.json",
    "--model.N_cycle", "1",
    "--sample_diffusion.N_sample", "1",
    "--sample_diffusion.N_step", "5",
    "--triangle_attention", "torch",
    "--triangle_multiplicative", "torch",
    "--use_msa", "false",
    "--dtype", "fp32",
]

from runner.inference import run
run()
