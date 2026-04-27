#!/usr/bin/env python3
"""
Package Protenix inference output into competition submission format.

Output format: output.zip containing {N}_pred.cif files
- 1_pred.cif for r001
- 2_pred.cif for r002
- 3_pred.cif for r003

Usage:
    python scripts/package_submission.py [--output_dir OUTPUT_DIR] [--protenix_dir PROTENIX_DIR]
"""
import argparse
import glob
import json
import os
import shutil
import zipfile


def find_best_cif(sample_dir: str, sample_name: str) -> str:
    """Find the best CIF file for a sample based on ranking score."""
    cif_files = glob.glob(os.path.join(sample_dir, "**/*.cif"), recursive=True)
    if not cif_files:
        raise FileNotFoundError(f"No CIF files found in {sample_dir}")
    
    # Try to find confidence JSON to rank
    json_files = glob.glob(os.path.join(sample_dir, "**/*confidence*.json"), recursive=True)
    
    if json_files:
        best_score = -float('inf')
        best_cif = None
        for jf in json_files:
            try:
                with open(jf) as f:
                    conf = json.load(f)
                score = conf.get("ranking_score", 0)
                if score > best_score:
                    best_score = score
                    # Find corresponding CIF
                    base = jf.replace("_summary_confidence_sample_", "_sample_").replace(".json", ".cif")
                    if os.path.exists(base):
                        best_cif = base
            except:
                pass
        if best_cif:
            return best_cif
    
    # Fallback: return first CIF
    return cif_files[0]


def package_submission(protenix_output_dir: str, submission_path: str):
    """Package Protenix output into submission zip."""
    sample_map = {
        "r001": "1_pred.cif",
        "r002": "2_pred.cif",
        "r003": "3_pred.cif",
    }
    
    temp_dir = "/tmp/submission_staging"
    os.makedirs(temp_dir, exist_ok=True)
    
    for sample_name, output_name in sample_map.items():
        sample_dir = os.path.join(protenix_output_dir, sample_name)
        if not os.path.exists(sample_dir):
            print(f"WARNING: {sample_dir} not found, skipping")
            continue
        
        try:
            best_cif = find_best_cif(sample_dir, sample_name)
            dest = os.path.join(temp_dir, output_name)
            shutil.copy2(best_cif, dest)
            size = os.path.getsize(dest)
            print(f"  {output_name} <- {best_cif} ({size/1024:.1f} KB)")
        except Exception as e:
            print(f"  ERROR for {sample_name}: {e}")
    
    # Create zip
    with zipfile.ZipFile(submission_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in os.listdir(temp_dir):
            if f.endswith('.cif'):
                zf.write(os.path.join(temp_dir, f), f)
    
    size = os.path.getsize(submission_path)
    print(f"\nSubmission: {submission_path} ({size/1024:.1f} KB)")
    
    # Cleanup
    shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None,
                       help="Protenix output directory")
    parser.add_argument("--protenix_dir", 
                       default=os.path.expanduser("~/Desktop/ai4s-rna-protein/data/raw/all_atom_diffusion_model/Protenix"))
    parser.add_argument("--submission_path",
                       default=os.path.expanduser("~/Desktop/ai4s-rna-protein/output.zip"))
    args = parser.parse_args()
    
    output_dir = args.output_dir or os.path.join(args.protenix_dir, "output_minimal")
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        print("Waiting for inference to complete...")
        return
    
    print(f"Packaging submission from: {output_dir}")
    print(f"Output: {args.submission_path}")
    package_submission(output_dir, args.submission_path)
    print("\nDone! Submit output.zip to the competition.")


if __name__ == "__main__":
    main()
