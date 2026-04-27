#!/bin/bash
# AI4S RNA-Protein Competition - Optimized CPU Inference
# r001 already completed, run r002 and r003 separately
set -e
cd ~/Desktop/ai4s-rna-protein
source venv/bin/activate

PROTENIX_DIR="data/raw/all_atom_diffusion_model/Protenix"
OUTPUT_DIR="${PROTENIX_DIR}/output_minimal"

export PYTHONPATH="${PROTENIX_DIR}:${PYTHONPATH}"
export LAYERNORM_TYPE=torch

# Create separate input JSONs for r002 and r003
echo "Creating separate input files..."

python3 -c "
import json
with open('${PROTENIX_DIR}/competition_input.json') as f:
    data = json.load(f)
for item in data:
    name = item['name']
    with open(f'input_{name}.json', 'w') as f:
        json.dump([item], f, indent=2)
    print(f'  Created input_{name}.json')
"

echo ""
echo "=== Inference Plan ==="
echo "r001: DONE (ranking_score=0.137)"
echo "r002: Starting now (1114 tokens, ~15-20 min estimated)"
echo "r003: After r002 (1415 tokens, ~20-30 min estimated)"
echo ""

# Run r002
echo "============================================"
echo "Starting r002 inference: $(date)"
echo "============================================"
cd "${PROTENIX_DIR}"
python3 runner/inference.py \
    --model_name protenix_base_default_v0.5.0 \
    --seeds 101 \
    --dump_dir "./output_minimal" \
    --input_json_path ~/Desktop/ai4s-rna-protein/input_r002.json \
    --model.N_cycle 5 \
    --sample_diffusion.N_sample 1 \
    --sample_diffusion.N_step 100 \
    --triangle_attention torch \
    --triangle_multiplicative torch \
    --use_msa false \
    --dtype fp32 \
    2>&1 | tee ~/Desktop/ai4s-rna-protein/inference_r002.log

echo ""
echo "r002 completed: $(date)"
echo ""

# Run r003
echo "============================================"
echo "Starting r003 inference: $(date)"
echo "============================================"
cd "${PROTENIX_DIR}"
python3 runner/inference.py \
    --model_name protenix_base_default_v0.5.0 \
    --seeds 101 \
    --dump_dir "./output_minimal" \
    --input_json_path ~/Desktop/ai4s-rna-protein/input_r003.json \
    --model.N_cycle 5 \
    --sample_diffusion.N_sample 1 \
    --sample_diffusion.N_step 100 \
    --triangle_attention torch \
    --triangle_multiplicative torch \
    --use_msa false \
    --dtype fp32 \
    2>&1 | tee ~/Desktop/ai4s-rna-protein/inference_r003.log

echo ""
echo "All inference completed: $(date)"
echo ""

# Show results
echo "=== Output Summary ==="
for sample in r001 r002 r003; do
    echo "--- ${sample} ---"
    find "${OUTPUT_DIR}/${sample}" -name "*.cif" 2>/dev/null || echo "  No CIF files"
done

echo ""
echo "Packaging submission..."
cd ~/Desktop/ai4s-rna-protein
python3 scripts/package_submission.py \
    --output_dir "${OUTPUT_DIR}" \
    --submission_path ~/Desktop/ai4s-rna-protein/output.zip

echo ""
echo "=== DONE ==="
echo "Submission: ~/Desktop/ai4s-rna-protein/output.zip"
echo "Completed: $(date)"
