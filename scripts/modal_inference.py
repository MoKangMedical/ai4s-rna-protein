"""
Protenix RNA-Protein Complex Inference on Modal GPU
Usage: modal run scripts/modal_inference.py
"""
import modal
import os

app = modal.App("protenix-rna-inference")

# Build image with Protenix dependencies
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0", 
        "torchaudio==2.4.0",
        "scipy>=1.9.0",
        "ml_collections==1.1.0",
        "tqdm==4.67.1",
        "pandas",
        "dm-tree==0.1.9",
        "PyYAML==6.0.2",
        "matplotlib",
        "biopython",
        "biotite",
        "modelcif",
        "gemmi",
        "rdkit",
        "scikit-learn",
        "pydantic>=2.0.0",
        "optree",
        "protobuf",
        "icecream",
        "fair-esm==2.0.0",
        "numpy==1.26.4",
        "pdbeccdutils",
    )
    .apt_install("wget", "git")
)

# Volume for model weights and CCD cache
volume = modal.Volume.from_name("protenix-models", create_if_missing=True)

PROTENIX_DIR = "/root/Protenix"
MODEL_DIR = "/root/Protenix/af3-dev/release_model"
CCD_DIR = "/root/Protenix/release_data/ccd_cache"
OUTPUT_DIR = "/root/output"


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": volume},
    timeout=7200,  # 2 hours max
    memory=32768,  # 32GB RAM
)
def run_inference():
    import subprocess
    import json
    import shutil
    
    # Clone Protenix if not cached
    if not os.path.exists(PROTENIX_DIR):
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/bytedance/Protenix.git", PROTENIX_DIR],
            check=True,
        )
    
    # Copy model weights from volume
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CCD_DIR, exist_ok=True)
    
    model_src = "/data/protenix_base_default_v0.5.0.pt"
    model_dst = f"{MODEL_DIR}/protenix_base_default_v0.5.0.pt"
    if os.path.exists(model_src) and not os.path.exists(model_dst):
        shutil.copy(model_src, model_dst)
        print(f"Copied model weights to {model_dst}")
    
    # Upload test data
    test_data = [
        {
            "name": "r001",
            "sequences": [
                {"proteinChain": {"sequence": "SEREELKALLDEYEQAMKELMKYKNQLLALERGTDLYDPEFAKYLKELLKLTEEYLNKILKKLKELIENSKDPLIQALLSVKGVGPITAAYLYAYVDLTKATSASALWAYLGIDKPSHKRYKKGEAGGGNKKLRTAVWNQARSMIKRRDSPYRKVYLEEKKRLSKSKKVTKSRNTQGELVKVKWSKAKPSHKHGAALRAVMKTFLADVWFVGHKIAGLPTRPLYVGIVDPEKRGFKY", "count": 2}},
                {"rnaSequence": {"sequence": "GCUUUUUUAUCGAAGAUGACUCCAAAGGCAGAAC", "count": 1}}
            ]
        },
        {
            "name": "r002",
            "sequences": [
                {"proteinChain": {"sequence": "GEEEVSLTESGGGLVPTGGSTTLSCKVTGAYISYYSVHWVAQPPGGPPVYVASISPYSGSTYYHPSVAGRITISRDTSNNTAYLKISNLTPADTATYYCALQGYRREYGRGFAYWGQGTLLTVSDRPPVPPKVYSLPPLCPGVPGSTVTCGCVIYDFFPPPIQPSWNGGSSTSGVVIFPPVLMPNGTYIAAIVRTVSPSTAPSSSWTCNVVHPPSNATKTVTCVPVSC", "count": 2}},
                {"proteinChain": {"sequence": "GVCTLTTSPSSLTAALGDSVTITCTASCPCGSYISWYLQKPGQPPQLLIYDGSTLAPGAPSIFSGSLSGTSYTLTITGLTKEYFGTYWCRCSYSFPSTFGQGTKVEPKAPPVPPQVSLLPPSPELIEEGKVIVVCVVLSYYPRDVTITWSLDGQTLSSSPQTSETEYNPTDLTYSLASFLVIPTEEWKKHTKITCNVNHSALPEPQTLSFEVGKC", "count": 2}},
                {"rnaSequence": {"sequence": "GGUGAUUCUCAGGUACAGCUAGUCUCAACUGUGAGGCGUAUACUCCUGAUGCAUGGAGGCAGUACGCUGUAAUGAGUAACUUUACAAUGAAACUAAUUGCCCUAGGUGUUCGCU", "count": 2}}
            ]
        },
        {
            "name": "r003",
            "sequences": [
                {"proteinChain": {"sequence": "ASRRFRRELELVLKKPVAPQITSGSGKLFIVVTAAKLLSSDIYKPYSENSKVAFVITDSEKDAEKLVEAFKNTLPLPVGRLLDLENAKKISEEEIKELLDKYRIFVATPEVMEYLFKEGKIDLKNIALIVFDDTSEAKEGSPFKEIMELIKKAGSKPLVIADDDDLLKGNPDPETLRARVAWLEELAGAPALAPRDLTNLEGDKNRPKIELKKSIPPEDKDGSSLLPLLLYLLALLLLLLGPGPLCPPALCPTRELLEWLRAALENRFFLGRYLADKILEEALERLREPLPGGLHLVRLLLLLVLLLVATVVYSILKSRIDPENPGLYDLEPLIKWLYEVLFENPEGFGEKKQNKIGIIFTRTVLAGVVLNRLIKELGKVDEKIKHVSSNFFGCSPLGPRRPDPEFEKECEQKLKEVIEKMKNGESDILILGPLELKNIDFPNANLALSIYKLDSPEEYFLALSYIKSKNGYLTYFDLGENFEEIKEFLKKVEALEEVLRRYARGEERPGDEYPEPVRDYSDIAPLGPETPDAGFEPVGEENLEEAWEELAKDYPTHKYDDTAPEPVLEKSPDGKWKCTVEVPNSWPIRETIEPEPQPDPYLCRYRAIVELANRLREAGYLGEGLRCVLGRGEEEGERPLDVDPDGIDLPGAPGSPTRRVAVPKAVPEALRDSAPVPGQPWHLYRIGFKLTEPYPPELNKDNLPVRRFEDHPERVGIITSKPIPPVTPFPVYTESGPVEVSIEDISTGREISEEELELIHEFHKWIFEYVLRLTPPNLVYDPERAEESFVVVPLKPTGNGDKVLLNLEFIEKFKKSPTAKGLPKEERPPEKPFVFDESVYKDAVLIPEYRNPENPELFEVVEVLPDLTCDSPFPSPEYSNFIEYYKKKYNIECNAKNVPLVVARKIAVDHNYLVPRYNLQILCPCVTSLFPLSSHLARCIDCLPSILHMLRDLLLAREVRDRLLEGSGPSDLDILKALTSREANYPFDNEEYEILGDAFLDFAVRTYLYLTNPPGHIADLLRRARDIVSPAALTRLGRKWGLAEKMIYDPFDPPTNWLPPGFVRVPPDPWTEQLIPDDLLADTVESLIGAVYKAEGEEAAIRAAHKLGVPVVPSVPSPEYQELLDLAKYELEKLKEVLNYEFKNPSLFLEAFTTPSYYFSKIARDYGRLDFLGDGIFEYIVAKKVFESPKKLSPVELKAVTKALTSNKLAAEIAAKNNLHKFLRTLSPKEFEIVLKAVKEIKKKKSVKDPYVLGDVFYAVIGAIYLDSGNNLEEVEKVIEPLIKPYIEELISKKVKIPRLEVLEKYPEGLVEECKDDELEGKCRTTSNVVGKGKFKGEGTSREDSLDAALGRLLEEE", "count": 1}},
                {"rnaSequence": {"sequence": "GCCCCGGGCAGGGGUGCCGUCCGCUGCGGGAGAAAGGUCGCGCAUCUCUCCGGGGCGC", "count": 1}}
            ]
        }
    ]
    
    # Save test data
    input_json = f"{PROTENIX_DIR}/competition_input.json"
    with open(input_json, "w") as f:
        json.dump(test_data, f, indent=2)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run inference
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROTENIX_DIR}:{env.get('PYTHONPATH', '')}"
    env["LAYERNORM_TYPE"] = "torch"
    
    cmd = [
        "python3", f"{PROTENIX_DIR}/runner/inference.py",
        "--model_name", "protenix_base_default_v0.5.0",
        "--seeds", "101",
        "--dump_dir", OUTPUT_DIR,
        "--input_json_path", input_json,
        "--model.N_cycle", "10",
        "--sample_diffusion.N_sample", "5",
        "--sample_diffusion.N_step", "200",
        "--triangle_attention", "torch",
        "--triangle_multiplicative", "torch",
        "--use_msa", "false",
        "--dtype", "bf16",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=False)
    
    print(f"\nInference completed with exit code: {result.returncode}")
    
    # Collect results
    print("\n=== Output files ===")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            filepath = os.path.join(root, f)
            size = os.path.getsize(filepath)
            print(f"  {filepath} ({size/1024:.1f} KB)")


@app.local_entrypoint()
def main():
    run_inference.remote()
