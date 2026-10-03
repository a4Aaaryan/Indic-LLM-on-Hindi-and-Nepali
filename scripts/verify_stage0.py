"""Stage 0 verification script.

Validates that directory structures, configuration files,
common utilities, and environment settings are correctly established.
"""

import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.config import get_repo_root, resolve_path, load_yaml_config
from common.utils import setup_logger, set_seed


def main():
    logger = setup_logger("verify_stage0")
    logger.info("Running Stage 0 validation checks...")

    # 1. Verify Repo Root
    repo_root = get_repo_root()
    logger.info("Repository Root: %s", repo_root)
    assert (repo_root / ".git").exists(), "Git directory not found at repo root"
    assert (repo_root / ".gitignore").exists(), ".gitignore missing"
    assert (repo_root / "requirements.txt").exists(), "requirements.txt missing"
    assert (repo_root / "README.md").exists(), "README.md missing"
    assert (repo_root / "report" / "phase1_report.md").exists(), "Phase 1 report template missing"

    # 2. Verify Directory Structure
    required_dirs = [
        "hindi/data/raw",
        "hindi/data/interim",
        "hindi/data/cleaned",
        "hindi/data/splits",
        "hindi/tokenizer/train",
        "hindi/tokenizer/models",
        "hindi/tokenizer/vocab",
        "hindi/notebooks",
        "hindi/scripts",
        "hindi/configs",
        "nepali/data/raw",
        "nepali/data/interim",
        "nepali/data/cleaned",
        "nepali/data/splits",
        "nepali/tokenizer/train",
        "nepali/tokenizer/models",
        "nepali/tokenizer/vocab",
        "nepali/notebooks",
        "nepali/scripts",
        "nepali/configs",
        "common",
        "report/figures",
        "report/tables",
    ]

    for rel_dir in required_dirs:
        dir_path = resolve_path(rel_dir)
        assert dir_path.exists() and dir_path.is_dir(), f"Required directory missing: {rel_dir}"

    logger.info("All %d required directories verified successfully.", len(required_dirs))

    # 3. Verify YAML Configurations
    hi_config = load_yaml_config("hindi/configs/phase1_config.yaml")
    assert hi_config["language"]["code"] == "hi", "Hindi config language mismatch"
    assert hi_config["targets"]["target_training_tokens"] == 500000000, "Hindi target token mismatch"

    ne_config = load_yaml_config("nepali/configs/phase1_config.yaml")
    assert ne_config["language"]["code"] == "ne", "Nepali config language mismatch"
    assert ne_config["targets"]["target_training_tokens"] == 500000000, "Nepali target token mismatch"

    logger.info("Configuration files verified (Hindi & Nepali).")

    # 4. Verify Script placeholders
    script_stages = [
        "01_collect_sources.py",
        "02_inspect_raw.py",
        "03_preprocess.py",
        "04_deduplicate.py",
        "05_split_data.py",
        "06_train_tokenizer.py",
        "07_generate_statistics.py",
    ]

    for lang in ["hindi", "nepali"]:
        for s in script_stages:
            script_path = resolve_path(f"{lang}/scripts/{s}")
            assert script_path.exists(), f"Script missing: {lang}/scripts/{s}"

    logger.info("All pipeline scripts verified for both languages.")

    # 5. Verify Random Seed
    set_seed(42)
    logger.info("Seed configuration verified.")

    logger.info("Stage 0 environment and directory validation completed successfully!")


if __name__ == "__main__":
    main()
