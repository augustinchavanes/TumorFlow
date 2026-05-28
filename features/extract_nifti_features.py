from pathlib import Path
import runpy


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    runpy.run_path(str(project_root / "extract_nifti_features.py"), run_name="__main__")
