# Post-Reorganization TODO

## ✅ Completed
- [x] Created new directory structure (`iguane_original/` and `iguane_2d/`)
- [x] Moved all files to appropriate locations
- [x] Created README.md for `iguane_original/`
- [x] Created README.md for `iguane_2d/`
- [x] Updated main README.md
- [x] Created REORGANIZATION.md with complete file mapping

## ⏳ Action Items Required

### 1. Update Import Statements

Several Python files need their import paths updated:

#### iguane_original/all_in_one.py
Current: `import pipeline`
Update to: `from . import pipeline` or ensure it's run from the correct directory

#### iguane_original/pipeline.py
Current: `from harmonization.model_architectures import Generator`
Update to: `from .harmonization.model_architectures import Generator`

#### iguane_2d/evaluation/evaluate_cyclegan_results.py
Current: `from train_fetal_2d_cyclegan import build_2d_generator`
Update to: `from ..training.train_fetal_2d_cyclegan import build_2d_generator`

### 2. Test Scripts

Test each major script in its new location:

```bash
# Test IGUANe Original
cd iguane_original
# Test import paths work
python -c "import pipeline"
python -c "from harmonization import model_architectures"

# Test IGUANe 2D
cd iguane_2d
cd training
python -c "import train_fetal_2d_cyclegan"
cd ../evaluation
python -c "from training import train_fetal_2d_cyclegan"  # This will likely fail, needs fixing
```

### 3. Update Path References in Code

Search for hardcoded paths that reference the old structure:

```bash
# Search for references to old paths
grep -r "preprocessing/" iguane_original/ --include="*.py"
grep -r "\./harmonization" iguane_original/ --include="*.py"
grep -r "processed_data" iguane_2d/ --include="*.py"
grep -r "weights/cyclegan_2d" iguane_2d/ --include="*.py"
grep -r "results/cyclegan_2d" iguane_2d/ --include="*.py"
```

Common paths to check:
- `iguane_original/pipeline.py`: Line 21 - `TEMPLATE_PATH = 'preprocessing/MNI152_T1_1mm_brain.nii.gz'`
  - Update to: `TEMPLATE_PATH = './preprocessing/MNI152_T1_1mm_brain.nii.gz'` or use absolute path
  
- `iguane_original/harmonization/inference.py`: Line 13 - `weights_path = './iguane_weights.h5'`
  - Already correct (relative to harmonization directory)

- `iguane_2d/evaluation/evaluate_cyclegan_results.py`:
  - Line 14: `'processed_data_4slice_fixed/test_4slice_data.pkl'`
  - Line 25: `Path('results/cyclegan_2d/evaluation').mkdir(...)`
  - Lines 41, 43: `weight_file = 'weights/cyclegan_2d/...'`
  - Update these paths based on where data/weights are actually stored

### 4. Update Scripts and Shell Files

#### iguane_2d/run_training.sh
Check if this script has any path references that need updating:
```bash
cat iguane_2d/run_training.sh
```

### 5. Update Documentation Paths

Check if any paths in documentation need updating:
- README files
- Comments in code
- Configuration files

### 6. Git Operations

```bash
# Check git status
git status

# Stage the changes
git add iguane_original/ iguane_2d/
git add README.md REORGANIZATION.md POST_REORGANIZATION_TODO.md

# Remove old locations from git
git rm -r preprocessing/ harmonization/ prediction/ metadata/
git rm *.py *.csv *.sh logs/ harmonized_results/ harmonized_monday/ harmonization_evaluation/

# Commit
git commit -m "Reorganize project: separate IGUANe original and IGUANe 2D implementations

- Create iguane_original/ for published 3D T1-weighted harmonization
- Create iguane_2d/ for 2D fetal brain harmonization (in development)
- Update README with new structure
- Add subdirectory README files for documentation
- Add REORGANIZATION.md with complete file mapping"
```

### 7. Update CI/CD (if applicable)

If you have any CI/CD pipelines, update:
- Test paths
- Build paths
- Deployment paths

### 8. Update External Documentation

If you have external documentation (wiki, docs site, etc.), update:
- Installation instructions
- Usage examples
- File structure diagrams

### 9. Notify Collaborators

If working with others, notify them about:
- New directory structure
- Import path changes
- How to update their local repos
- Any breaking changes

## Testing Checklist

After making the above updates, test:

- [ ] IGUANe Original all_in_one.py works
- [ ] IGUANe Original harmonization inference works
- [ ] IGUANe Original training pipeline works (if applicable)
- [ ] IGUANe 2D training script works
- [ ] IGUANe 2D preprocessing scripts work
- [ ] IGUANe 2D evaluation scripts work
- [ ] IGUANe 2D harmonization inference works
- [ ] All imports resolve correctly
- [ ] Data paths are correct
- [ ] Weight file paths are correct
- [ ] Output directories are created properly

## Notes

- Keep `venv/` in root (don't move it)
- Keep `__pycache__/` as is (gitignored)
- `debug_output.png`, `iguane.png`, `nohup.out` can stay in root or be moved/cleaned up as needed
- Consider adding `.gitignore` entries for output directories if not already present

## Recommended Next Immediate Actions

1. **Fix import in evaluate_cyclegan_results.py** (most critical)
2. **Test the most commonly used scripts**
3. **Update any active development scripts you're currently using**
4. **Commit changes to git**
5. **Test on a clean clone to ensure everything works**

---

Generated: November 1, 2025
