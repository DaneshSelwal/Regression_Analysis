# Regression Uncertainty Quantification Analysis - Applied Changes

The following changes have been implemented across the repository:

- [x] **Removed `CONTRIBUTING.md`**: File deleted as requested.
- [x] **Git Author Configuration**: Configured author to `claude <claude@anthropic.com>`.
- [x] **TypeError Fix**: Fixed `_ensure_excel_file(filename=excel_file_path)` to `_ensure_excel_file(excel_file_path)` across all notebooks.
- [x] **URL Corrections**: 
    - Fixed missing `/` in `git+https://github.com/DaneshSelwal/treeffuser.git`.
    - Fixed missing `/` in `git clone https://github.com/jjbrophy47/ibug.git`.
- [x] **Import Fixes**: Corrected `dataLoader` to `DataLoader` in `torch.utils.data` imports and usages.
- [x] **Slicing Integer Error**: Wrapped `t / 2` with `int(t / 2)` in `conformal_predictions_online_super` to avoid `TypeError: slice indices must be integers`.
- [x] **ACP Indentation Fix**: Repaired the corrupted single-line code block in `Adaptive_Coverage_Policies(ACP).ipynb` and other relevant notebooks, restoring proper Python indentation and formatting.
- [x] **HCM Notebook Restarts**: Added `os._exit(0)` to pip installation cells in HCM notebooks for automatic runtime restart.
- [x] **HCM Feature Logic**: Replaced dynamic/automatic feature name selection with standard manual entry logic consistent with other notebooks.
- [x] **Comprehensive Repository Scan**: Verified consistency of these fixes across all `examples/` subdirectories.

Completed by: claude
Date: 2026-05-06
