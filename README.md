# Global Macro Database (Python)

Independent Python implementation of the Global Macro Database pipeline, aligned to Stata do/ado semantics but not dependent on the Stata repository runtime.

## Installation

```bash
pip install -e .
```

## Main APIs

```python
import global_macro_data as gmd
```

- `gmd.download_source(source, ...)`: download one supported raw source.
- `gmd.clean_source(source, ...)`: clean one source into GMD clean format.
- `gmd.rebuild_clean_sources(...)`: batch clean sources (from `source_list.csv` or explicit list).
- `gmd.combine_variable(varname, ...)`: build one final chainlinked variable.
- `gmd.combine_all(...)`: build all combine outputs from bundled pipeline specs.
- `gmd.build_documentation_all(...)`: build documentation outputs.
- `gmd.merge_final_data(...)`: produce `data_final.dta`.
- `gmd.run_master_pipeline(...)`: run initialize/clean/combine/document/final stages.

## Minimal Example

```python
from pathlib import Path
import global_macro_data as gmd

data_root = Path("path/to/data")  # contains raw/helpers
work = Path("path/to/work")

gmd.rebuild_clean_sources(
    data_raw_dir=data_root / "raw",
    data_clean_dir=work / "clean",
    data_helper_dir=data_root / "helpers",
    data_temp_dir=work / "tempfiles",
)

gmd.combine_all(
    data_clean_dir=work / "clean",
    data_temp_dir=work / "tempfiles",
    data_final_dir=work / "final",
)

gmd.merge_final_data(
    data_temp_dir=work / "tempfiles",
    data_final_dir=work / "final",
    data_helper_dir=data_root / "helpers",
)
```
