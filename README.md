# Cat 3D Refinement Pipeline

Refines a raw Hunyuan3D mesh using a 4-stage pipeline:

```
Hunyuan3D output (.obj/.glb)
    ↓
A  Mesh Cleanup       — remove fragments, remesh, Laplacian smooth
    ↓
B  Vision Audit       — render 6 views, diff vs real photos via Claude vision
    ↓
C  Texture Refinement — displacement mapping + PBR map baking
    ↓
D  QA Loop            — silhouette Chamfer error, iterate until < 5px
    ↓
Final mesh + textures + report
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

## Usage

```bash
python pipeline.py \
  --mesh  ./hunyuan_output.obj \
  --photos ./ref_photos/ \
  --output ./output/ \
  --iterations 3
```

## Output

```
output/
├── stage_a_clean.obj          # cleaned mesh
├── stage_c_displaced.obj      # geometry-refined mesh
├── final_mesh.obj             # final output
├── textures/
│   ├── albedo.png
│   ├── roughness.png
│   └── normal.png
├── renders_b/                 # renders used for audit
│   ├── front.png
│   └── ...
├── audit.json                 # Claude's discrepancy report
└── pipeline_report.json       # full summary
```

## Swapping Components

| Stage | What to swap | Where |
|---|---|---|
| Vision model | Change `claude-opus-4-5` in `b_audit.py` | `stages/b_audit.py` |
| Displacement map source | Replace `_generate_displacement_map()` with ControlNet output | `stages/c_refine.py` |
| Normal map source | Replace `_bake_normal_map()` with SD ControlNet Normal | `stages/c_refine.py` |
| Renderer | `pyrender` auto-detected; fallback is orthographic silhouette | `utils/renderer.py` |
| QA threshold | Change `ERROR_THRESHOLD_PX` | `stages/d_qa.py` |

## Notes on Displacement Maps

The current `_generate_displacement_map()` uses luminance as a depth proxy.
For best results, replace it with the output of a ControlNet Depth model:

```python
# In stages/c_refine.py, replace _generate_displacement_map() with:
def _generate_displacement_map(reference_photo_path, size=1024):
    # Call your ControlNet Depth API / local model here
    # Return a (size, size) float32 array in [0, 1]
    ...
```
