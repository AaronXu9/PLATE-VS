"""Quick smoke test for the full model pipeline.

Run from project root:
    python3 -m benchmarks.06_binding_affinity_model.smoke_test
    OR: python3 benchmarks/06_binding_affinity_model/smoke_test.py
"""
import sys
import os
import torch

# Ensure project root paths
_this = os.path.abspath(__file__) if "__file__" in dir() else os.path.abspath("benchmarks/06_binding_affinity_model/smoke_test.py")
_script_dir = os.path.dirname(_this)
_project_root = os.path.dirname(os.path.dirname(_script_dir))

# Critical: insert our package dir LAST so it becomes sys.path[0]
# (sys.path.insert(0, ...) pushes previous entries down)
for p in [
    os.path.join(_project_root, "external", "GEMS"),
    _script_dir,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(_project_root)

from torch.utils.data import DataLoader  # noqa: E402
from data.build_dataset import load_dataset  # noqa: E402
from data.collate import custom_collate  # noqa: E402
from model.binding_affinity_model import BindingAffinityModel  # noqa: E402
from model.losses import CompositeLoss  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dataset = load_dataset("data/pdbbind_cleansplit/binding_affinity_dataset/processed/casf2016_data.pt")
print(f"Loaded {len(dataset)} samples")

loader = DataLoader(dataset[:8], batch_size=4, shuffle=False, num_workers=0, collate_fn=custom_collate)

model = BindingAffinityModel(
    esm_dim=1088, proj_dim=64, et_layers=2, et_heads=4,
    et_rbf=32, et_cutoff=5.0, cross_attn_layers=1,
    cross_attn_heads=4, dropout=0.1,
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {n_params:,}")

criterion = CompositeLoss()
batch = next(iter(loader))
batch = batch.to(device)

print(f"Input: z={batch.z.shape}, pos={batch.pos.shape}, prot={batch.prot_padded.shape}, y={batch.y.shape}")
pred = model(batch)
print(f"Output: {pred.shape}, values={[round(v, 4) for v in pred.detach().cpu().flatten().tolist()]}")

loss, ld = criterion(pred, batch.y.unsqueeze(-1))
print(f"Loss: { {k: round(v, 4) for k, v in ld.items()} }")

loss.backward()
gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
print(f"Grad norm: {gn:.4f}")
print("\nFull model smoke test PASSED")
