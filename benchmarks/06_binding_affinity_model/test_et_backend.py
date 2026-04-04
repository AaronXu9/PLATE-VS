"""Test TorchMD-NET ET backend on GPU."""
import sys, os, torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "external", "GEMS"))
sys.path.insert(0, SCRIPT_DIR)

print(f"torch {torch.__version__}, CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test 1: Import TorchMD_ET
from torchmdnet.models.torchmd_et import TorchMD_ET
print("TorchMD_ET import: OK")

# Test 2: ET forward pass
device = "cuda" if torch.cuda.is_available() else "cpu"
et = TorchMD_ET(
    hidden_channels=64, num_layers=2, num_heads=4,
    num_rbf=32, cutoff_upper=5.0, max_z=100,
).to(device)
z = torch.tensor([6, 7, 8, 6], dtype=torch.long, device=device)
pos = torch.randn(4, 3, device=device)
batch = torch.zeros(4, dtype=torch.long, device=device)
out = et(z, pos, batch)
print(f"ET forward: scalar={out[0].shape}, vector={out[1].shape}")

# Test 3: Full model with ET backend
from torch.utils.data import DataLoader
from data.build_dataset import load_dataset
from data.collate import custom_collate
from model.binding_affinity_model import BindingAffinityModel
from model.losses import CompositeLoss

dataset = load_dataset(os.path.join(
    PROJECT_ROOT, "data/pdbbind_cleansplit/binding_affinity_dataset/processed/casf2016_data.pt"
))
loader = DataLoader(dataset[:4], batch_size=2, collate_fn=custom_collate)

model = BindingAffinityModel(
    esm_dim=1088, proj_dim=64, et_layers=2, et_heads=4,
    et_rbf=32, et_cutoff=5.0, cross_attn_layers=1,
    cross_attn_heads=4, dropout=0.1,
).to(device)

print(f"Backend: {model.ligand_encoder.backend}")
print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

batch_data = next(iter(loader)).to(device)
pred = model(batch_data)
loss, ld = CompositeLoss()(pred, batch_data.y.view(-1, 1))
loss.backward()
gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
print(f"Output: {pred.shape}, loss={ld['total']:.4f}, grad_norm={gn:.4f}")
print("\nTORCHMD-NET ET BACKEND TEST PASSED")
