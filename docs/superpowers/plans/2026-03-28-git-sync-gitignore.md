# Git Sync with Remote — Fix .gitignore for Phosphorus Outputs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `.gitignore` to cover `_generated_smiles_zinc_phosphorus` output directories, then push the branch to remote.

**Architecture:** The current `.gitignore` pattern `external/DeepCoy/*_generated_smiles_zinc` does not match the 4 untracked directories ending in `_generated_smiles_zinc_phosphorus`. A single pattern addition fixes the gap. After that, commit and push.

**Tech Stack:** Git, bash

---

### Task 1: Fix .gitignore to Cover Phosphorus Output Directories

**Files:**
- Modify: `.gitignore`

**Context:**

The 4 untracked directories are:
```
external/DeepCoy/2025-12-12-12-10-42_3582111_generated_smiles_zinc_phosphorus
external/DeepCoy/2025-12-12-12-10-45_2225391_generated_smiles_zinc_phosphorus
external/DeepCoy/2025-12-12-12-36-29_2405952_generated_smiles_zinc_phosphorus
external/DeepCoy/2025-12-12-12-56-36_2409717_generated_smiles_zinc_phosphorus
```

The existing `.gitignore` line `external/DeepCoy/*_generated_smiles_zinc` does NOT match these because they end with `_phosphorus`.

- [ ] **Step 1: Add pattern for phosphorus directories**

In `.gitignore`, find the DeepCoy section:
```
# DeepCoy generated outputs
external/DeepCoy/*_generated_smiles_zinc
external/DeepCoy/*_params_zinc.json
```

Replace it with:
```
# DeepCoy generated outputs
external/DeepCoy/*_generated_smiles_zinc
external/DeepCoy/*_generated_smiles_zinc_*
external/DeepCoy/*_params_zinc.json
external/DeepCoy/*_params_zinc_*.json
```

The `*_generated_smiles_zinc_*` pattern covers any element-specific variants (phosphorus, nitrogen, etc.) generically.

- [ ] **Step 2: Verify the 4 directories are now ignored**

Run:
```bash
git status
```

Expected: The 4 `_generated_smiles_zinc_phosphorus` directories should NO LONGER appear as untracked. Output should show only `.gitignore` as modified:
```
 M .gitignore
```

If they still appear, double-check the pattern was saved correctly.

- [ ] **Step 3: Commit the .gitignore update**

```bash
git add .gitignore
git commit -m "chore: ignore phosphorus variant of DeepCoy generated SMILES outputs"
```

Expected output:
```
[feature/deepcoy-decoy-generation <hash>] chore: ignore phosphorus variant of DeepCoy generated SMILES outputs
 1 file changed, 2 insertions(+)
```

---

### Task 2: Push Branch to Remote

**Files:** (none modified — git operation only)

- [ ] **Step 1: Confirm branch tracking**

Run:
```bash
git branch -vv
```

Expected: `* feature/deepcoy-decoy-generation ... [origin/feature/deepcoy-decoy-generation]` — confirms remote tracking is set.

- [ ] **Step 2: Push to remote**

Run:
```bash
git push
```

Expected:
```
Enumerating objects: 5, done.
...
   9e7f46d...<new_hash>  feature/deepcoy-decoy-generation -> feature/deepcoy-decoy-generation
```

If the push is rejected with "non-fast-forward", run `git pull --rebase` first, then re-push.

- [ ] **Step 3: Verify sync**

Run:
```bash
git log --oneline -3 && git status
```

Expected: `nothing to commit, working tree clean` and the latest commit visible in log.

---
