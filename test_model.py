"""Full test suite for Mytho with all advanced features."""
import torch
from mytho_model import (
    MythoConfig, MythoModel,
    LatentScratchpad, VerifierHead, BranchingController,
    ExpertMetrics, DynamicExpertGrowth,
    SelfConsistencyDecoder, MCDropoutEstimator, SwitchMoELayer,
)

cfg = MythoConfig(
    d_model=256, n_heads=4, d_head=64,
    d_latent_kv=64, d_rope=16, max_depth=4,
    n_experts=4, n_active_experts=2,
    d_expert_ff=512, max_seq_len=64, vocab_size=1000,
)
ids = torch.randint(1, 1000, (2, 32))

# 1. Base model (backward compat)
model = MythoModel(cfg)
out = model(ids, labels=ids)
print(f"[Base]        Loss: {out['loss'].item():.4f}  Depth: {out['mean_depth'].item():.1f}")

# 2. Scratchpad + Verifier + Uncertainty-Driven ACT
model_s = MythoModel(cfg, use_scratchpad=True, d_scratch=64)
out_s = model_s(ids, labels=ids)
print(f"[Scratchpad]  Loss: {out_s['loss'].item():.4f}  "
      f"Conf: {out_s.get('confidence', torch.tensor(0)).item():.4f}  "
      f"Unc: {out_s.get('uncertainty', torch.tensor(0)).item():.4f}")

# 3. Scratchpad + Branching
model_b = MythoModel(cfg, use_scratchpad=True, d_scratch=64, use_branching=True, n_branches=2)
out_b = model_b(ids, labels=ids)
print(f"[Branching]   Loss: {out_b['loss'].item():.4f}  Depth: {out_b['mean_depth'].item():.1f}")

# 4. Standalone scratchpad
sp = LatentScratchpad(256, 64)
scratch = sp.init_scratch(2, 32, ids.device)
h = torch.randn(2, 32, 256)
ctx = sp.read(h, scratch)
scratch2 = sp.write(h, scratch)
print(f"[Scratchpad]  Read: {ctx.shape}  Write: {scratch2.shape}")

# 5. Standalone verifier
vf = VerifierHead(256, 64)
sig = vf(h, scratch)
halt = vf.should_halt(sig)
print(f"[Verifier]    Conf: {sig['confidence'].mean():.4f}  Halt: {halt.sum().item()}/{halt.numel()}")

# 6. Branching controller
bc = BranchingController(256, n_branches=3)
branches = bc.branch(h)
scores = [torch.rand(2, 32, 1) for _ in range(3)]
selected = bc.select_hard(branches, scores)
disagree = bc.branch_disagreement(branches)
print(f"[Branching]   Branches: {len(branches)}  Disagreement: {disagree.mean():.6f}")

# 7. Expert metrics
em = ExpertMetrics(4)
logits = torch.randn(2, 32, 4)
topk_i = logits.topk(2, dim=-1).indices
em.update(logits, topk_i)
em.update(logits, topk_i)
report = em.compute(model.blocks[0].moe)
print(f"[ExpertMetr]  Entropy: {report['expert_entropy']:.4f}  "
      f"Stability: {report['routing_stability']:.4f}  "
      f"Similarity: {report['mean_expert_similarity']:.4f}")

# 8. Dynamic expert growth
deg = DynamicExpertGrowth()
actions = deg.step(model.blocks[0].moe, em)
print(f"[ExpertGrow]  Actions: {actions}")

# 9. Generation with scratchpad model
gen = model_s.generate(ids[:1, :8], max_new_tokens=16)
print(f"[Generate]    Shape: {gen.shape}")

# 10. Core modules still work
sc = SelfConsistencyDecoder(model, n_paths=2)
unc = MCDropoutEstimator(model, n_samples=2)
sw = SwitchMoELayer(cfg)
print(f"[Core]        All core modules OK")

# 11. Optional agent scaffolding (imported from submodules)
from mytho_model.reflexion import ReflexionController
from mytho_model.react import ReActController
rc = ReflexionController(cfg)
react = ReActController(model)
react.register_tool("echo", lambda x: x)
print(f"[Scaffolding] ReAct + Reflexion importable OK")

print("\n ALL TESTS PASSED")

