"""Evidence for docs/LITERATURE.md (GateSkip claim 2). Runs GateSkip's GatedWrapper.update_kv_cache, extracted verbatim
from baselines/GateSkip (pinned 8d56513), on real Hugging Face eager-attention outputs of a tiny random Llama.

Usage: python scripts/check_gateskip_kvcopy.py baselines/GateSkip/experiment/models/gating/GatedWrapper.py
"""
import ast, sys, torch, textwrap
from transformers import LlamaConfig, LlamaForCausalLM, DynamicCache
src = open(sys.argv[1]).read()
fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "update_kv_cache")
ns = {"torch": torch}; exec(textwrap.dedent(ast.get_source_segment(src, fn)), ns)
update_kv_cache = ns["update_kv_cache"]
torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, attn_implementation="eager")
m = LlamaForCausalLM(cfg).eval()
for B in (1, 4):
    ids = torch.randint(0, 64, (B, 5)); cache = DynamicCache()
    with torch.no_grad(): m(ids, past_key_values=cache, use_cache=True)
    layer = 1; attn = m.model.layers[layer].self_attn
    captured = {}
    h = attn.register_forward_hook(lambda mod, a, k, out: captured.update(out=out, kw=k), with_kwargs=True)
    with torch.no_grad(): m(torch.randint(0, 64, (B, 1)), past_key_values=cache, use_cache=True)
    h.remove()
    kw = captured["kw"]; out = captured["out"]
    print(f"B={B}: attn kwargs keys={sorted(k for k in kw if 'past' in k)}; output types={[type(o).__name__ for o in out]}")
    before = [t.clone() for t in (cache.layers[layer].keys, cache.layers[layer].values)]
    skip = torch.ones(B, 1, 1, dtype=torch.bool)  # force 'skip' for every token
    class Self: layer_idx = layer
    for label, past in (("kwargs.get('past_key_value')", kw.get("past_key_value")), ("actual cache object", cache)):
        try: r = update_kv_cache(Self, out, past, skip); res = f"returned {type(r).__name__}"
        except Exception as e: res = f"raised {type(e).__name__}: {str(e)[:90]}"
        after = (cache.layers[layer].keys, cache.layers[layer].values)
        changed = any(not torch.equal(a, b) for a, b in zip(before, after))
        print(f"   past={label}: {res}; layer-{layer} cache changed: {changed}")
print("Simulating transformers 4.48-4.55 semantics (kwarg 'past_key_value', cache[i] -> (k, v)):")
for B in (1, 4, 16):
    ids = torch.randint(0, 64, (B, 5)); cache = DynamicCache()
    with torch.no_grad(): m(ids, past_key_values=cache, use_cache=True)
    captured = {}
    h = m.model.layers[1].self_attn.register_forward_hook(lambda mod, a, k, out: captured.update(out=out), with_kwargs=True)
    with torch.no_grad(): m(torch.randint(0, 64, (B, 1)), past_key_values=cache, use_cache=True)
    h.remove()
    legacy = [(l.keys, l.values) for l in cache.layers]           # what cache[i] returned in 4.5x
    before = [t.clone() for t in legacy[1]]
    class Self: layer_idx = 1
    try: r = update_kv_cache(Self, captured["out"], legacy, torch.ones(B, 1, 1, dtype=torch.bool)); res = f"returned {type(r).__name__}"
    except Exception as e: res = f"raised {type(e).__name__}: {str(e)[:80]}"
    print(f"  B={B}: out[1] is {tuple(captured['out'][1].shape)} attention weights; {res}; "
          f"cache changed: {any(not torch.equal(a, b) for a, b in zip(before, legacy[1]))}")
