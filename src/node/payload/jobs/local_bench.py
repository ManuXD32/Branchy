"""Job: time a whole synthetic model on this device alone (single-device baselines).

args: {"model": "branched"|"sequential", "kwargs": {...}, "prompts": [32], "decodes": [16], "runs": 3,
       "threads": 4, "linear": "gemm"|"mv", "kernels": "native"}
"""
import json
import sys

args = json.load(sys.stdin)
import torch  # noqa: E402

torch.set_num_threads(int(args.get('threads', 4)))
torch.set_grad_enabled(False)
import cpu_kernels  # noqa: E402
import stacks  # noqa: E402
import cluster_model  # noqa: E402

cpu_kernels.install_kernels(args.get('kernels', 'native'))
stacks.set_linear_mode(args.get('linear', 'gemm'))
kwargs = args.get('kwargs', {})
model = cluster_model.Model.synthetic_sequential(**kwargs) if args.get('model') == 'sequential' \
    else cluster_model.Model.synthetic(**kwargs)
rows = cluster_model.time_local(model, args.get('prompts', [32]), args.get('decodes', [16]), int(args.get('runs', 3)))
print(json.dumps(dict(label=model.label, parameters=model.parameters(), threads=torch.get_num_threads(),
                      linear=stacks.LINEAR_MODE, kernels=args.get('kernels', 'native'), rows=rows)))
