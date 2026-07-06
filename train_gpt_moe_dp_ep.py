import os
import random
import time
import typer
from functools import partial

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from dataset.tinystories import get_tinystories_loaders

from models import GptMoE
from transformers import AutoTokenizer
from models.token_shuffler import EPBackend

from utils import yellow, cyan, green
from utils import rank_print as print_
from utils import Meter
from profiler import NsysTrainProfiler

# disable tf32 compute, critical for gradient accumulation equivalence
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

app = typer.Typer(pretty_exceptions_enable=False,
                  help="GptMoE pretraining (intra-node, multi-gpu, dp and/or ep)")

model_def = {
    "dev":           dict(D=64,   Dff=128,  H=8,  G=1, S=512,  E=16,  K=2), # for fast dev iteration
    "olmoe-1b-7b":   dict(D=2048, Dff=1024, H=16, G=1, S=4096, E=64,  K=8), # allenai/OLMoE-1B-7B-0125
    "qwen3-30b-a3b": dict(D=2048, Dff=6144, H=16, G=4, S=4096, E=128, K=16) # Qwen/Qwen3-30B-A3B
}


@app.command()
def main(
    model_arch: str = typer.Option("dev", "-m", help=f"model arch: {set(model_def.keys())}"),
    bf16: bool = typer.Option(False, "-bf16", help="use bf16 for training"),
    seed: int = typer.Option(704, "-seed", help="random seed for reproducibility"),
    ep_backend: str = typer.Option("local", "-ep-backend", help=f"backend for expert parallelism {EPBackend.choices()}"),
    use_lbloss: bool = typer.Option(False, "-lbloss", help="penalize for load balancing"),
    use_lbbias: bool = typer.Option(False, "-lbbias", help="biasing router for load balancing"),
    use_lbperf: bool = typer.Option(False, "-lbperf", help="perf related paths, including force load balancing by directly overriding router decisions with statistically even distribution"),
    gbs: int = typer.Option(64, "-gbs", help="global batch size"),
    lr: float = typer.Option(5e-3, "-lr", help="learning rate"),
    max_steps: int = typer.Option(1000, "-max-step", help="stop training after this many training (optimizer) steps"),
    max_epochs: int = typer.Option(None, "-max-epoch", help="maximum number of epochs to train"),
    ptick: int = typer.Option(100, "-ptick", help="print period in training steps"),
    valtick: int = typer.Option(200, "-valtick", help="validation period in training steps)"),
    skip_eval: bool = typer.Option(False, "-skip-eval", help="skip evaluation during training"),
    # Nsys profiling options
    nsys_start: int = typer.Option(None, "-nsys-start", help="the global step id (zero-indexed) to start nsys profiling, default to none for no nsys profiling"),
    nsys_end: int = typer.Option(None, "-nsys-end", help="the global step id (zero-indexed) to end nsys profiling, default to none for no nsys profiling")
):

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    
    if os.environ.get("EP_BACKEND", "nccl").lower() == "nccl":
        dist.init_process_group(backend="nccl", device_id=rank)
    else:
        opts = dist.ProcessGroupNCCL.Options()
        opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
        dist.init_process_group(backend="nccl", device_id=rank, pg_options=opts)

    world_size = dist.get_world_size()
    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    DEVICE = f"cuda:{rank}"

    ARCH = model_arch
    USE_BF16 = bf16
    USE_MOE = True
    EP_BACKEND = ep_backend
    assert ARCH in model_def, f"Invalid model architecture: {ARCH}. Available options: {set(model_def.keys())}"
    assert EP_BACKEND in EPBackend.choices(), f"Invalid EP backend: {EP_BACKEND}. Available options: {EPBackend.choices()}"
    USE_EP = EPBackend(EP_BACKEND) != EPBackend.LOCAL

    USE_LBL = use_lbloss
    USE_LBB = use_lbbias
    USE_LBFORCE = use_lbperf
    PERF_RUN = use_lbperf
    LBGAMMA = 0.1
    LBL_COEF = 1.5

    PTICK = ptick # print period
    VALTICK = valtick # validation period
    SEP = '───'
    
    GBS = gbs
    MBS = GBS // world_size # evenly split GBS by number of gpu, n_gpu = n_dp ranks, each rank gets MBS
    N_GA = 1   # do not support gradient accumulation for now.
    LR = lr
    MAX_EPOCHS = max_epochs
    MAX_STEPS = max_steps
    SKIP_EVAL = skip_eval

    NSYS_START = nsys_start
    NSYS_END = nsys_end

    assert GBS%MBS==0, "GBS must be divisible by MBS"

    assert (MAX_EPOCHS is None) != (MAX_STEPS is None), \
    "MAX_EPOCHS and MAX_STEPS are mutually exclusive, set one of them to None. "
    "We recommend MAX_STEPS for large dataset, and MAX_EPOCHS for small dataset"

    profiler = NsysTrainProfiler(start=NSYS_START, end=NSYS_END, 
                                 train_step_label=f"ep={EP_BACKEND} @ {ARCH}")
    if profiler.enabled:
        if MAX_STEPS is not None:
            print_(f"WARNING: -max-step is set {MAX_STEPS} but will be overridden by -nsys-end {NSYS_END}")
        MAX_STEPS = NSYS_END + 1  # stop training right after profiling window closes
        SKIP_EVAL = True # for skipping evaluation

    if USE_MOE:
        if not USE_LBL and not USE_LBB and not USE_LBFORCE:
            # we do need to test when both lb penalty or lb biasing are turned off
            pass
        else:
            assert USE_LBL ^ USE_LBB ^ USE_LBFORCE, "-lbloss, -lbbias, and -lbforce are mutually exclusive, and only works with moe"

    # ── Tokenizer & Data Loaders ─────────────────────────────────────────────────────────────────────

    model_cfg = model_def[ARCH]

    # TOKENIZER_ID = 'gpt2'
    # TOKENIZER_ID = 'google/byt5-small'
    # TOKENIZER_ID = "meta-llama/Llama-2-7b-hf" # need hf login
    TOKENIZER_ID = "mistralai/Mistral-7B-v0.1" # 30K vocab
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    train_loader, val_loader, train_sampler = get_tinystories_loaders(
        tokenizer=tokenizer, batch_size=MBS, ctx_len=model_cfg['S'], n_row=250_000)
    
    V = len(tokenizer)    
    step_per_ep = len(train_loader)
    step_per_val = len(val_loader)

    if MAX_STEPS is None:
        MAX_STEPS = step_per_ep * MAX_EPOCHS

    # ── Model ────────────────────────────────────────────────────────────────────────────────────────
    model_dtype = torch.bfloat16 if USE_BF16 else torch.float32

    if USE_MOE:
        model_cfg['V'] = V
        model_cfg['L'] = 1 # less confusion in nsys profiling
        model_cfg['EP'] = world_size if USE_EP else 1
        model_cfg['ep_backend'] = EP_BACKEND
        model_cfg['lbgamma'] = LBGAMMA if USE_LBB else 0.0
        model_cfg['lbforce'] = USE_LBFORCE
        
        model = GptMoE(**model_cfg)

        if USE_EP: 
            # configure DDP to disable gradient reduction for expert parameters
            expert_params = [n for n, _ in model.named_parameters() if ".moe.mlp" in n]
            model._ddp_params_and_buffers_to_ignore = expert_params
    else:
        raise NotImplementedError

    model.to(model_dtype).to(DEVICE)
    model = DDP(model, device_ids=[rank])
    model_dtype = next(model.parameters()).dtype
    
    print_(model)

    if USE_EP is True:
        print_(f"- DP + EP -, disable grad reduce for expert params: \n\t{model.parameters_to_ignore}")
        
        # Non-expert params are replicated across all DP ranks → count once.
        # Expert params are sharded (each rank owns E//world_size experts) →
        # sum the local counts across ranks to get the true global total.
        ignored = set(model.module._ddp_params_and_buffers_to_ignore)
        non_expert_params, local_expert_params = 0, 0
        for n, p in model.module.named_parameters():
            if p.requires_grad:
                if n in ignored:
                    local_expert_params += p.numel()
                else:
                    non_expert_params += p.numel()
        expert_params_total = local_expert_params * world_size  # symmetric sharding
        trainable_params = non_expert_params + expert_params_total
        print_(f"model dtype: {model_dtype},\n\ttrainable n_params: {trainable_params:,} "
                   f"(non-expert: {non_expert_params:,} + expert total: {expert_params_total:,} "
                   f"= {world_size} ranks * {local_expert_params:,} (local experts))")
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_(f"| DP only | model dtype: {model_dtype}, trainable n_params: {trainable_params:,}")
    
    vocab_emb_params = model.module.token_embed.weight.numel()

    print_(f"- {vocab_emb_params:,} (vocab emb) = {trainable_params-vocab_emb_params:,} params")

    # ── Optimizer ────────────────────────────────────────────────────────────────────────────────────────

    # reduction='sum' so gradient-accumulation stays *exactly* equivalent under masking (see note)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # ── Train loop ────────────────────────────────────────────────────────────────────────────────────────

    model_generate = partial(model.module.generate, tokenizer, 
                             prompts=['Once upon a t'], max_new_tokens=16, T=0.5)

    def val_and_gen(model, global_step, gen=True):
        was_training = model.training
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['input_ids'][:, :-1].to(DEVICE)
                y = batch['labels'][:, 1:].to(DEVICE)
                logits = model(x)
                loss = criterion(logits.reshape(-1, V), y.reshape(-1))
                n_valid = (y != -100).sum().item()
                total_loss += loss.item()
                total_tokens += n_valid
        # Aggregate across all ranks before computing the scalar loss
        stats = torch.tensor([total_loss, total_tokens], device=DEVICE, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, total_tokens = stats[0].item(), stats[1].item()

        val_loss = total_loss / max(total_tokens, 1)
        bpb = val_loss / torch.log(torch.tensor(2.0)).item()
        print_(f"{SEP} val @ {global_step:13,} steps *** val loss: {green(f'{val_loss:.4f}')} {SEP} bpb: {bpb:.4f}")
        if gen:
            print_(f"{SEP} gen @ {SEP*4} {yellow(model_generate())}\n")
        if was_training:
            model.train()

    print_(f"\n{model.module.extra_repr()}")
    print_(f"{step_per_ep} steps/epoch, {step_per_val} steps/val, " \
           f"{GBS} gbs, {MBS} mbs, {MAX_STEPS} train steps, skip_val={SKIP_EVAL}")

    epoch = 0
    global_step = 0
    step_loss   = torch.zeros((), device=DEVICE, dtype=model_dtype)
    step_celoss = torch.zeros((), device=DEVICE, dtype=model_dtype)
    step_lbloss = torch.zeros((), device=DEVICE, dtype=model_dtype)
    elapsed_ms  = Meter()

    model.train()
    train_iter = iter(train_loader)
    while True:
        with profiler.step(global_step) as prof_step:
            with prof_step.range('full'):
                t0 = time.perf_counter()

                optimizer.zero_grad(set_to_none=True)

                step_loss.zero_()
                step_celoss.zero_()
                step_lbloss.zero_()

                # ----------------------------------------------- get batch
                with prof_step.range('data'):
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        epoch += 1
                        train_sampler.set_epoch(epoch)
                        train_iter = iter(train_loader)
                        batch = next(train_iter)

                    x = batch["input_ids"][:, :-1].to(DEVICE)
                    y = batch["labels"][:, 1:].to(DEVICE)
                    # total valid (non-padded) targets across the whole global batch
                    n_valid_token = (y != -100).sum().clamp(min=1)

                    mb_x = x.chunk(N_GA, dim=0)
                    mb_y = y.chunk(N_GA, dim=0)

                for i in range(N_GA):
                    # ----------------------------------------------- Forward pass 
                    with prof_step.range('fwd'):
                        
                        logits = model(mb_x[i])
                        ce_loss = criterion(logits.reshape(-1, V), mb_y[i].reshape(-1)) / n_valid_token 
                        # loss must be scaled for gradient accumulation # TODO
                        
                        if USE_MOE and USE_LBL:
                            lbloss = LBL_COEF * torch.stack([l.moe.layer_lbloss for l in model.module.layers]).sum()
                            loss = ce_loss + lbloss
                            step_celoss += ce_loss.detach()
                            step_lbloss += lbloss.detach()
                        else:
                            loss = ce_loss

                    # ----------------------------------------------- Backward pass 
                    with prof_step.range('bwd'):
                        loss.backward()
                        step_loss += loss.detach()

                # ----------------------------------------------- Optimizer step (weight update) 
                with prof_step.range('opt'):
                    if not PERF_RUN: # guarantee optimizer stepping for profiling purpose.
                        # -- Gradient Checks
                        # grad * min(1, max_norm/global grad L2norm)
                        # it means if a grad has its norm exceeding max_norm, rescale max_norm/grad_norm
                        preclip_gradnorm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                        if not torch.isfinite(preclip_gradnorm):
                            # nan and inf will show up in the calculation
                            # gonna skip update if nan/inf exist
                            optimizer.zero_grad(set_to_none=True) # no needed because duplicated, but putting here for brevity.
                            continue
                    
                    optimizer.step() # gradient descent

                # ----------------------------------------------- Logging
                if ((global_step+1) % PTICK) == 0:
                    # Loss Aggregation 
                    dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)
                    if USE_MOE and USE_LBL:
                        dist.all_reduce(step_celoss, op=dist.ReduceOp.AVG)
                        dist.all_reduce(step_lbloss, op=dist.ReduceOp.AVG)
                    
                    # Expert Load
                    load_str = ''
                    moeloss_str = ''
                    if USE_MOE:
                        # NOTE: report rank-local load instead of global aggregated load
                        expert_load = torch.stack([l.moe.load_per_expert for l in model.module.layers]).mean(dim=0)
                        load_str = f"{SEP} "
                        load_str += "DP+EP" if USE_EP else "DP-only"
                        load_str += f", {world_size} ws, {model.module.EPR} epr, e{model.module.E}.k{model.module.K}, rank-local routing % (load, first 8): " + \
                                    ', '.join(list(map(lambda x: f"{x*100:5.2f}", expert_load[:8].tolist())))

                        if USE_LBL:
                            moeloss_str = f"= {step_celoss:.4f} (ce) + {step_lbloss:.4f} (lb)"
                        if USE_LBB:
                            moeloss_str = f", lbgamma={LBGAMMA}"
                        if USE_LBFORCE:
                            moeloss_str = f", force load balance"

                    # close to nvsmi mem usage value
                    # mem = torch.cuda.max_memory_allocated() / 1e9
                    free, total = torch.cuda.mem_get_info()
                    nvsmi_used = (total - free) / 1e9

                    elapsed_ms.update((time.perf_counter() - t0) * 1000)

                    print_(f"{SEP*3} {global_step+1:13,} steps, {(global_step+1)/step_per_ep:.3f} ep "
                           f"{SEP} {elapsed_ms.avg:5.1f} ms, {nvsmi_used:.1f} gb "
                           f"{SEP} {cyan(f'{step_loss:.4f}')} (loss) {moeloss_str}"
                           f"\n\t\t\t{load_str} \n")

                    if global_step+1 == 10:
                        elapsed_ms.reset() # assume 10 steps are enough of warm up.

                if (global_step+1) % VALTICK == 0 and not SKIP_EVAL:
                    val_and_gen(model, global_step+1)

        global_step += 1
        if global_step >= MAX_STEPS:
            print_(f"Training ends. ")
            if not profiler.enabled and not SKIP_EVAL:
                print_(f"Final evaluation ...")
                val_and_gen(model, global_step)
            break

    print_("Tearing down...")
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    DBG_ATTACH = int(os.environ.get("DBG_ATTACH", "0")) == 1
    if DBG_ATTACH and ((int(os.environ.get("RANK", "0")) == 0)):
        import debugpy
        debugpy.listen(("127.0.0.1", 5678))
        print('Waiting for debugger attach...', flush=True)
        debugpy.wait_for_client()
    app()
    
    