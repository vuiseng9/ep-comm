ngpu ?= $(shell nvidia-smi -L | wc -l)
torchrun_intra = torchrun --standalone --nproc-per-node
dbg ?= 0
xargs +=
nsys_cmd ?=
nsys_args ?=
prof_step ?= 
prof_name ?=
prof_dir ?= ./nsys-prof
logs_dir ?= ./logs
log_cmd ?=
rep ?= 
ol_step ?= 55
q3_step ?= 28

ifeq ($(log),1)
log_cmd = 2>&1 | tee $(logs_dir)/$(logname).log
endif

# disable logging by design
ifeq ($(prof),1)
define nsys_cmd
mkdir -p $(prof_dir)
nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -t cuda,nvtx,cudnn,cublas \
  -o $(prof_dir)/$(prof_name)-step_$(prof_step) \
  --force-overwrite true
endef
nsys_args = -nsys-start $(prof_step) -nsys-end $(prof_step)
log_cmd =
endif

# nvidia pytorch docker image work out of the box, not installation needed.
# recent version of nsys installed is also one of the reasons to use the image.
# do install torch >= 2.11 yourself if not using pytorch docker image above
install-dep:
	pip install transformers datasets typer debugpy

purge-prof-dir:
	rm -rf $(prof_dir)

purge-log-dir:
	rm -rf $(logs_dir)

clear-output: purge-prof-dir purge-log-dir

bench-all:
	$(MAKE) bench-all-olmoe
	$(MAKE) bench-all-qwen3

bench-all-olmoe:
	$(MAKE) 100-bench-olmoe-dp-only
	$(MAKE) 105-bench-olmoe-ep-host-nccl
	$(MAKE) 107-bench-olmoe-ep-naive_symm
	$(MAKE) 108-bench-olmoe-ep-pooled-symm
	$(MAKE) 109-bench-olmoe-ep-zerocopy-symm

bench-all-qwen3:
	$(MAKE) 200-bench-qwen3-dp-only
	$(MAKE) 205-bench-qwen3-ep-host-nccl
	$(MAKE) 207-bench-qwen3-ep-naive_symm
	$(MAKE) 208-bench-qwen3-ep-pooled-symm
	$(MAKE) 209-bench-qwen3-ep-zerocopy-symm

do-prof-analyze-all: prof-all analyze-all

prof-all:
	$(MAKE) prof-all-olmoe
	$(MAKE) prof-all-qwen3

prof-all-olmoe:
	$(MAKE) prof-100-olmoe-dp-only
	$(MAKE) prof-105-olmoe-ep-host-nccl
	$(MAKE) prof-107-olmoe-ep-naive_symm
	$(MAKE) prof-108-olmoe-ep-pooled-symm
	$(MAKE) prof-109-olmoe-ep-zerocopy-symm

prof-all-qwen3:
	$(MAKE) prof-200-qwen3-dp-only
	$(MAKE) prof-205-qwen3-ep-host-nccl
	$(MAKE) prof-207-qwen3-ep-naive_symm
	$(MAKE) prof-208-qwen3-ep-pooled-symm
	$(MAKE) prof-209-qwen3-ep-zerocopy-symm

analyze-all:
	$(MAKE) analyze-all-olmoe
	$(MAKE) analyze-all-qwen3

analyze-all-olmoe:
	$(MAKE) __analyze-nsys-rep rep=$(prof_dir)/prof-100-olmoe-dp-only-step_$(ol_step).nsys-rep
	$(MAKE) __analyze-nsys-rep rep=$(prof_dir)/prof-105-olmoe-ep-host-nccl-step_$(ol_step).nsys-rep
	$(MAKE) __analyze-nsys-rep rep=$(prof_dir)/prof-107-olmoe-ep-naive_symm-step_$(ol_step).nsys-rep
	$(MAKE) __analyze-nsys-rep rep=$(prof_dir)/prof-108-olmoe-ep-pooled-symm-step_$(ol_step).nsys-rep
	$(MAKE) __analyze-nsys-rep rep=$(prof_dir)/prof-109-olmoe-ep-zerocopy-symm-step_$(ol_step).nsys-rep

analyze-all-qwen3: 
	$(MAKE) __analyze-nsys-rep rep=$(prof_dir)/prof-200-qwen3-dp-only-step_$(q3_step).nsys-rep
	$(MAKE) __analyze-nsys-rep rep=$(prof_dir)/prof-205-qwen3-ep-host-nccl-step_$(q3_step).nsys-rep
	$(MAKE) __analyze-nsys-rep rep=$(prof_dir)/prof-207-qwen3-ep-naive_symm-step_$(q3_step).nsys-rep
	$(MAKE) __analyze-nsys-rep rep=$(prof_dir)/prof-208-qwen3-ep-pooled-symm-step_$(q3_step).nsys-rep
	$(MAKE) __analyze-nsys-rep rep=$(prof_dir)/prof-209-qwen3-ep-zerocopy-symm-step_$(q3_step).nsys-rep

# usage: make analyze-nsys-rep rep=<path_to_nsys_report>
__analyze-nsys-rep:
	nsys stats --report nvtx_gpu_proj_trace --timeunit msec --format csv $(rep) --output $(rep) --force-export=true --force-overwrite=true
	nsys stats --report nvtx_gpu_proj_sum   --timeunit msec --format csv $(rep) --output $(rep) --force-overwrite=true
	grep ^Range $(rep)_nvtx_gpu_proj_sum.csv | tee $(rep).keyranges.csv
	grep tr-    $(rep)_nvtx_gpu_proj_sum.csv | tee -a $(rep).keyranges.csv
	grep step_  $(rep)_nvtx_gpu_proj_sum.csv | tee -a $(rep).keyranges.csv
	grep '\.fw' $(rep)_nvtx_gpu_proj_sum.csv | tee -a $(rep).keyranges.csv
	grep 'w\.'  $(rep)_nvtx_gpu_proj_sum.csv | tee -a $(rep).keyranges.csv
	
# ──────────────────────
# -lbperf, forcing balanced expert load for benchmarking/profiling
___bench_dist_train_moe___:
	DBG_ATTACH=$(dbg) \
		$(nsys_cmd) \
			$(torchrun_intra) $(ngpu) \
				train_gpt_moe_dp_ep.py \
					-bf16 \
					-lbperf \
					-m $(model) -ep-backend $(ep) \
					-gbs $(shell echo $$(($(mbs) * $(ngpu)))) \
					-ptick $(ptick) -skip-eval \
					-max-step $(max_steps) \
					$(nsys_args)

# ──────────────────────
__bench_olmoe__:
	mkdir -p $(logs_dir)
	$(MAKE) ___bench_dist_train_moe___ model=olmoe-1b-7b \
				mbs=16 max_steps=100 ptick=10 \
				$(log_cmd)

100-bench-olmoe-dp-only:
	$(MAKE) __bench_olmoe__ log=1 logname=$@ ep=local 

prof-100-olmoe-dp-only:
	$(MAKE) 100-bench-olmoe-dp-only \
		prof_name=$@ prof=1 prof_step=$(ol_step) 

105-bench-olmoe-ep-host-nccl:
	$(MAKE) __bench_olmoe__ log=1 logname=$@ ep=host_nccl 

prof-105-olmoe-ep-host-nccl:
	$(MAKE) 105-bench-olmoe-ep-host-nccl \
		prof_name=$@ prof=1 prof_step=$(ol_step) 

107-bench-olmoe-ep-naive_symm:
	$(MAKE) __bench_olmoe__ log=1 logname=$@ ep=naive_symm

prof-107-olmoe-ep-naive_symm:
	$(MAKE) 107-bench-olmoe-ep-naive_symm \
		prof_name=$@ prof=1 prof_step=$(ol_step) 

108-bench-olmoe-ep-pooled-symm:
	$(MAKE) __bench_olmoe__ log=1 logname=$@ ep=pooled_symm

prof-108-olmoe-ep-pooled-symm:
	$(MAKE) 108-bench-olmoe-ep-pooled-symm \
		prof_name=$@ prof=1 prof_step=$(ol_step) 

109-bench-olmoe-ep-zerocopy-symm:
	$(MAKE) __bench_olmoe__ log=1 logname=$@ ep=zerocopy_symm

prof-109-olmoe-ep-zerocopy-symm:
	$(MAKE) 109-bench-olmoe-ep-zerocopy-symm \
		prof_name=$@ prof=1 prof_step=$(ol_step) 

# ──────────────────────
__bench_qwen3__:
	mkdir -p $(logs_dir)
	$(MAKE) ___bench_dist_train_moe___ model=qwen3-30b-a3b \
				mbs=4 max_steps=100 ptick=10 \
				$(log_cmd)

200-bench-qwen3-dp-only:
	mkdir -p $(logs_dir)
	$(MAKE) __bench_qwen3__ log=1 logname=$@ ep=local 

prof-200-qwen3-dp-only:
	$(MAKE) 200-bench-qwen3-dp-only \
		prof_name=$@ prof=1 prof_step=$(q3_step)

205-bench-qwen3-ep-host-nccl:
	$(MAKE) __bench_qwen3__ log=1 logname=$@ ep=host_nccl 

prof-205-qwen3-ep-host-nccl:
	$(MAKE) 205-bench-qwen3-ep-host-nccl \
		prof_name=$@ prof=1 prof_step=$(q3_step)

207-bench-qwen3-ep-naive_symm:
	mkdir -p $(logs_dir)
	$(MAKE) __bench_qwen3__ log=1 logname=$@ ep=naive_symm

prof-207-qwen3-ep-naive_symm:
	$(MAKE) 207-bench-qwen3-ep-naive_symm \
		prof_name=$@ prof=1 prof_step=$(q3_step)

208-bench-qwen3-ep-pooled-symm:
	mkdir -p $(logs_dir)
	$(MAKE) __bench_qwen3__ log=1 logname=$@ ep=pooled_symm

prof-208-qwen3-ep-pooled-symm:
	$(MAKE) 208-bench-qwen3-ep-pooled-symm \
		prof_name=$@ prof=1 prof_step=$(q3_step)

209-bench-qwen3-ep-zerocopy-symm:
	mkdir -p $(logs_dir)
	$(MAKE) __bench_qwen3__ log=1 logname=$@ ep=zerocopy_symm

prof-209-qwen3-ep-zerocopy-symm:
	$(MAKE) 209-bench-qwen3-ep-zerocopy-symm \
		prof_name=$@ prof=1 prof_step=$(q3_step)

# ──────────────────────
__dev_dist_train_moe__:
	DBG_ATTACH=$(dbg) \
		$(nsys_cmd) \
			$(torchrun_intra) $(ngpu) \
				train_gpt_moe_dp_ep.py \
					-m dev -bf16 \
					-ep-backend $(ep) \
					$(nsys_args) \
					$(xargs)

# append prof=1 prof_step=5 to turn on nsys profiling
000-dev-dp-only:
	$(MAKE) __dev_dist_train_moe__ ep=local xargs=-lbloss

005-dev-ep-host-nccl:
	$(MAKE) __dev_dist_train_moe__ ep=host_nccl xargs=-lbbias

007-dev-ep-naive-symm:
	$(MAKE) __dev_dist_train_moe__ ep=naive_symm xargs=-lbperf

008-dev-ep-pooled-symm:
	$(MAKE) __dev_dist_train_moe__ ep=pooled_symm xargs=-lbperf

009-dev-ep-zerocopy-symm:
	$(MAKE) __dev_dist_train_moe__ ep=zerocopy_symm xargs=-lbperf

dev-prof:
	$(MAKE) 000-dev-dp-only prof=1 prof_name=$@ prof_step=5