ngpu ?= $(shell nvidia-smi -L | wc -l)
torchrun_intra = torchrun --standalone --nproc-per-node
dbg ?= 0
xargs +=
nsys_cmd ?=
nsys_args ?=
base_step ?= 
end_step ?=
rep_name ?=
prof_dir ?= ./nsys-prof
bench_dir ?= ./nsys-bench
logs_dir ?= ./logs
log_cmd ?=
rep ?= 
ol_step ?= 55
q3_step ?= 28

ifeq ($(log),1)
log_cmd = 2>&1 | tee $(logs_dir)/$(logname).log
endif

# disable logging by design
ifneq ($(filter 1,$(prof) $(bench)),)
  ifeq ($(bench),1)
    _ := $(shell mkdir -p $(bench_dir))
    end_step = $(shell echo $$(($(base_step) + 25)))
    out_path = $(bench_dir)/$(rep_name)-step_$(base_step)-$(end_step)
  else
    _ := $(shell mkdir -p $(prof_dir))
    end_step = $(base_step)
    out_path = $(prof_dir)/$(rep_name)-step_$(base_step)
  endif
  define nsys_cmd
DBG_ATTACH=$(dbg) \
	nsys profile \
		--capture-range=cudaProfilerApi \
		--capture-range-end=stop \
		-t cuda,nvtx,cudnn,cublas \
		-o $(out_path) \
		--force-overwrite true
  endef
  nsys_args = -nsys-start $(base_step) -nsys-end $(end_step)
  log_cmd =
endif

# nvidia pytorch docker image work out of the box, not installation needed.
# recent version of nsys installed is also one of the reasons to use the image.
# do install torch >= 2.11 yourself if not using pytorch docker image above
install-dep:
	pip install transformers datasets typer nvtx debugpy

clear-output: purge-prof-dir purge-log-dir purge-bench-dir

purge-bench-dir:
	rm -rf $(bench_dir)

purge-prof-dir:
	rm -rf $(prof_dir)

purge-log-dir:
	rm -rf $(logs_dir)

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

do-bench-prof-analyze-all: bench-all postprocess-nsys-bench prof-all

bench-all:
	$(MAKE) bench-all-olmoe
	$(MAKE) bench-all-qwen3

bench-all-olmoe:
	$(MAKE) bench-100-olmoe-dp-only
	$(MAKE) bench-105-olmoe-ep-host-nccl
	$(MAKE) bench-107-olmoe-ep-naive_symm
	$(MAKE) bench-108-olmoe-ep-pooled-symm
	$(MAKE) bench-109-olmoe-ep-zerocopy-symm

bench-all-qwen3:
	$(MAKE) bench-200-qwen3-dp-only
	$(MAKE) bench-205-qwen3-ep-host-nccl
	$(MAKE) bench-207-qwen3-ep-naive_symm
	$(MAKE) bench-208-qwen3-ep-pooled-symm
	$(MAKE) bench-209-qwen3-ep-zerocopy-symm

# analyze every .nsys-rep file found in prof_dir
postprocess-nsys-bench:
	find $(bench_dir) -name '*.nsys-rep' | sort | while read rep; do \
		$(MAKE) __get-stats-keyranges__ rep=$$rep; \
	done
	python process_keyranges.py

# usage: make get-stats-keyranges rep=<path_to_nsys_report>
__get-stats-keyranges__:
	nsys stats --report nvtx_gpu_proj_trace --timeunit msec --format csv $(rep) --output $(rep) --force-export=true --force-overwrite=true
	nsys stats --report nvtx_gpu_proj_sum   --timeunit msec --format csv $(rep) --output $(rep) --force-overwrite=true
	grep ^Range $(rep)_nvtx_gpu_proj_sum.csv | tee $(rep).keyranges.csv
	grep tr-    $(rep)_nvtx_gpu_proj_sum.csv | tee -a $(rep).keyranges.csv
	grep step_  $(rep)_nvtx_gpu_proj_sum.csv | tee -a $(rep).keyranges.csv
	grep '\.fw' $(rep)_nvtx_gpu_proj_sum.csv | tee -a $(rep).keyranges.csv
	grep 'w\.'  $(rep)_nvtx_gpu_proj_sum.csv | tee -a $(rep).keyranges.csv
	
# ──────────────────────
# -lbperf, forcing balanced expert load for benchmarking/profiling
___dist_train_moe___:
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
__train_olmoe__:
	mkdir -p $(logs_dir)
	$(MAKE) ___dist_train_moe___ model=olmoe-1b-7b \
				mbs=16 max_steps=100 ptick=10 \
				$(log_cmd)

100-olmoe-dp-only:
	$(MAKE) __train_olmoe__ log=1 logname=$@ ep=local 

prof-100-olmoe-dp-only:
	$(MAKE) 100-olmoe-dp-only \
		rep_name=$@ prof=1 base_step=$(ol_step) 

bench-100-olmoe-dp-only:
	$(MAKE) 100-olmoe-dp-only \
		rep_name=$@ bench=1 base_step=$(ol_step)

105-olmoe-ep-host-nccl:
	$(MAKE) __train_olmoe__ log=1 logname=$@ ep=host_nccl 

prof-105-olmoe-ep-host-nccl:
	$(MAKE) 105-olmoe-ep-host-nccl \
		rep_name=$@ prof=1 base_step=$(ol_step) 

bench-105-olmoe-ep-host-nccl:
	$(MAKE) 105-olmoe-ep-host-nccl \
		rep_name=$@ bench=1 base_step=$(ol_step)

107-olmoe-ep-naive_symm:
	$(MAKE) __train_olmoe__ log=1 logname=$@ ep=naive_symm

prof-107-olmoe-ep-naive_symm:
	$(MAKE) 107-olmoe-ep-naive_symm \
		rep_name=$@ prof=1 base_step=$(ol_step) 

bench-107-olmoe-ep-naive_symm:
	$(MAKE) 107-olmoe-ep-naive_symm \
		rep_name=$@ bench=1 base_step=$(ol_step)

108-olmoe-ep-pooled-symm:
	$(MAKE) __train_olmoe__ log=1 logname=$@ ep=pooled_symm

prof-108-olmoe-ep-pooled-symm:
	$(MAKE) 108-olmoe-ep-pooled-symm \
		rep_name=$@ prof=1 base_step=$(ol_step) 

bench-108-olmoe-ep-pooled-symm:
	$(MAKE) 108-olmoe-ep-pooled-symm \
		rep_name=$@ bench=1 base_step=$(ol_step)

109-olmoe-ep-zerocopy-symm:
	$(MAKE) __train_olmoe__ log=1 logname=$@ ep=zerocopy_symm

prof-109-olmoe-ep-zerocopy-symm:
	$(MAKE) 109-olmoe-ep-zerocopy-symm \
		rep_name=$@ prof=1 base_step=$(ol_step) 

bench-109-olmoe-ep-zerocopy-symm:
	$(MAKE) 109-olmoe-ep-zerocopy-symm \
		rep_name=$@ bench=1 base_step=$(ol_step)

# ──────────────────────
__train_qwen3__:
	mkdir -p $(logs_dir)
	$(MAKE) ___dist_train_moe___ model=qwen3-30b-a3b \
				mbs=4 max_steps=100 ptick=10 \
				$(log_cmd)

200-qwen3-dp-only:
	mkdir -p $(logs_dir)
	$(MAKE) __train_qwen3__ log=1 logname=$@ ep=local 

prof-200-qwen3-dp-only:
	$(MAKE) 200-qwen3-dp-only \
		rep_name=$@ prof=1 base_step=$(q3_step)

bench-200-qwen3-dp-only:
	$(MAKE) 200-qwen3-dp-only \
		rep_name=$@ bench=1 base_step=$(q3_step)

205-qwen3-ep-host-nccl:
	$(MAKE) __train_qwen3__ log=1 logname=$@ ep=host_nccl 

prof-205-qwen3-ep-host-nccl:
	$(MAKE) 205-qwen3-ep-host-nccl \
		rep_name=$@ prof=1 base_step=$(q3_step)

bench-205-qwen3-ep-host-nccl:
	$(MAKE) 205-qwen3-ep-host-nccl \
		rep_name=$@ bench=1 base_step=$(q3_step)

207-qwen3-ep-naive_symm:
	mkdir -p $(logs_dir)
	$(MAKE) __train_qwen3__ log=1 logname=$@ ep=naive_symm

prof-207-qwen3-ep-naive_symm:
	$(MAKE) 207-qwen3-ep-naive_symm \
		rep_name=$@ prof=1 base_step=$(q3_step)

bench-207-qwen3-ep-naive_symm:
	$(MAKE) 207-qwen3-ep-naive_symm \
		rep_name=$@ bench=1 base_step=$(q3_step)

208-qwen3-ep-pooled-symm:
	mkdir -p $(logs_dir)
	$(MAKE) __train_qwen3__ log=1 logname=$@ ep=pooled_symm

prof-208-qwen3-ep-pooled-symm:
	$(MAKE) 208-qwen3-ep-pooled-symm \
		rep_name=$@ prof=1 base_step=$(q3_step)

bench-208-qwen3-ep-pooled-symm:
	$(MAKE) 208-qwen3-ep-pooled-symm \
		rep_name=$@ bench=1 base_step=$(q3_step)

209-qwen3-ep-zerocopy-symm:
	mkdir -p $(logs_dir)
	$(MAKE) __train_qwen3__ log=1 logname=$@ ep=zerocopy_symm

prof-209-qwen3-ep-zerocopy-symm:
	$(MAKE) 209-qwen3-ep-zerocopy-symm \
		rep_name=$@ prof=1 base_step=$(q3_step)

bench-209-qwen3-ep-zerocopy-symm:
	$(MAKE) 209-qwen3-ep-zerocopy-symm \
		rep_name=$@ bench=1 base_step=$(q3_step)

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

# append prof=1 base_step=5 to turn on nsys profiling
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
	$(MAKE) 000-dev-dp-only prof=1 rep_name=$@ base_step=5