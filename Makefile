# Top-level orchestration for the navigable-affordance RSA pipeline.
#
# Targets:
#   make install     — pip install -e ".[sam,dev]"
#   make data        — download Net2Brain stimuli (requires net2brain)
#   make masks       — SAM segmentation (GPU/CPU; caches per image)
#   make features    — extract 8-dim affordance vectors
#   make rdm         — build SAM-derived RDM (scene-ordered)
#   make rsa         — standard + partial RSA against OPA/PPA/EVC
#   make ablation    — per-feature partial RSA vs NAM + EVC
#   make all         — masks → features → rdm → rsa → ablation
#   make analysis    — features → rdm → rsa → ablation (masks must exist)
#   make test        — pytest tests/
#   make smoke       — pytest -m "not slow"
#   make clean       — rm -rf outputs/ (keeps data/)
#   make clean-all   — clean + rm -rf __pycache__ .pytest_cache

CONFIG  ?= configs/default.yaml
DEVICE  ?= auto
SAM_MODEL ?= vit_b
CHECKPOINT ?= checkpoints/sam_vit_b_01ec64.pth

.PHONY: install data masks features rdm rsa ablation all analysis test smoke clean clean-all

install:
	pip install -e ".[sam,dev]"

data:
	python -c "from net2brain.datasets import load_dataset; \
	           p, _ = load_dataset('bonner_pnas2017'); \
	           print('stimuli at', p)"

masks:
	affordance-build-masks --config $(CONFIG) \
	    --device $(DEVICE) --sam-model $(SAM_MODEL) \
	    --checkpoint $(CHECKPOINT)

features:
	affordance-extract-features --config $(CONFIG)

rdm:
	affordance-build-rdm --config $(CONFIG)

rsa:
	affordance-rsa --config $(CONFIG)

ablation:
	affordance-ablation --config $(CONFIG)

all: masks features rdm rsa ablation

analysis: features rdm rsa ablation

test:
	pytest tests/ -v

smoke:
	pytest tests/ -v -m "not slow"

clean:
	rm -rf outputs/

clean-all: clean
	rm -rf __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete