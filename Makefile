# Minecraft 1.8.9 Simulation Makefile
# Target: Vulkan compute shaders via MoltenVK (macOS)

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Directories
SHADER_DIR := cpp/shaders
SPV_DIR := cpp/spirv
BUILD_DIR := build
TEST_DIR := tests

# Tools
GLSLC := glslangValidator
SPIRV_OPT := spirv-opt
PYTHON := python3
PYTEST := pytest

# Find all shaders
COMP_SHADERS := $(wildcard $(SHADER_DIR)/*.comp)
GLSL_SHADERS := $(wildcard $(SHADER_DIR)/*.glsl)
ALL_SHADERS := $(COMP_SHADERS) $(GLSL_SHADERS)

# Output SPIR-V files
SPV_FILES := $(patsubst $(SHADER_DIR)/%.comp,$(SPV_DIR)/%.spv,$(COMP_SHADERS))
SPV_FILES += $(patsubst $(SHADER_DIR)/%.glsl,$(SPV_DIR)/%.spv,$(GLSL_SHADERS))

# Colors for output
GREEN := \033[0;32m
RED := \033[0;31m
YELLOW := \033[0;33m
NC := \033[0m

.PHONY: help check-tools shaders test test-fast test-shaders test-verifiers clean validate-shaders

help:
	@echo "╔═══════════════════════════════════════════════════════════════╗"
	@echo "║       Minecraft 1.8.9 Simulation - Build & Test               ║"
	@echo "╠═══════════════════════════════════════════════════════════════╣"
	@echo "║ make check-tools      - Check required tools are installed    ║"
	@echo "║ make validate-shaders - Validate shader syntax (no compile)   ║"
	@echo "║ make shaders          - Compile shaders to SPIR-V             ║"
	@echo "║ make test             - Run all tests                         ║"
	@echo "║ make test-fast        - Run fast tests (skip GPU/slow)        ║"
	@echo "║ make test-shaders     - Run shader tests only                 ║"
	@echo "║ make test-verifiers   - Run verifier tests only               ║"
	@echo "║ make clean            - Clean build artifacts                 ║"
	@echo "╚═══════════════════════════════════════════════════════════════╝"

check-tools:
	@echo "Checking required tools..."
	@which $(GLSLC) > /dev/null 2>&1 || (echo "$(RED)✗ glslangValidator not found$(NC)" && echo "  Install with: brew install glslang" && exit 1)
	@echo "$(GREEN)✓ glslangValidator$(NC)"
	@which $(PYTHON) > /dev/null 2>&1 || (echo "$(RED)✗ python3 not found$(NC)" && exit 1)
	@echo "$(GREEN)✓ python3$(NC)"
	@$(PYTHON) -c "import pytest" 2>/dev/null || (echo "$(YELLOW)⚠ pytest not installed$(NC)" && echo "  Install with: pip install pytest")
	@echo "$(GREEN)✓ All tools available$(NC)"

# Create SPIR-V output directory
$(SPV_DIR):
	@mkdir -p $(SPV_DIR)

# Compile .comp shaders to SPIR-V
$(SPV_DIR)/%.spv: $(SHADER_DIR)/%.comp | $(SPV_DIR)
	@echo "Compiling $<..."
	@$(GLSLC) -V --target-env vulkan1.2 -o $@ $< 2>&1 || (echo "$(RED)✗ Failed: $<$(NC)" && exit 1)
	@echo "$(GREEN)✓ $@$(NC)"

# Compile .glsl shaders to SPIR-V (requires -S stage flag)
$(SPV_DIR)/%.spv: $(SHADER_DIR)/%.glsl | $(SPV_DIR)
	@echo "Compiling $<..."
	@$(GLSLC) -V -S comp --target-env vulkan1.2 -o $@ $< 2>&1 || (echo "$(RED)✗ Failed: $<$(NC)" && exit 1)
	@echo "$(GREEN)✓ $@$(NC)"

# Validate shader syntax without full compilation
validate-shaders:
	@echo "Validating shader syntax..."
	@errors=0; \
	for shader in $(ALL_SHADERS); do \
		if $(GLSLC) -V --target-env vulkan1.2 -o /dev/null "$$shader" 2>&1; then \
			echo "$(GREEN)✓ $$shader$(NC)"; \
		else \
			echo "$(RED)✗ $$shader$(NC)"; \
			errors=$$((errors + 1)); \
		fi; \
	done; \
	if [ $$errors -gt 0 ]; then \
		echo "$(RED)$$errors shader(s) failed validation$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)All shaders validated successfully$(NC)"

# Compile all shaders
shaders: check-tools $(SPV_FILES)
	@echo "$(GREEN)All shaders compiled to $(SPV_DIR)/$(NC)"

# Run all tests
test:
	@echo "Running all tests..."
	@cd $(CURDIR) && $(PYTEST) $(TEST_DIR)/ -v --tb=short

# Run fast tests (skip slow and GPU tests)
test-fast:
	@echo "Running fast tests..."
	@cd $(CURDIR) && $(PYTEST) $(TEST_DIR)/ -v -m "not slow and not gpu" --tb=short

# Run shader tests only
test-shaders:
	@echo "Running shader tests..."
	@cd $(CURDIR) && $(PYTEST) $(TEST_DIR)/test_shaders.py -v --tb=short

# Run verifier tests only
test-verifiers:
	@echo "Running verifier tests..."
	@cd $(CURDIR) && $(PYTEST) $(TEST_DIR)/test_verifiers.py -v --tb=short

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(SPV_DIR) $(BUILD_DIR) __pycache__ .pytest_cache
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -delete
	@echo "$(GREEN)Clean complete$(NC)"

# List all shaders
list-shaders:
	@echo "Compute shaders (.comp):"
	@for f in $(COMP_SHADERS); do echo "  $$f"; done
	@echo ""
	@echo "GLSL shaders (.glsl):"
	@for f in $(GLSL_SHADERS); do echo "  $$f"; done
	@echo ""
	@echo "Total: $$(echo $(ALL_SHADERS) | wc -w) shaders"

# Summary of verification files
list-verifiers:
	@echo "Verification files:"
	@ls -1 verification/*.py 2>/dev/null | head -20 || echo "  (none found)"
	@echo ""
	@echo "Test data files:"
	@ls -1 verification/*.json 2>/dev/null || echo "  (none found)"
