.PHONY: help env-dev env-train env-serve tree-train tree-all lock lint format test test-gpu clean

# 默认目标
help: ## 显示帮助信息
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ==================== 环境管理 ====================
env-dev: ## 安装开发环境(含dev+train依赖)
	uv sync --group dev --group train --frozen
	@echo "✅ Dev environment ready. Activate: source .venv/bin/activate"

env-train: ## 仅安装训练依赖(无serve/export)
	uv sync --group train --frozen

env-serve: ## 仅安装推理服务依赖(生产用)
	uv sync --group serve --frozen --no-dev

tree-train: ## 查看训练依赖树
	uv tree --group train

tree-all: ## 查看所有依赖组的依赖树
	uv tree --all-groups

lock: ## 更新 lockfile (添加新依赖后执行)
	uv lock
	@echo "🔒 Lock file updated. Run 'uv sync' to apply."

lock-check: ## CI中检查lockfile是否过期
	uv lock --check

# ==================== 代码质量 ====================
lint: ## Ruff 静态检查
	uv run ruff check src/ tests/

format: ## Ruff 格式化
	uv run ruff format src/ tests/

fix: ## 自动修复可修复的lint问题 + 格式化
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

# ==================== 测试 ====================
test: ## 运行单元测试(CPU only, 快速)
	uv run pytest tests/unit/ -v --cov=src --cov-report=term-missing -m "not gpu"

test-gpu: ## 运行GPU相关测试(⚠️ 4GB显存注意OOM)
	uv run pytest tests/ -v -m "gpu"

test-all: ## 全量测试
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

# ==================== 清理 ====================
clean: ## 清理缓存和临时文件
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .coverage htmlcov wandb/
	@echo "🧹 Cleaned."