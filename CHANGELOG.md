
---

## CHANGELOG.md

```markdown
# Changelog

## [0.1.0] - 2026-07-29

### Added

- 初始版本发布
- 实现 JSONL 数据加载，支持脏数据自动跳过
- 实现 BIO 联合标注体系生成（触发词 + 论元角色）
- 实现子词对齐，处理特殊 token 忽略逻辑
- 实现 RoBERTa + CRF 模型结构，支持 Encoder/Head 可插拔
- 实现训练器与实体级 F1 评估指标
- 实现 Top 10 高频事件筛选
- 实现 YAML + OmegaConf 配置管理
- 添加 25 个单元测试，覆盖数据处理、对齐、批处理、评估等核心模块
