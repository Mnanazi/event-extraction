"""数据处理器单元测试 (Python 3.12+)"""

import json
from pathlib import Path

import pytest
from src.data.loader import load_duee_file
from src.data.processor import build_joint_bio_labels, process_and_save

# ==================== Fixtures ====================


@pytest.fixture
def sample_event() -> dict:
    return {
        "text": "消失的外企光环5月份在华裁员900余人",
        "id": "test_001",
        "event_list": [
            {
                "event_type": "组织关系-裁员",
                "trigger": "裁员",
                "trigger_start_index": 12,
                "arguments": [
                    {"argument_start_index": 14, "role": "裁员人数", "argument": "900余人"},
                    {"argument_start_index": 7, "role": "时间", "argument": "5月份"},
                ],
            }
        ],
    }


@pytest.fixture
def valid_types() -> set[str]:
    return {"组织关系-裁员"}


@pytest.fixture
def dirty_jsonl(tmp_path: Path) -> Path:
    """创建包含脏数据的 JSONL 文件"""
    fp = tmp_path / "dirty.jsonl"
    lines = [
        '{"text": "good line", "id": "1", "event_list": []}',
        "{bad json here",  # 脏数据
        "",  # 空行
        '{"text": "another good", "id": "2", "event_list": []}',
        '{"text": "truncated',  # 截断JSON
    ]
    fp.write_text("\n".join(lines), encoding="utf-8")
    return fp


# ==================== 测试用例 ====================


class TestJointBioLabels:
    def test_basic_label_generation(self, sample_event: dict, valid_types: set[str]) -> None:
        text = sample_event["text"]
        labels = build_joint_bio_labels(text, sample_event["event_list"], valid_types)

        assert len(labels) == len(text)
        # 验证触发词
        assert labels[12] == "B-Trigger_组织关系-裁员"
        assert labels[13] == "I-Trigger_组织关系-裁员"
        # 验证论元
        assert labels[7] == "B-Arg_时间"
        assert labels[8] == "I-Arg_时间"
        assert labels[9] == "I-Arg_时间"
        assert labels[14] == "B-Arg_裁员人数"

    def test_trigger_overrides_argument_on_overlap(self) -> None:
        """当触发词与论元位置重叠时，触发词优先"""
        text = "公司宣布裁员计划"
        event_list = [
            {
                "event_type": "组织关系-裁员",
                "trigger": "裁员",
                "trigger_start_index": 4,
                "arguments": [{"argument_start_index": 4, "role": "动作", "argument": "裁员计划"}],
            }
        ]
        labels = build_joint_bio_labels(text, event_list, {"组织关系-裁员"})
        # 触发词应覆盖论元
        assert labels[4] == "B-Trigger_组织关系-裁员"
        assert labels[5] == "I-Trigger_组织关系-裁员"

    def test_invalid_event_type_ignored(self, sample_event: dict) -> None:
        labels = build_joint_bio_labels(
            sample_event["text"], sample_event["event_list"], {"无关类型"}
        )
        assert all(l == "O" for l in labels)  # noqa: E741

    def test_out_of_range_indices_handled(self) -> None:
        """索引超出文本长度时不应崩溃"""
        text = "短文本"
        event_list = [
            {
                "event_type": "T1",
                "trigger": "超长触发词",
                "trigger_start_index": 2,
                "arguments": [],
            }
        ]
        labels = build_joint_bio_labels(text, event_list, {"T1"})
        assert len(labels) == len(text)
        assert labels[2] == "B-Trigger_T1"
        assert all(l == "O" for i, l in enumerate(labels) if i != 2)


class TestDataLoader:
    def test_dirty_jsonl_skips_errors(self, dirty_jsonl: Path) -> None:
        samples = list(load_duee_file(dirty_jsonl))
        assert len(samples) == 2
        assert samples[0]["id"] == "1"
        assert samples[1]["id"] == "2"

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            list(load_duee_file("/nonexistent/path.json"))


class TestProcessAndSave:
    def test_filters_non_top10_events(self, tmp_path: Path) -> None:
        input_file = tmp_path / "input.jsonl"
        records = [
            {
                "text": "有裁员事件",
                "id": "1",
                "event_list": [
                    {
                        "event_type": "组织关系-裁员",
                        "trigger": "裁员",
                        "trigger_start_index": 1,
                        "arguments": [],
                    }
                ],
            },
            {
                "text": "无关事件",
                "id": "2",
                "event_list": [
                    {
                        "event_type": "灾害-地震",
                        "trigger": "地震",
                        "trigger_start_index": 0,
                        "arguments": [],
                    }
                ],
            },
        ]
        input_file.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
            encoding="utf-8",
        )

        output_file = tmp_path / "output.jsonl"
        stats = process_and_save(input_file, output_file, {"组织关系-裁员"})

        assert stats["kept"] == 1
        assert stats["dropped"] == 1
        results = list(load_duee_file(output_file))
        assert len(results) == 1
        assert results[0]["id"] == "1"
