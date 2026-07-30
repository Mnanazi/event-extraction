"""项目级自定义类型定义 (Python 3.12+)"""

from typing import Protocol, runtime_checkable

from transformers.tokenization_utils_base import BatchEncoding


@runtime_checkable
class TokenizerProtocol(Protocol):
    """
    项目统一的 Tokenizer 接口契约。
    兼容 AutoTokenizer / PreTrainedTokenizerFast / PreTrainedTokenizer。
    避免 transformers 内部联合类型导致的 Pyright/Pylance 误报。
    """

    def __call__(
        self,
        text: str,
        max_length: int | None = ...,
        truncation: bool = ...,
        padding: bool | str = ...,
        return_offsets_mapping: bool = ...,
        **kwargs,
    ) -> BatchEncoding: ...

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]: ...
