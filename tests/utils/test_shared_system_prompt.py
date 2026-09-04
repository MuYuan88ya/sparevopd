import pytest
from omegaconf import OmegaConf

from verl.utils.dataset.rl_dataset import RLHFDataset


def _dataset_for_message_building(system_prompt: str | None) -> RLHFDataset:
    dataset = RLHFDataset.__new__(RLHFDataset)
    dataset.prompt_key = "prompt"
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.processor = None
    dataset.system_prompt = system_prompt
    return dataset


def test_load_system_prompt_strips_outer_whitespace(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("\n  shared prompt  \n", encoding="utf-8")

    assert RLHFDataset._load_system_prompt(str(prompt_path)) == "shared prompt"


def test_load_system_prompt_rejects_missing_or_empty_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="System prompt file not found"):
        RLHFDataset._load_system_prompt(str(tmp_path / "missing.txt"))

    prompt_path = tmp_path / "empty.txt"
    prompt_path.write_text(" \n", encoding="utf-8")
    with pytest.raises(ValueError, match="System prompt file is empty"):
        RLHFDataset._load_system_prompt(str(prompt_path))


def test_shared_system_prompt_is_prepended_without_mutating_source():
    dataset = _dataset_for_message_building("shared prompt")
    example = {"prompt": [{"role": "user", "content": "question"}]}

    messages = dataset._build_messages(example)

    assert messages == [
        {"role": "system", "content": "shared prompt"},
        {"role": "user", "content": "question"},
    ]
    assert example["prompt"] == [{"role": "user", "content": "question"}]


def test_shared_system_prompt_replaces_existing_system_message():
    dataset = _dataset_for_message_building("shared prompt")
    example = {
        "prompt": [
            {"role": "system", "content": "dataset prompt"},
            {"role": "user", "content": "question"},
        ]
    }

    messages = dataset._build_messages(example)

    assert messages[0] == {"role": "system", "content": "shared prompt"}
    assert sum(message["role"] == "system" for message in messages) == 1


def test_shared_system_prompt_does_not_consume_image_placeholder():
    dataset = _dataset_for_message_building("shared prompt")
    dataset.processor = object()
    example = {
        "prompt": [{"role": "user", "content": "<image>question"}],
        "images": [{"path": "/tmp/image.png"}],
    }

    messages = dataset._build_messages(example)

    assert messages[0] == {
        "role": "system",
        "content": [{"type": "text", "text": "shared prompt"}],
    }
    assert messages[1]["content"] == [
        {"type": "image", "path": "/tmp/image.png", "image": "/tmp/image.png"},
        {"type": "text", "text": "question"},
    ]


def test_data_config_declares_system_prompt_file():
    config = OmegaConf.load("verl/trainer/config/data/legacy_data.yaml")

    assert "system_prompt_file" in config
    assert config.system_prompt_file is None
