from dataclasses import dataclass
from datasets import Dataset, IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict, IterableDatasetDict

from analysis.utils import assert_type, simple_parse_args_string


@dataclass
class DataConfig:
    model: str
    dataset: str
    split: str
    subset: str
    data_args: str
    prompt_column: str
    completion_column: str
    conversation_column: str
    truncation: bool
    max_length: int
    batch_size: int


def tokenize(batch: dict, *, args: DataConfig, tokenizer):
    """Tokenize a batch of data with `tokenizer` according to `args`."""
    # Ensure pad token is set (single line for debugging)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    kwargs = dict(
        return_attention_mask=False,
        return_length=True,
        truncation=args.truncation,
        padding="max_length",  # Enable padding to max_length
        max_length=args.max_length if hasattr(args, 'max_length') else 2048,  # Set max length
    )
    
    if args.completion_column:
        # We're dealing with a prompt-completion dataset
        convos = [
            [
                {"role": "user", "content": assert_type(str, prompt)},
                {"role": "assistant", "content": assert_type(str, resp)},
            ]
            for prompt, resp in zip(
                batch[args.prompt_column], batch[args.completion_column]
            )
        ]
    elif args.conversation_column:
        # We're dealing with a conversation dataset
        convos = assert_type(list, batch[args.conversation_column])
    else:
        # We're dealing with vanilla next-token prediction
        return tokenizer(batch[args.prompt_column], **kwargs)

    # Make sure we only compute loss on the assistant's responses
    strings = tokenizer.apply_chat_template(convos, tokenize=False)
    encodings = tokenizer(strings, **kwargs)
    labels_list: list[list[int]] = []

    for i, convo in enumerate(convos):
        # Find the spans of the assistant's responses in the tokenized output
        pos = 0
        spans: list[tuple[int, int]] = []

        for msg in convo:
            if msg["role"] != "assistant":
                continue

            ans = msg["content"]
            start = strings[i].rfind(ans, pos)
            if start < 0:
                raise RuntimeError(
                    "Failed to find completion in the chat-formatted conversation. "
                    "Make sure the chat template does not alter the completion, e.g. "
                    "by removing leading whitespace."
                )

            # move past this match
            pos = start + len(ans)

            start_token = encodings.char_to_token(i, start)
            end_token = encodings.char_to_token(i, pos)
            spans.append((start_token, end_token))

        # Labels are -100 everywhere except where the assistant's response is
        tokens = encodings["input_ids"][i]
        labels = [-100] * len(tokens)
        for start, end in spans:
            if start is not None and end is not None:
                labels[start:end] = tokens[start:end]
        
        # Ensure padding tokens are also -100 in labels (single line for debugging)
        for j, token_id in enumerate(tokens): labels[j] = -100 if token_id == tokenizer.pad_token_id else labels[j]

        labels_list.append(labels)

    return dict(**encodings, labels=labels_list)


def load_data_string(
    data_str: str,
    split: str = "train",
    subset: str | None = None,
    data_args: str = "",
) -> Dataset | IterableDataset:
    """Load a dataset from a string identifier or path."""
    if data_str.endswith(".csv"):
        ds = assert_type(Dataset, Dataset.from_csv(data_str))
    elif data_str.endswith(".json") or data_str.endswith(".jsonl"):
        ds = assert_type(Dataset, Dataset.from_json(data_str))
    else:
        try:
            kwargs = simple_parse_args_string(data_args)
            ds = load_dataset(data_str, subset, split=split, **kwargs)

            if isinstance(ds, DatasetDict) or isinstance(ds, IterableDatasetDict):
                raise NotImplementedError(
                    "DatasetDicts and IterableDatasetDicts are not supported."
                )
        except ValueError as e:
            # Automatically use load_from_disk if appropriate
            if "load_from_disk" in str(e):
                ds = Dataset.load_from_disk(data_str, keep_in_memory=False)
            else:
                raise e

    return ds


def setup_data_pipeline(cfg: DataConfig) -> Dataset | IterableDataset:
    """Handle data loading and preprocessing"""
    ds = load_data_string(
        cfg.dataset, cfg.split, cfg.subset, cfg.data_args
    )

    # In many cases the token_batch_size may be smaller than the max length allowed by
    # the model. If cfg.data.truncation is True, we use the tokenizer to truncate
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.model_max_length = min(tokenizer.model_max_length, cfg.max_length)

    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=cfg, tokenizer=tokenizer),
        batch_size=cfg.batch_size,
    )

    return ds
