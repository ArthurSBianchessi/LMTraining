
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from typing import Iterator, Optional, Any, Dict, Union

class FineWebDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: Any,
        seq_len: int = 2048,
        dataset_name: str = "HuggingFaceFW/fineweb",
        subset: str = "sample-10BT",
        split: str = "train",
        tokenization_batch_size: int = 1000,
        streaming: bool = True,
        shuffle: bool = True,
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
    ):
        """
        Iterable dataset for FineWeb using HuggingFace Datasets features.
        
        Args:
            tokenizer: The tokenizer instance.
            seq_len: The target sequence length (input length). The yield will consume seq_len + 1 tokens.
            dataset_name: HuggingFace dataset name.
            subset: Dataset subset/config name.
            split: Dataset split to load.
            tokenization_batch_size: Batch size for `map` operations (tokenization).
            streaming: Whether to stream the dataset.
            shuffle: Whether to shuffle the dataset.
            shuffle_buffer_size: Buffer size for shuffling if streaming is True.
            seed: Random seed for shuffling.
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.tokenization_batch_size = tokenization_batch_size
        self.streaming = streaming
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # Load the dataset
        dataset = load_dataset(
            self.dataset_name,
            name=self.subset,
            split=self.split,
            streaming=self.streaming
        )
        
        # Shuffle if enabled
        if self.shuffle:
            if self.streaming:
                dataset = dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)
            else:
                dataset = dataset.shuffle(seed=self.seed)
        
        # Define tokenization function
        def tokenize_batch(batch):
            return self.tokenizer(
                batch['text'],
                add_special_tokens=False,
                return_attention_mask=False
            )
        
        # Apply mapping
        # remove_columns=['text'] to avoid carrying over massive text data
        # If features are not available (e.g. some streaming setups), we assume 'text' is there or just try to remove it.
        # However, map might complain if we try to remove a column that it doesn't know about if strict.
        # But commonly for FineWeb 'text' is the column.
        
        remove_cols = ['text']
        if dataset.features is not None:
             remove_cols = [c for c in dataset.features if c == 'text']
        
        dataset = dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=self.tokenization_batch_size,
            remove_columns=remove_cols
        )
        
        # Buffer for flattening
        # We need seq_len + 1 tokens to create input (0..N-1) and target (1..N)
        chunk_size = self.seq_len + 1
        
        buffer_ids = []
        buffer_seq_ids = []
        
        # Document counter (sequence counter)
        # Using enumerate on the iterator gives us a unique ID per document in this stream
        for doc_idx, sample in enumerate(dataset):
            tokens = sample['input_ids']
            
            # Extend buffers
            buffer_ids.extend(tokens)
            buffer_seq_ids.extend([doc_idx] * len(tokens))
            
            # Yield chunks
            while len(buffer_ids) >= chunk_size:
                chunk_ids = torch.tensor(buffer_ids[:chunk_size], dtype=torch.long)
                chunk_seq_ids = torch.tensor(buffer_seq_ids[:chunk_size], dtype=torch.long)
                
                input_ids = chunk_ids[:-1]
                target = chunk_ids[1:]
                
                # sequence_id for input tokens
                input_seq_ids = chunk_seq_ids[:-1]
                
                yield {
                    "input": {
                        "input_ids": input_ids,
                        "sequence_id": input_seq_ids
                    },
                    "target": target
                }
                
                buffer_ids = buffer_ids[self.seq_len:]
                buffer_seq_ids = buffer_seq_ids[self.seq_len:]
