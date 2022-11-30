import torch
from typing import Tuple, List, Dict, Any, Iterable, Set

# КОД ВЗЯТ ИЗ НОУТБУКА ПО КУРСУ DL (3ье домашнее задание)
class NERDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for NER.
    """

    def __init__(
        self,
        token_seq: List[List[str]],
        token2idx: Dict[str, int],
    ):
        self.token2idx = token2idx

        self.token_seq = [self.process_tokens(tokens, token2idx) for tokens in token_seq]

    def __len__(self):
        return len(self.token_seq)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        first_item = torch.LongTensor(self.token_seq[idx])
        return first_item
        
    @staticmethod
    def process_tokens(
        tokens: List[str],
        token2idx: Dict[str, int],
        unk: str = "<UNK>",
    ) -> List[int]:
        """
        Transform list of tokens into list of tokens' indices.
        """
        idxs = []
        for token in tokens:
            key = unk
            if token in token2idx:
                key = token
            idx = token2idx[key]
            idxs.append(idx)
        return idxs


# КОД ВЗЯТ ИЗ НОУТБУКА ПО КУРСУ DL (3ье домашнее задание)
class NERCollator:
    """
    Collator that handles variable-size sentences.
    """

    def __init__(
        self,
        token_padding_value: int,
    ):
        self.token_padding_value = token_padding_value

    def __call__(
        self,
        batch: List[torch.LongTensor],
    ) -> torch.LongTensor:

        tokens = []

        for token in batch:
            tokens.append(torch.LongTensor(token))

        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.token_padding_value)
        return tokens