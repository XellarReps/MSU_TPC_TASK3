import torch
import lzma
import psutil
import pickle
from typing import Tuple, List, Dict, Any, Iterable, Set
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, BertForTokenClassification, BertTokenizerFast
from dataset import NERDataset, NERCollator
from utils import read_pickle
import sys
from dataclasses import dataclass

def ent_type(lbl: str) -> str:
    return lbl.split('-')[1]


def is_ent(lbl: str) -> bool:
    return lbl.split('-')[0] in {'B', 'I'}


def is_ent_start(lbl: str) -> bool:
    return lbl.startswith('B')


def decode_labels(labels: Iterable[str], spans: List[Tuple[int, int]]):

    prev_label = 'O'
    start: Optional[int] = None

    curr_idx = 0
    for curr_idx, label in enumerate(labels):
        if is_ent(label):
            if is_ent(prev_label):
                # ... X1-ENT1 X2-ENT2 ...
                if ent_type(label) != ent_type(prev_label) or is_ent_start(label):
                    # ENT1 != ENT2 or X2 == B - entity ended + start new entity
                    yield (spans[start][0], spans[curr_idx - 1][1], ent_type(prev_label))
                    start = curr_idx
                # X2 != B and ENT1 == ENT2 - do nothing
            else:
                # ... O X-ENT ... - start new entity
                start = curr_idx
        else:
            if is_ent(prev_label):
                # ... X-ENT O ... - entity ended
                yield (spans[start][0], spans[curr_idx - 1][1], ent_type(prev_label))
                start = None  # do not start new entity
            # ... O O ... - do nothing

        prev_label = label

    if is_ent(prev_label):
        # sequence ended on entity
        yield (spans[start][0], spans[curr_idx][1], ent_type(prev_label))

tokenizer = BertTokenizerFast.from_pretrained("tokenizer")
with open("../../Documents/task3_texts/state.xz", "rb") as state_f:
    with open("../../Documents/task3_texts/config.xz", "rb") as config_f:
        state_data = state_f.read()
        config_data = config_f.read()
 
        filters = [{"id":lzma.FILTER_LZMA2,"dict_size":268435456, "preset":9, "mf":lzma.MF_HC3, "depth":0, "lc":3}]
        state_data = lzma.decompress(state_data, format=lzma.FORMAT_RAW, filters=filters)
        config_data = lzma.decompress(config_data, format=lzma.FORMAT_RAW, filters=filters)
 
        state = pickle.loads(state_data)
        config = pickle.loads(config_data)
        model = BertForTokenClassification.from_pretrained(config=config, state_dict=state, pretrained_model_name_or_path=None)

token2idx = read_pickle('token2idx.pkl')
label2idx = read_pickle('label2idx.pkl')
idx2label: Dict[int, str] = {}
for key, val in label2idx.items():
    idx2label[val] = key

def label_without_bio(label_with_bio):
    return label_with_bio[2:]

def bio_without_lable(label_with_bio):
    return label_with_bio[0:1]

class Solution:
    def predict(self, texts: List[str]) -> Iterable[Set[Tuple[int, int, str]]]:
        with torch.no_grad():
            tokens_, spans_ = [], []

            for text in texts:
                spans = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False).offset_mapping
                tokens = []
                for span in spans:
                    tokens.append(text[span[0]:span[1]])
                tokens_.append(tokens)
                spans_.append(spans)
            

            data = NERDataset(
                token2idx = token2idx,
                token_seq = tokens_,
            )
            collator = NERCollator(
                token_padding_value=tokenizer.pad_token_id,
            )
            dataloader = torch.utils.data.DataLoader(
                data,
                batch_size=1,
                shuffle=False,
                collate_fn=collator, 
            )
            pred = []
            for i, tokens in enumerate(dataloader):
                outputs = model(tokens)
                out = outputs.logits.argmax(dim=-1).squeeze(0)
                #print(out)
                l = [idx2label[elem.item()] for elem in out]
                pred.append(set(decode_labels(l, spans_[i])))

            return pred
