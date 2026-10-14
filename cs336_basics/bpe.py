import json
import os
import regex
from collections import defaultdict
from collections.abc import Iterable
from typing import List, Tuple, Dict, Set

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None=None):
        self.vocab = vocab
        self.merges = merges
        self.merges_rank = {merge: i for i, merge in enumerate(self.merges)}
        # 按照长度从长到短排序，能避免较长的special_token被拆分
        self.special_tokens = sorted(special_tokens or [], key=lambda x: -len(x))
        self.token2id = {v:k for k, v in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # vocab为json文件，token: id
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        
        vocab = {int(v): k.encode("utf-8") for k, v in vocab_json.items()}
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                tmp = line.strip().split()
                if len(tmp) == 2:
                    merges.append((tmp[0].encode("utf-8"), tmp[1].encode("utf-8")))

        return cls(vocab, merges, special_tokens=special_tokens)

    def encode(self, text:str) -> list[int]:
        if not text:
            return []

        if self.special_tokens:
            # 按照sepcial_tokens先拆分
            pattern = "(" + "|".join(regex.escape(t) for t in self.special_tokens) + ")"
            chunks = regex.split(pattern, text)
        else:
            chunks = [text]
        result = []
        for chunk in chunks:
            if not chunk:
                continue

            if chunk in self.special_tokens:
                result.append(self.token2id[chunk.encode('utf-8')])
            else:
                for word in regex.findall(PAT, chunk):
                    if not word:
                        continue
                    # 按照PAT拆分后，每个token拆分字节
                    tokens = [bytes([b]) for b in word.encode("utf-8")]
                    while len(tokens) > 1:
                        minm_rank = len(self.merges) + 1
                        best_pair = None
                        # 找优先级最高的token先合并
                        for i in range(len(tokens) - 1):
                            tmp_rank = self.merges_rank.get((tokens[i], tokens[i + 1]), float('inf'))
                            if tmp_rank < minm_rank:
                                minm_rank = tmp_rank
                                best_pair = (tokens[i], tokens[i + 1])

                        if best_pair is None:
                            break
                        new_tokens = []
                        i = 0
                        best_pair_str = best_pair[0] + best_pair[1]
                        while i < len(tokens):
                            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                                new_tokens.append(best_pair_str)
                                i += 2
                            else:
                                new_tokens.append(tokens[i])
                                i += 1
                        tokens = new_tokens
                    for i in range(len(tokens)):
                        result.append(self.token2id[tokens[i]])
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        result = bytes()
        for token_id in ids:
            if token_id < len(self.vocab):
                token = self.vocab[token_id]
            else:
                token = bytes("\uFFFD", encoding='utf-8')
            result += token
        result = result.decode(encoding='utf-8', errors='replace')
        return result

def find_max_pair(count):
    max_count = 0
    best_pair = None
    for pair, cnt in count.items():
        if cnt > max_count:
            max_count = cnt
            best_pair = pair
        elif cnt == max_count:
            if best_pair is None or best_pair < pair:
                best_pair = pair

    return max_count, best_pair

def merge_token(old_token: Tuple, best_pair) -> Tuple:
    pair_str = best_pair[0] + best_pair[1]
    idx = 0
    new_token = []
    while idx < len(old_token):
        if idx != len(old_token) - 1 and old_token[idx] == best_pair[0] and old_token[idx + 1] == best_pair[1]:
            new_token.append(pair_str)
            idx += 2
        else:
            new_token.append(old_token[idx])
            idx += 1
    return tuple(new_token)

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 初始化词表
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes
    try:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        print("FileNotFoundError")
        text = ""
    # pre-tokenization
    chunks = regex.split('|'.join(map(regex.escape, special_tokens)), text)
    pretoken_counts = defaultdict(int)
    for chunk in chunks:
        for word in regex.findall(PAT, chunk):
            word_bytes = word.encode("utf-8")
            bytes_list = [bytes([x]) for x in word_bytes]
            pretoken_counts[tuple(bytes_list)] += 1
    count = defaultdict(int)
    for token in pretoken_counts.keys():
        for i in range(len(token) - 1):
            count[(token[i], token[i + 1])] += pretoken_counts[token]
    print(len(pretoken_counts))
    # 计算merges
    merges = []
    while len(vocab) < vocab_size:
        if not count:
            break

        max_count, best_pair = find_max_pair(count)
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        affected_indices = []
        # 找所有包含best_pair的token，重新更新pretoken
        for token, cnt_tmp in pretoken_counts.items():
            flag = False
            for i in range(len(token) - 1):
                if token[i] == best_pair[0] and token[i + 1] == best_pair[1]:
                    flag = True
                    break
            if flag:
                affected_indices.append((token, cnt_tmp))

        for token, cnt_tmp in affected_indices:
            for i in range(len(token) - 1):
                count[(token[i], token[i + 1])] -= cnt_tmp
                if count[(token[i], token[i + 1])] <= 0:
                    del count[(token[i], token[i + 1])]
            new_token = merge_token(token, best_pair)
            for i in range(len(new_token) - 1):
                count[(new_token[i], new_token[i + 1])] += cnt_tmp
            del pretoken_counts[token]
            pretoken_counts[new_token] += cnt_tmp

    return vocab, merges

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    print(special_tokens)
    vocab, merges = train_bpe("../data/TinyStoriesV2-GPT4-valid.txt", 20000, [""])

    print(vocab)
    print(merges)