# SPDX-License-Identifier: MIT
"""Dependency-free ByteLevel BPE for the pinned Qwen and MiniMind tokenizers.

The pretokenizers implement each pinned model's Unicode splitting rules.
Tests compare Unicode, whitespace and contractions against reference token IDs.
"""
import json
import unicodedata
from functools import lru_cache


def byte_alphabet():
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = values[:]
    n = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            chars.append(256 + n)
            n += 1
    return dict(zip(values, map(chr, chars)))


def pieces(text):
    """Equivalent to 's|'t|'re|'ve|'m|'ll|'d| ?L+| ?N+| ?other+|ws-lookahead|ws+."""
    def kind(c):
        category = unicodedata.category(c)[0]
        return category if category in ('L', 'N') else ('W' if c.isspace() else 'O')
    i = 0
    while i < len(text):
        contraction = next((s for s in ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d") if text.startswith(s, i)), None)
        if contraction:
            yield contraction
            i += len(contraction)
            continue
        start = i
        if text[i] == ' ' and i + 1 < len(text) and kind(text[i + 1]) != 'W':
            i += 1
        group = kind(text[i])
        i += 1
        while i < len(text) and kind(text[i]) == group:
            i += 1
        if group == 'W' and i < len(text) and i - start > 1:
            i -= 1
        yield text[start:i]


def qwen_pieces(text):
    def kind(c):
        return unicodedata.category(c)[0]
    i = 0
    while i < len(text):
        start = i
        contraction = next((s for s in ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d") if text[i:i+len(s)].lower() == s), None)
        if contraction:
            i += len(contraction)
        elif kind(text[i]) == 'L' or (text[i] not in '\r\n' and kind(text[i]) not in ('L', 'N')
                                      and i + 1 < len(text) and kind(text[i+1]) == 'L'):
            i += 1
            while i < len(text) and kind(text[i]) == 'L':
                i += 1
        elif kind(text[i]) == 'N':
            i += 1
        elif (not text[i].isspace() and kind(text[i]) not in ('L', 'N')) or (
                text[i] == ' ' and i + 1 < len(text) and not text[i+1].isspace() and kind(text[i+1]) not in ('L', 'N')):
            if text[i] == ' ':
                i += 1
            while i < len(text) and not text[i].isspace() and kind(text[i]) not in ('L', 'N'):
                i += 1
            while i < len(text) and text[i] in '\r\n':
                i += 1
        else:
            while i < len(text) and text[i].isspace():
                i += 1
            last_newline = max(text.rfind('\r', start, i), text.rfind('\n', start, i))
            if last_newline >= start:
                i = last_newline + 1
            elif i < len(text) and i - start > 1:
                i -= 1
        yield text[start:i]


class Tokenizer:
    def __init__(self, path):
        data = json.loads(path.read_text(encoding='utf-8'))
        self.qwen = data['pre_tokenizer']['type'] == 'Sequence'
        assert data['normalizer'] == ({'type': 'NFC'} if self.qwen else None)
        self.vocab = data['model']['vocab']
        self.inverse = {v: k for k, v in self.vocab.items()}
        self.ranks = {tuple(m.split(' ') if isinstance(m, str) else m): i
                      for i, m in enumerate(data['model']['merges'])}
        self.byte_chars = byte_alphabet()
        self.char_bytes = {v: k for k, v in self.byte_chars.items()}

    @lru_cache(maxsize=4096)
    def bpe(self, text):
        word = list(text)
        while len(word) > 1:
            pair = min(zip(word, word[1:]), key=lambda p: self.ranks.get(p, float('inf')))
            if pair not in self.ranks:
                break
            out = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                    out.append(word[i] + word[i + 1])
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = out
        return tuple(self.vocab[w] for w in word)

    def encode(self, text):
        out = []
        if self.qwen:
            text = unicodedata.normalize('NFC', text)
        for part in (qwen_pieces(text) if self.qwen else pieces(text)):
            out.extend(self.bpe(''.join(self.byte_chars[b] for b in part.encode('utf-8'))))
        return out

    def prompt(self, text):
        # Literal user special-token strings are encoded as ordinary text.
        bos, eos = (151644, 151645) if self.qwen else (1, 2)
        system = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' if self.qwen else 'You are a helpful assistant'
        return ([bos] + self.encode('system\n' + system) + [eos]
                + self.encode('\n') + [bos] + self.encode('user\n' + text) + [eos]
                + self.encode('\n') + [bos] + self.encode('assistant\n'))

    def decode_bytes(self, token):
        return bytes(self.char_bytes[c] for c in self.inverse[token])
