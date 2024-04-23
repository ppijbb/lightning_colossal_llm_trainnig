import os
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from datetime import datetime as dt
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.trainer.supporters import CombinedLoader

from copy import deepcopy
from scipy.stats import poisson

__all__ = ["LanguageDataModule"]


class Pet_Dataset(Dataset):
    def __init__(self,
                 max_seq_len:int,
                 file_path:str,
                 tokenizer_path:str):
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path).dropna()
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.masking_start, self.masking_end =self.tokenizer.encode("[]")

    def __len__(self):
        return self.data.__len__()

    def _encode(self, text):
        tokens = [self.tokenizer.bos_token] + self.tokenizer.tokenize(text) + [self.tokenizer.eos_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1]*len(input_ids)
        if len(input_ids) < self.max_seq_len:
            while len(input_ids)<self.max_seq_len:
                input_ids+=[self.tokenizer.pad_token_id]
                attention_mask+=[0]
        else:
            input_ids = input_ids[:self.max_seq_len-1]+[self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_seq_len]
        return input_ids, attention_mask

    def _labeling(self, label):
        tokens = self.tokenizer.tokenize(label)+[self.tokenizer.eos_token]
        label_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(label_ids) < self.max_seq_len:
            while len(label_ids)<self.max_seq_len:
                label_ids+=[-100]
        else:
            label_ids = label_ids[:self.max_seq_len-1] + [self.tokenizer.eos_token_id]
        return label_ids

    def _masking(self, tokens):
        masked = tokens
        mask_idx=[]
        while self.masking_start in masked and self.masking_end in masked:
            start_idx = masked.index(self.masking_start)
            end_idx = masked.index(self.masking_end)+1
            if start_idx < end_idx:
                mask_idx +=[[start_idx,end_idx]]
                masked[start_idx:end_idx] = [self.tokenizer.mask_token_id]*len(masked[start_idx:end_idx])
            else: break
        return masked, mask_idx

    def _label_masking(self, tokens, idx):
        masked = np.array(tokens)
        index,position =[],[]
        for i in idx:
            index+=range(i[0],i[1])
        for n,x in enumerate(tokens):
            position += [n not in index]
        masked[position] = [-100]
        return masked#.tolist()


class Masked_Dataset(Pet_Dataset):
    def _random_masking(self, input, mask=None, ratio=0.15):
        if mask is None:
            mask = self.tokenizer.mask_token_id
        input = np.array(input)
        rand = np.random.rand(input.size)
        mask_arr = (rand < ratio) * (input != self.tokenizer.bos_token_id) * (input != self.tokenizer.eos_token_id)
        input[mask_arr.nonzero()] = mask
        return input

    def __getitem__(self, index):
        record = self.data.iloc[index]
        pattern, label = record["pattern"], record["label"]
        encoder_input_ids, encoder_attention_mask = self._encode(pattern)
        decoder_input_ids, decoder_attention_mask = self._encode(pattern+label)
        labels = self._labeling(pattern+label)
        encoder_input_ids, mask_idx = self._masking(encoder_input_ids)
        encoder_input_ids = self._random_masking(encoder_input_ids)

        return {"input_ids":np.array(encoder_input_ids, dtype=np.int_),
                "attention_mask":np.array(encoder_attention_mask,dtype=np.float32),
                "decoder_input_ids":np.array(decoder_input_ids, dtype=np.int_),
                "decoder_attention_mask":np.array(decoder_attention_mask,dtype=np.float32),
                "labels":np.array(labels,dtype=np.int_)}


class Permutation_Dataset(Pet_Dataset):
    def _random_rotation(self, input):
        input = np.array(input)
        start_idx = np.where(input==self.tokenizer.bos_token_id)[0][0]
        end_idx = np.where(input==self.tokenizer.eos_token_id)[0][0]
        np.random.shuffle(input[start_idx:end_idx])
        return input

    def __getitem__(self, index):
        record = self.data.iloc[index]
        pattern, label = record["pattern"], record["label"]
        encoder_input_ids, encoder_attention_mask = self._encode(pattern)
        decoder_input_ids, decoder_attention_mask = self._encode(pattern+label)
        labels = self._labeling(pattern+label)
        encoder_input_ids = self._random_rotation(encoder_input_ids)

        return {"input_ids":np.array(encoder_input_ids, dtype=np.int_),
                "attention_mask":np.array(encoder_attention_mask,dtype=np.float32),
                "decoder_input_ids":np.array(decoder_input_ids, dtype=np.int_),
                "decoder_attention_mask":np.array(decoder_attention_mask,dtype=np.float32),
                "labels":np.array(labels,dtype=np.int_)}


class Deletion_Dataset(Pet_Dataset):
    def _random_deletion(self, input, ratio=0.15):
        input = np.array(input)
        eos_idx = np.where(input==self.tokenizer.eos_token_id)[0][0]
        rand = np.random.rand(input[:eos_idx].size)
        rand = np.append(rand,[1.]*input[eos_idx:].size)
        del_arr = (rand < ratio) * (input != self.tokenizer.bos_token_id) * (input != self.tokenizer.eos_token_id)
        input = np.delete(input, del_arr.nonzero())
        return np.int_(np.append(input,[self.tokenizer.pad_token_id]*del_arr.nonzero()[0].size))#.tolist()

    def __getitem__(self, index):
        record = self.data.iloc[index]
        pattern, label = record["pattern"], record["label"]
        encoder_input_ids, encoder_attention_mask = self._encode(pattern)
        decoder_input_ids, decoder_attention_mask = self._encode(pattern+label)
        labels = self._labeling(pattern+label)
        encoder_input_ids = self._random_deletion(encoder_input_ids)

        return {"input_ids":np.array(encoder_input_ids, dtype=np.int_),
                "attention_mask":np.array(encoder_attention_mask,dtype=np.float32),
                "decoder_input_ids":np.array(decoder_input_ids, dtype=np.int_),
                "decoder_attention_mask":np.array(decoder_attention_mask,dtype=np.float32),
                "labels":np.array(labels,dtype=np.int_)}


class Infilling_Dataset(Pet_Dataset):
    def _random_infilling(self, input, ratio=0.15, l=3):
        input = np.array(input)
        eos_idx = np.where(input==self.tokenizer.eos_token_id)[0][0]
        text_range = input[1:eos_idx]
        poi = poisson(l).pmf(text_range) > ratio
        infill = np.where(poi, np.array([self.tokenizer.mask_token_id]+[-100]*(poi.size-1)), text_range)
        infill = np.append(input[0], np.delete(infill,np.where(infill == -100)))
        infill = np.append(infill, [self.tokenizer.pad_token_id]*(np.count_nonzero(poi)-1))
        infill = np.append(infill, input[eos_idx:])
        if poi.max() == poi.min():
            if not len(infill) > len(input):
                infill = np.insert(infill, np.random.choice(len(text_range)), 
                                   self.tokenizer.mask_token_id)[:len(input)]                    
            else:
                _size = int(len(text_range)*ratio)
                infill = np.insert(infill, np.random.choice(len(text_range),
                                                            size=_size), self.tokenizer.mask_token_id)[:len(input)-1]
                infill = np.append(infill, [self.tokenizer.eos_token_id])
        if infill.size == (self.max_seq_len-1):
            infill = np.append(infill, input[-1])
        return np.int_(infill)

    def __getitem__(self, index):
        record = self.data.iloc[index]
        pattern, label = record["pattern"], record["label"]
        encoder_input_ids, encoder_attention_mask = self._encode(pattern)
        decoder_input_ids, decoder_attention_mask = self._encode(pattern+label)
        labels = self._labeling(pattern+label)
        encoder_input_ids = self._random_infilling(encoder_input_ids)

        return {"input_ids":np.array(encoder_input_ids, dtype=np.int_),
                "attention_mask":np.array(encoder_attention_mask,dtype=np.float32),
                "decoder_input_ids":np.array(decoder_input_ids, dtype=np.int_),
                "decoder_attention_mask":np.array(decoder_attention_mask,dtype=np.float32),
                "labels":np.array(labels,dtype=np.int_)}

class RandomMultiToeknFilling_Dataset(
    Masked_Dataset, Permutation_Dataset, Deletion_Dataset, Infilling_Dataset):
    def _random_pattern(self, input):
        _func = np.random.choice([self._random_masking, 
                                  self._random_rotation, 
                                  self._random_deletion, 
                                  self._random_infilling], 1)[0]
        return _func(input=input)

    def __getitem__(self, index):
        record = self.data.iloc[index]
        pattern, label = record["pattern"], record["label"]
        encoder_input_ids, encoder_attention_mask = self._encode(pattern)
        decoder_input_ids, decoder_attention_mask = self._encode(pattern+label)
        labels = self._labeling(pattern+label)
        encoder_input_ids = self._random_pattern(encoder_input_ids)
        input_ids = np.array(encoder_input_ids, dtype=np.int_)
        attention_mask = np.array(encoder_attention_mask,dtype=np.float32)
        decoder_input_ids = np.array(decoder_input_ids, dtype=np.int_)
        decoder_attention_mask = np.array(decoder_attention_mask,dtype=np.float32)
        labels = np.array(labels,dtype=np.int_)
        return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels          


class LanguageDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_file:str,
                 val_file:str,
                 test_file:str, 
                 tokenizer_path:str,
                 max_seq_len:int=1024,
                 batch_size:int=4,
                 num_workers:int=0,
                 pinned:bool=True):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_file_path = train_file
        self.val_file_path = val_file
        self.test_file_path = test_file
        self.tokenizer_path = tokenizer_path
        self.num_workers = num_workers
        self.pinned = pinned

    def _load_multiple(self, file_path, shuffle):
        return {
            "Deletion Data": DataLoader(
                Deletion_Dataset(
                    max_seq_len=self.max_seq_len,
                    file_path=file_path,
                    tokenizer_path=self.tokenizer_path),
                pin_memory=self.pinned,
                batch_size=self.batch_size,
                num_workers=self.num_workers, 
                shuffle=shuffle),
            "Permutation Data": DataLoader(
                Permutation_Dataset(
                    max_seq_len=self.max_seq_len,
                    file_path=file_path,
                    tokenizer_path=self.tokenizer_path),
                pin_memory=self.pinned,
                batch_size=self.batch_size,
                num_workers=self.num_workers, 
                shuffle=shuffle),
            "Masked Data": DataLoader(
                Masked_Dataset(
                    max_seq_len=self.max_seq_len,
                    file_path=file_path,
                    tokenizer_path=self.tokenizer_path),
                pin_memory=self.pinned,
                batch_size=self.batch_size,
                num_workers=self.num_workers, 
                shuffle=shuffle),
            "Infilling Data": DataLoader(
                Infilling_Dataset(
                    max_seq_len=self.max_seq_len,
                    file_path=file_path,
                    tokenizer_path=self.tokenizer_path),
                pin_memory=self.pinned,
                batch_size=self.batch_size,
                num_workers=self.num_workers, 
                shuffle=shuffle)
        }

    def _load_random_token(self, file_path, shuffle):
        return DataLoader(
            RandomMultiToeknFilling_Dataset(
                max_seq_len=self.max_seq_len,
                file_path=file_path,
                tokenizer_path=self.tokenizer_path),
            pin_memory=self.pinned,
            batch_size=self.batch_size,
            num_workers=self.num_workers, 
            shuffle=shuffle)

    def setup(self, stage):
        # self.train = self._load_multiple(file_path=self.train_file_path, shuffle=True)
        # self.val = self._load_multiple(file_path=self.val_file_path, shuffle=False)
        # self.test = self._load_multiple(file_path=self.test_file_path, shuffle=False)
        self.train = self._load_random_token(file_path=self.train_file_path, shuffle=True)
        self.val = self._load_random_token(file_path=self.val_file_path, shuffle=False)
        self.test = self._load_random_token(file_path=self.test_file_path, shuffle=False)

    def train_dataloader(self):
        return self.train
        # return CombinedLoader(self.train, mode="max_size_cycle")

    def val_dataloader(self):
        return self.val
        # return CombinedLoader(self.val, mode="max_size_cycle")

    def test_dataloader(self):
        return self.test
        # return CombinedLoader(self.test, mode="max_size_cycle")
