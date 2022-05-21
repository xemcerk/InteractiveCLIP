# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

import random
import numpy as np
import os
import json
import torch.utils.data
import PIL
from .base_dataset import BaseDataset

CATEGORIES = ["dress", "shirt", "toptee"]


class FashionIQ(BaseDataset):

    def __init__(
        self,
        path,
        split='train',
        transform=None,
        batch_size=None,
        processor=None,
        val_loader_mode=None
    ):
        super(FashionIQ, self).__init__()
        self.categories = CATEGORIES
        self.batch_size = batch_size
        self.processor = processor
        self.filter_category = False

        self.val_loader_mode = val_loader_mode

        self.split = split
        self.transform = transform
        self.img_path = path + '/'

        failures = []

        data = {
            'image_splits': {},
            'captions': {}
        }

        for data_type in data:
            for datafile in os.listdir(os.path.join(path, data_type)):
                if split in datafile:
                    data[data_type][datafile] = \
                        json.load(
                            open(os.path.join(path, data_type, datafile)))

        split_labels = sorted(list(data["image_splits"].keys()))

        # load all images, well just record its file path and give them unique id
        global_imgs = []
        img_by_cat = {cat: [] for cat in CATEGORIES}
        self.asin2id = {}
        for splabel in split_labels:
            for asin in data['image_splits'][splabel]:
                category = splabel.split(".")[1]
                file_path = os.path.join(path, "images", asin+".jpg")
                # file_path = path + '/img/' + category + '/' + asin
                if os.path.exists(file_path) or split == "test":
                    global_id = len(global_imgs)
                    category_id = len(img_by_cat[category])
                    entry = [{
                        'asin': asin,
                        'file_path': file_path,
                        'captions': [global_id],
                        "image_id": global_id,
                        "category": {category: category_id}
                    }]
                    if asin in self.asin2id:
                        # handle duplicates
                        oldglobal = self.asin2id[asin]
                        subentry = global_imgs[oldglobal]
                        assert category not in subentry["category"], \
                            "{} duplicated in {}".format(asin, category)

                        # update entry to include additional category and id
                        subentry["category"][category] = category_id
                        img_by_cat[category] += [subentry]
                    else:
                        # just add the entry
                        global_imgs += entry
                        img_by_cat[category] += entry
                        self.asin2id[asin] = global_id
                else:
                    failures.append(asin)

        print(len(failures), " files not found in ", split)
        assert len(global_imgs) > 0, "no data found"

        queries = []
        captions = sorted(list(data["captions"].keys()))
        for cap in captions:
            for query in data['captions'][cap]:
                if split != "test" and (query['candidate'] in failures
                                        or query.get('target') in failures):
                    continue
                query['source_id'] = self.asin2id[query['candidate']]
                query["category"] = cap.split(".")[1]
                if split != "test":
                    query['target_id'] = self.asin2id[query['target']]
                    tarcat = global_imgs[query['target_id']]["category"]
                    if query["category"] not in tarcat:
                        print("WARNING: a {} found with a target in {}".format(
                            query["category"], tarcat
                        ))
                soucat = global_imgs[query['source_id']]["category"]
                assert query["category"] in soucat

                queries += [query]

        self.data = data
        self.imgs = global_imgs
        self.img_by_cat = img_by_cat
        self.queries = queries
        
        # tokenize text if processor exist
        if(self.processor):
            self.tokenize_text()
        
        self.queries_by_cat = {}
        for cat in CATEGORIES:
                self.queries_by_cat[cat] = [que for que in self.queries if que["category"] == cat]


        if split == "val":
            self.test_queries = [{
                'source_img_id': query['source_id'],
                'target_img_id': query['target_id'],
                'target_caption': query['target_id'],
                'mod': {'str': query['captions'][0] + ' inadditiontothat ' +
                                 query['captions'][1]},
                "category": query["category"]
            } for query in queries]
            
            self.test_queries_by_cat = {}
            for cat in CATEGORIES:
                self.test_queries_by_cat[cat] = self.get_test_queries(category=cat)
                    

        if split == "test":
            self.test_queries = [{
                'source_img_id': query['source_id'],
                'mod': {'str': query['captions'][0] + ' inadditiontothat ' +
                                 query['captions'][1]},
                "category": query["category"]
            } for query in queries]

        self.id2asin = {val: key for key, val in self.asin2id.items()}
        self.current_category = CATEGORIES[0]

    def get_all_texts(self):
        texts = [' inadditiontothat ']
        for query in self.queries:
            texts += query['captions']
        return texts

    def __len__(self):
        if(self.split == "val" and self.val_loader_mode == "imgs"):
            return len(self.img_by_cat[self.current_category]) \
                if self.filter_category else len(self.imgs)
        else: # for train set and val queries set
            return len(self.queries_by_cat[self.current_category]) \
                if self.filter_category else len(self.queries)

    def get_loader(self,
                   batch_size,
                   shuffle=False,
                   drop_last=False,
                   num_workers=0):
        if self.split == "train" or self.val_loader_mode == "query":
            def _collate_fn(samples):
                ref_imgs = {"pixel_values": torch.cat(
                    [sample['source_img_data']['pixel_values'] for sample in samples], 0)}
                tgt_imgs = {"pixel_values": torch.cat(
                    [sample['target_img_data']['pixel_values'] for sample in samples], 0)}
                mod_strs = {
                    "input_ids": torch.stack([sample['tokenized_mod_str']['input_ids'] for sample in samples]),
                    "attention_mask": torch.stack([sample['tokenized_mod_str']['attention_mask'] for sample in samples])
                }
                return ref_imgs, tgt_imgs, mod_strs
            self.batch_size = batch_size
            return torch.utils.data.DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                collate_fn=_collate_fn)
        else: # for val imgs set
            def _collate_fn(samples):
                return {"pixel_values": torch.cat([sample['pixel_values'] for sample in samples], 0)}
            return torch.utils.data.DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                collate_fn=_collate_fn)

    def __getitem__(self, idx):
        if(self.split == "val" and self.val_loader_mode == "imgs"): # for val imgs set
            entry = self.img_by_cat[self.current_category][idx] \
                if self.filter_category else self.imgs[idx]
            return self.processor(
                images=self.get_img(entry["image_id"]),
                return_tensors='pt'
            ) if(self.processor) else self.get_img(entry["image_id"])
        else:
            query = self.queries_by_cat[self.current_category][idx] \
                if self.filter_category else self.queries[idx]
            return {
                'source_img_id': query['source_id'],
                'source_img_data': self.processor(
                    images=self.get_img(query['source_id']),
                    return_tensors='pt'
                ) if(self.processor) else self.get_img(query['source_id']),
                'target_img_id': query['target_id'],
                'target_caption': query['target_id'],
                'target_img_data': self.processor(
                    images=self.get_img(query['target_id']),
                    return_tensors='pt'
                ) if(self.processor != None) else self.get_img(query['target_id']),
                'mod': {'str': query['mod_str']},
                'tokenized_mod_str': query['tokenized_mod_str']
            }

    def tokenize_text(self):
        mod_strs = []
        # combine captions into modification text
        for idx, query in enumerate(self.queries):
            mod_str = random.choice([
                query['captions'][0] + ' inadditiontothat ' +
                query['captions'][1],
                query['captions'][1] + ' inadditiontothat ' +
                query['captions'][0]
            ])
            self.queries[idx]['mod_str'] = mod_str
            mod_strs.append(mod_str)
        
        # tokenize all modification text
        self.tokenized_mod_strs = self.processor(
            text=mod_strs, return_tensors='pt', padding=True)
        
        # tuck tokenized text into queries
        for idx, query in enumerate(self.queries):
            self.queries[idx]['tokenized_mod_str'] = {
                'input_ids': self.tokenized_mod_strs['input_ids'][idx],
                'attention_mask': self.tokenized_mod_strs['attention_mask'][idx]
            }
        
        

    def get_img(self, idx, raw_img=True):
        """Retrieve image by global index."""
        img_path = self.imgs[idx]['file_path']
        try:
            with open(img_path, 'rb') as f:
                img = PIL.Image.open(f)
                img = img.convert('RGB')
        except EnvironmentError as ee:
            print("WARNING: EnvironmentError, defaulting to image 0", ee)
            img = self.get_img(0, raw_img=True)
        if raw_img:
            return img
        if self.transform:
            img = self.transform(img)
        return img

    def get_test_queries(self, category=None):
        if category is not None:
            return [que for que in self.test_queries if que["category"] == category]
        elif(self.filter_category):
            return [que for que in self.test_queries if que["category"] == self.current_category]
        return self.test_queries
