import pytorch_lightning as pl
import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor
import numpy as np


class InteractiveCLIP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # TODO make it configurable
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.compose_choice = 'mean'

    def forward(self, x):
        pass

    def compute_query_features(self, ref_imgs, mod_strs, tgt_imgs=None, normalize=True):
        text_features = self.clip.get_text_features(
            input_ids=mod_strs['input_ids'], attention_mask=mod_strs['attention_mask'])
        ref_imgs_features = self.clip.get_image_features(
            pixel_values=ref_imgs['pixel_values'])
        if tgt_imgs:
            tgt_imgs_features = self.clip.get_image_features(
                pixel_values=tgt_imgs['pixel_values'])

        # normalize
        if(normalize):
            text_features = text_features / \
                text_features.norm(dim=-1, keepdim=True)
            ref_imgs_features = ref_imgs_features / \
                ref_imgs_features.norm(dim=-1, keepdim=True)
            if tgt_imgs:
                tgt_imgs_features = tgt_imgs_features / \
                    tgt_imgs_features.norm(dim=-1, keepdim=True)

        # compose reference image feature and modifciation text feature
        if (self.compose_choice == "mean"):
            mod_imgs_features = (text_features + ref_imgs_features) / 2
        if tgt_imgs:
            return tgt_imgs_features, mod_imgs_features
        else: return mod_imgs_features

    def training_step(self, batch, batch_idx):

        def _contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
            return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

        def _clip_loss(similarity: torch.Tensor) -> torch.Tensor:
            ref_loss = _contrastive_loss(similarity)
            tgt_loss = _contrastive_loss(similarity.T)
            return (ref_loss + tgt_loss) / 2.0

        ref_imgs, tgt_imgs, mod_strs = batch
        tgt_imgs_features, mod_imgs_features = self.compute_query_features(
                ref_imgs, mod_strs, tgt_imgs)

        # cosine similarity as logits
        logit_scale = self.clip.logit_scale.exp()
        logits_per_ref = torch.matmul(
            mod_imgs_features, tgt_imgs_features.t()) * logit_scale

        loss = _clip_loss(logits_per_ref)

        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx) :
        if(dataloader_idx == 0):
            ref_imgs, _, mod_strs = batch
            mod_imgs_features = self.compute_query_features(
                ref_imgs, mod_strs)
            return {
                "dataloader_idx": dataloader_idx, 
                'batch_idx':batch_idx,
                "mod_imgs_features": mod_imgs_features
            }
        else:
            cand_imgs = batch
            cand_imgs_features = self.clip.get_image_features(pixel_values=cand_imgs['pixel_values'])
            return {
                "dataloader_idx": dataloader_idx, 
                'batch_idx':batch_idx,
                "cand_imgs_features": cand_imgs_features
            }
            
    def validation_step_end(self, batch_parts):
        
        gathered_batch_parts = self.all_gather(batch_parts)
        
        # if self.trainer.is_global_zero:
        #     print("I know I'm global zero")
        #     import ipdb;ipdb.set_trace()
        # # to prevent other processes from moving forward until all processes are in sync
        # torch.distributed.barrier()
        feature_key = "mod_imgs_features" if batch_parts["dataloader_idx"] == 0 else "cand_imgs_features"

        return {
                "dataloader_idx": batch_parts["dataloader_idx"], 
                'batch_idx':batch_parts["dataloader_idx"],
                feature_key: gathered_batch_parts[feature_key].reshape((-1, 512))
            }
            
    def validation_epoch_end(self, step_outputs: list):
        # if self.trainer.is_global_zero:
        # get valset, test_queries, all_captions, all_target_captions
        # note: all_captions is basically [i for i in range(len(all_imgs))]
        valset, test_queries, all_captions, all_target_captions = self.get_queries_captions()
        
        # get all queries and all images from step output
        all_queries = sorted(step_outputs[0], key=lambda k: k['batch_idx'])
        all_queries = torch.cat([ii['mod_imgs_features'] for ii in all_queries])
        all_imgs = sorted(step_outputs[1], key=lambda k: k['batch_idx'])
        all_imgs = torch.cat([ii['cand_imgs_features'] for ii in all_imgs])
        
        all_queries = all_queries.data.cpu().numpy()
        all_imgs = all_imgs.data.cpu().numpy()
        
        # normalize
        for i in range(all_queries.shape[0]):
            all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
            
        for i in range(all_imgs.shape[0]):
            all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
            
        # compute similarity    
        sims = all_queries.dot(all_imgs.T)
        print(sims.shape)
        
        # import ipdb; ipdb.set_trace()
        # remove reference image id
        for i, t in enumerate(test_queries):
            source_id = t['source_img_id']
            if(valset.filter_category):
                source_id = valset.imgs[source_id]["category"][valset.current_category]
            sims[i, source_id] = sims[i].min()
        
        
        # import ipdb;ipdb.set_trace()
        # handle ddp uneven sampling problem
        if(sims.shape[1] != len(all_captions)): sims = sims[:, :len(all_captions)]
        
        nn_result = [np.argsort(-sims[i, :]) for i in range(sims.shape[0])]
        nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
        
        sorted_sims = [np.sort(sims[ii, :])[::-1] for ii in range(sims.shape[0])]
        
        out = []
        for k in [1, 5, 10, 50, 100]:
            recall = 0.0
            for i, nns in enumerate(nn_result):
                if all_target_captions[i] in nns[:k]:
                    recall += 1
            recall /= len(nn_result)
            out += [('recall_top' + str(k) + '_correct_composition', recall)]
            print(('recall_top' + str(k) + '_correct_composition', recall))
        self.log("recall_10_50_mean", (out[3][1] + out[4][1])/2)
        # torch.distributed.barrier()
            
    def get_queries_captions(self):
        try:
            val_dataloader = self.trainer.val_dataloaders[0]
        except:
            val_dataloader = self.trainer.test_dataloaders[0]
        valset = val_dataloader.dataset
        test_queries = valset.get_test_queries()
        
        imgset = valset.img_by_cat[valset.current_category] \
                if valset.filter_category else valset.imgs
        all_captions = [img['captions'][0] for img in imgset ]
        
        all_target_captions = [tq['target_caption'] for tq in test_queries]
        
        return valset,test_queries,all_captions,all_target_captions
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    # TODO implement eval method
    # def test(self):