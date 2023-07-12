#import clip
import os
import cv2
import sys
import CLIP.clip as clip
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms


clip.clip._MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}


class CLIP_Loss(nn.Module):
    def __init__(self, augmentation_num=0, sim_scale=1, use_clip_normalization=True):
        super().__init__()
        clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        # CLIP Transform
        if use_clip_normalization:
            self.clip_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                clip_normalizer
            ])
        else:
            self.clip_transform = transforms.Compose([
                transforms.Resize((224, 224)),
            ])
        self.augment_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.8,0.95)),
                    transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
                    clip_normalizer
                    ])
        self.clip_model, preprocess = clip.load('ViT-B/32', jit=False, device='cpu')
        self.clip_model.eval()
        self.augmentation_num = augmentation_num
        self.sim_scale = sim_scale
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def encode_prompt_token(self, prompt_token):
        encoded_text = self.clip_model.encode_text(prompt_token).mean(dim=0, keepdim=True)
        return encoded_text


    def forward(self, image, prompt_token, unnormalize_image=True):
        penalty = 0.
        if unnormalize_image:
            image = (image+1)/2.
        _image = self.clip_transform(image)
        encoded_image = self.clip_model.encode_image(_image)
        encoded_text = self.clip_model.encode_text(prompt_token).mean(dim=0, keepdim=True)
        loss = torch.cosine_similarity(encoded_image, encoded_text)
        loss = 1-loss.mean() / self.sim_scale
        penalty += loss
        for i in range(self.augmentation_num):
            _image = self.augment_transform(image)
            encoded_image = self.clip_model.encode_image(_image)
            loss = torch.cosine_similarity(encoded_image, encoded_text)
            loss = 1-loss.mean() / self.sim_scale
            penalty += loss

        penalty = penalty / (self.augmentation_num+1)

        return penalty


    def forward_img(self, image, ref_images, unnormalize_image=True):
        penalty = 0.
        if unnormalize_image:
            image = (image+1)/2.
        _image = self.clip_transform(image)
        encoded_image = self.clip_model.encode_image(_image)
        #TODO?? encode_image 도 aug를 하는게??, 여기도 nce 로스처럼 적용

        assert len(ref_images) ==1 # for now
        for ref_image in ref_images:
            _ref_image = self.clip_transform(ref_image)
            ref_encoded_image = self.clip_model.encode_image(_ref_image)

        loss = torch.cosine_similarity(encoded_image, ref_encoded_image)
        loss = 1-loss.mean() / self.sim_scale
        penalty += loss
        for i in range(self.augmentation_num):
            _image = self.augment_transform(image)
            encoded_image = self.clip_model.encode_image(_image)
            loss = torch.cosine_similarity(encoded_image, ref_encoded_image)
            loss = 1-loss.mean() / self.sim_scale
            penalty += loss

        penalty = penalty / (self.augmentation_num+1)

        return penalty


    def forward_img_diff(self, image, image_origin, ref_images, unnormalize_image=True):
        penalty = 0.
        if unnormalize_image:
            image = (image+1)/2.
            image_origin = (image_origin+1)/2.


        _image = self.clip_transform(image)
        _image_origin = self.clip_transform(image_origin)
        encoded_image = self.clip_model.encode_image(_image)
        encoded_image_ori = self.clip_model.encode_image(_image_origin)

        assert len(ref_images) ==1 # for now
        for ref_image in ref_images:
            _ref_image = self.clip_transform(ref_image)
            ref_encoded_image = self.clip_model.encode_image(_ref_image)

        img_emb_diff = encoded_image - encoded_image_ori
        ref_emb_diff = ref_encoded_image - encoded_image_ori

        loss = torch.cosine_similarity(img_emb_diff, ref_emb_diff)
        loss = 1-loss.mean() / self.sim_scale
        penalty += loss
        for i in range(self.augmentation_num):
            _image = self.augment_transform(image)
            encoded_image = self.clip_model.encode_image(_image)
            loss = torch.cosine_similarity(encoded_image, ref_encoded_image)
            loss = 1-loss.mean() / self.sim_scale
            penalty += loss

        penalty = penalty / (self.augmentation_num+1)

        return penalty


    def forward_diff_only(self, image, image_origin, prompt_token, src_token, unnormalize_image=True):
        penalty = 0.
        if unnormalize_image:
            image = (image+1)/2.
            image_origin = (image_origin+1)/2.
        _image = self.clip_transform(image)
        _image_origin = self.clip_transform(image_origin)
        encoded_image = self.clip_model.encode_image(_image)
        encoded_image_ori = self.clip_model.encode_image(_image_origin)
        img_emb_diff = encoded_image - encoded_image_ori # (b, 512)

        encoded_text = self.clip_model.encode_text(prompt_token).mean(dim=0, keepdim=True) # (1, 512)
        encoded_text_src = self.clip_model.encode_text(src_token) # (tn, 512)
        text_emb_diff = encoded_text - encoded_text_src # (tn, 512)

        cross_emb_diff = encoded_text - encoded_image_ori # (b, 512)

        #neg_diff = encoded_text_src.unsqueeze(0) - encoded_image_ori.unsqueeze(1) # (b, tn, 512)

        #tlt = torch.inner(img_emb_diff, text_emb_diff.mean(dim=0)) #(b,)
        #tlc = torch.sum(img_emb_diff * cross_emb_diff, dim=-1) # (b,)
        #nl = torch.sum(img_emb_diff.unsqueeze(1) * neg_diff, dim=-1) # (b, tn, )

        #t_logits = torch.cat((tlt.unsqueeze(1), nl), dim=-1) # (b, tn+1)
        #c_logits = torch.cat((tlc.unsqueeze(1), nl), dim=-1) # (b, tn+1)
        #target = torch.zeros(t_logits.shape[0]).type(torch.LongTensor).to(t_logits.device)

        #loss = self.ce_loss(t_logits, target) + self.ce_loss(c_logits, target)

        it_sim = torch.cosine_similarity(img_emb_diff, text_emb_diff.mean(dim=0)) #(b,)
        c_sim = torch.cosine_similarity(img_emb_diff, cross_emb_diff) #(b,)
        loss = 2 - (it_sim.mean() + c_sim.mean()) / self.sim_scale

        #loss = 1-loss.mean() / self.sim_scale
        penalty += loss
        for i in range(self.augmentation_num):
            _image = self.augment_transform(image)
            encoded_image = self.clip_model.encode_image(_image)
            img_emb_diff = encoded_image - encoded_image_ori # (b, 512)

            #tlt = torch.inner(img_emb_diff, text_emb_diff.mean(dim=0)) #(b,)
            #tlc = torch.sum(img_emb_diff * cross_emb_diff, dim=-1) # (b,)
            #nl = torch.sum(img_emb_diff.unsqueeze(1) * neg_diff, dim=-1) # (b, tn, )

            #t_logits = torch.cat((tlt.unsqueeze(1), nl), dim=-1) # (b, tn+1)
            #c_logits = torch.cat((tlc.unsqueeze(1), nl), dim=-1) # (b, tn+1)
            #target = torch.zeros(t_logits.shape[0]).type(torch.LongTensor).to(t_logits.device)

            #loss = self.ce_loss(t_logits, target) + self.ce_loss(c_logits, target)
            it_sim = torch.cosine_similarity(img_emb_diff, text_emb_diff.mean(dim=0)) #(b,)
            c_sim = torch.cosine_similarity(img_emb_diff, cross_emb_diff) #(b,)
            loss = 2 - (it_sim.mean() + c_sim.mean()) / self.sim_scale

            penalty += loss

        penalty = penalty / (self.augmentation_num+1)

        return penalty

    def forward_diff(self, image, image_origin, prompt_token, src_token, unnormalize_image=True):
        penalty = 0.
        if unnormalize_image:
            image = (image+1)/2.
            image_origin = (image_origin+1)/2.
        _image = self.clip_transform(image)
        _image_origin = self.clip_transform(image_origin)
        encoded_image = self.clip_model.encode_image(_image)
        encoded_image_ori = self.clip_model.encode_image(_image_origin)
        img_emb_diff = encoded_image - encoded_image_ori # (b, 512)

        encoded_text = self.clip_model.encode_text(prompt_token).mean(dim=0, keepdim=True) # (1, 512)
        encoded_text_src = self.clip_model.encode_text(src_token) # (tn, 512)
        text_emb_diff = encoded_text - encoded_text_src # (tn, 512)

        cross_emb_diff = encoded_text - encoded_image_ori # (b, 512)

        neg_diff = encoded_text_src.unsqueeze(0) - encoded_image_ori.unsqueeze(1) # (b, tn, 512)

        tlt = torch.inner(img_emb_diff, text_emb_diff.mean(dim=0)) #(b,)
        tlc = torch.sum(img_emb_diff * cross_emb_diff, dim=-1) # (b,)
        nl = torch.sum(img_emb_diff.unsqueeze(1) * neg_diff, dim=-1) # (b, tn, )

        t_logits = torch.cat((tlt.unsqueeze(1), nl), dim=-1) # (b, tn+1)
        c_logits = torch.cat((tlc.unsqueeze(1), nl), dim=-1) # (b, tn+1)
        target = torch.zeros(t_logits.shape[0]).type(torch.LongTensor).to(t_logits.device)

        loss = self.ce_loss(t_logits, target) + self.ce_loss(c_logits, target)

        #loss = torch.cosine_similarity(encoded_image, encoded_text)
        #loss = 1-loss.mean() / self.sim_scale
        penalty += loss
        for i in range(self.augmentation_num):
            _image = self.augment_transform(image)
            encoded_image = self.clip_model.encode_image(_image)
            img_emb_diff = encoded_image - encoded_image_ori # (b, 512)

            tlt = torch.inner(img_emb_diff, text_emb_diff.mean(dim=0)) #(b,)
            tlc = torch.sum(img_emb_diff * cross_emb_diff, dim=-1) # (b,)
            nl = torch.sum(img_emb_diff.unsqueeze(1) * neg_diff, dim=-1) # (b, tn, )

            t_logits = torch.cat((tlt.unsqueeze(1), nl), dim=-1) # (b, tn+1)
            c_logits = torch.cat((tlc.unsqueeze(1), nl), dim=-1) # (b, tn+1)
            target = torch.zeros(t_logits.shape[0]).type(torch.LongTensor).to(t_logits.device)

            loss = self.ce_loss(t_logits, target) + self.ce_loss(c_logits, target)

            penalty += loss

        penalty = penalty / (self.augmentation_num+1)

        return penalty


class CLIP_VIS(nn.Module):
    def __init__(self, num_layers=10, ratio=None):
        super().__init__()

        self.model, self.preprocess = clip.load("ViT-B/32", device='cpu', jit=False)
        self.num_layers = num_layers

        clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        # CLIP Transform
        self.clip_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            clip_normalizer
        ])
        self.ratio=4
        if ratio:
            self.ratio = ratio
        
    def interpret(self, images, prompt_token, image_size):
        text_batch_size = prompt_token.shape[0]
        image_batch_size = images.shape[0]
        logits_per_image, logits_per_text = self.model(images, prompt_token)
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        index = [i for i in range(text_batch_size)]
        one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
        one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * logits_per_image)
        self.model.zero_grad()

        image_attn_blocks = list(dict(self.model.visual.transformer.resblocks.named_children()).values())
        #print(image_attn_blocks)
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).cuda()
        R = R.unsqueeze(0).expand(image_batch_size, num_tokens, num_tokens)
        for i, blk in enumerate(image_attn_blocks):
            if i <=self.num_layers:
              continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
            cam = blk.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(image_batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R = R + torch.bmm(cam, R)
        image_relevance = R[:, 0, 1:]
        image_relevance = image_relevance.reshape(image_batch_size, 1, 7, 7)

        image_relevance = torch.nn.functional.interpolate(image_relevance, size=image_size, mode='bilinear')
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min() + 1e-6)

        return image_relevance

    def forward(self, image, prompt_token, neg_prompt_token=None, unnormalize_image=True):
        image_size = int(image.shape[2]/self.ratio)
        batch_size = image.shape[0]
        if unnormalize_image:
            image = (image+1)/2.
        image = self.clip_transform(image)
        device = image.device
        irs = []
        for pt in prompt_token:
            _image_relevance = self.interpret(image, pt.unsqueeze(0), image_size) #(b,1,128,128)
            irs.append(_image_relevance)
        image_relevance = torch.cat(irs, dim=1).mean(dim=1, keepdim=True)

        if neg_prompt_token is not None:
            nirs = []
            for npt in neg_prompt_token:
                _neg_image_relevance = self.interpret(image, npt.unsqueeze(0), image_size)
                nirs.append(_neg_image_relevance)
            neg_image_relevance = torch.cat(nirs, dim=1).mean(dim=1, keepdim=True)
            image_relevance = image_relevance - neg_image_relevance
            image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min() + 1e-6)

        return image_relevance

        
'''
device = 'cuda:0'
img = torch.randn(2,3,256,256)
tanh = nn.Tanh()
img = tanh(img)


cliploss=CLIP_VIS()
cliploss = cliploss.to(device)
prompt_token = clip.tokenize(["I am Junha"])
loss = cliploss(img.to(device), prompt_token.to(device))
print(loss)
print(loss.shape)
'''
