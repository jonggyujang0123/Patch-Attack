from torchvision.transforms import Pad 
import torch
from einops import rearrange
import numpy as np

class patch_util():
    def __init__(
            self,
            img_size = 32,
            patch_size = 4,
            patch_margin = 2,
            device = 'cuda:0'
            ):
        self.img_size = img_size
        self.device =device
        self.patch_size = patch_size
        self.patch_margin = patch_margin
        self.num_patches = (img_size//patch_size) **2
        self.pad = Pad(self.patch_margin, fill=0)
        self.get_mask()

    def get_mask(self):
        patch_arange = (1-torch.triu(torch.ones([self.num_patches,self.num_patches]),diagonal = 0)).unsqueeze(2).unsqueeze(3)
#        patch_arange = (torch.triu(torch.ones([self.num_patches,self.num_patches]))).unsqueeze(2).unsqueeze(3)
        one_patch  = torch.ones([1,1,self.patch_size, self.patch_size])
        C = patch_arange *one_patch

        self.mask = rearrange(
                C, 
                'i (w1 h1) w2 h2 -> i 1 (w1 w2) (h1 h2)', 
                w1 = self.img_size//self.patch_size, 
                w2 = self.patch_size
                ).to(self.device)
        # mask : (num_patch, 1, patch_size, patch_size)
        # mask[0,:] : first mask

        self.mask_to_extended_patch = torch.ones([1, 1, 2 * self.patch_margin + self.patch_size, 2 * self.patch_margin + self.patch_size]).to(self.device)
        self.mask_to_extended_patch[:,:,
                self.patch_margin ::,
                self.patch_margin ::] = 0

    def get_patch(self, img, pat_ind = None):
        bs = img.shape[0]
        if pat_ind == None:
            pat_ind = torch.tensor(np.random.choice(self.num_patches,size = (bs,), replace=True)).to(self.device)

        ## Generate Patch Images
        patch = img.unfold(2, self.patch_size, self.patch_size).unfold(3,self.patch_size, self.patch_size)
        patch = patch.reshape(bs, img.shape[1], self.num_patches, self.patch_size, self.patch_size)
        patch = patch[range(len(pat_ind)), :, pat_ind, :, :]

        ## Generated Masked Conditional Images
        mask = self.mask[pat_ind, :]
        img_masked = self.pad(img * mask)
#        img_masked = self.pad(img * mask + (1-mask) * (-1) )
        img_masked = img_masked.unfold(2, self.patch_size + self.patch_margin *2, self.patch_size).unfold(3, self.patch_size + self.patch_margin *2, self.patch_size)
        img_masked = img_masked.reshape(bs, img.shape[1], self.num_patches, self.patch_size + self.patch_margin*2, self.patch_size + self.patch_margin*2)
        # (bs, ch, num_patch, overlap_size, overlap_size)
        img_masked = img_masked[range(len(pat_ind)), :, pat_ind, :, :]
        origin = img_masked+0.0
        origin[:,:,
                self.patch_margin:self.patch_size+self.patch_margin,
                self.patch_margin:self.patch_size+self.patch_margin] = patch
        # (bs, ch, overlap_size, overlap_size)
        img_masked = img_masked * self.mask_to_extended_patch #- 1.0 * (1-self.mask_to_extended_patch)
#        emb_size = self.channel * (3 * patch_margin ** 2 + 2 * patch_margin * patch_size)
#        img_masked = img_masked[img_masked>-100].reshape([img_masked.shape[0],-1])
#        print(img_masked.shape)
        # (bs, len_conditional)

        return patch, img_masked, pat_ind, origin

    def concat_patch(self, img, patch, pat_ind):
        img = rearrange(img, 'b c (w1 w2) (h1 h2) -> b c (w1 h1) w2 h2', w2 = self.patch_size, h2 = self.patch_size)
        img[range(len(pat_ind)), :, pat_ind, :, :] = patch
        img = rearrange(img, 'b c (w1 h1) w2 h2 -> b c (w1 w2) (h1 h2)', w1 =self.img_size//self.patch_size)
        return img

    def concat_extended_patch(self, pat, cond_img):
        cond_img[:,:,
                self.patch_margin:self.patch_size+self.patch_margin,
                self.patch_margin:self.patch_size+self.patch_margin] = pat
        return cond_img 
