import os
import clip
import wandb
import torch
import dnnlib
import legacy
import lpips
import imageio
import numpy as np
import mrcfile
from config.return_kwargs import return_kwargs
from training.clipedit.cliploss import CLIP_VIS, CLIP_Loss
from training.clipedit.mapper import EmbedCond
from torch_utils import misc
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import torch.utils.data as utils_data
from torch.nn.parallel import DistributedDataParallel as DDP
from camera_utils import LookAtPoseSampler
from arcface import arcface_loss
import PIL.Image
#from training.cli
#from training.clipedit import masknet, mapper


class EG3DMaskEditTrainer(object):
    TRAINER_NAME = "EG3DMaskEditTrainer"

    def __init__(self, configs, rank, world_size, mode, random_seed):
        self.configs = configs
        self.rank = rank
        self.random_seed = random_seed
        self.define_model()
        
        batch_size = self.configs['training']['batch_size']
        multiview_num = self.configs['training']['multiview_num']
        batch_gpu = batch_size // self.configs['ngpus']
        
        self.model_type = self.configs['model']['generator']['model_type']

        if batch_size % self.configs['ngpus'] != 0:
            print(f"batch size {batch_size} is not multiple of ngpus {self.configs['ngpus']}")
            assert False

        if batch_size != (batch_gpu * self.configs['ngpus']):
            print(f"batch size {batch_size} is not equal to ngpus*batch_gpu {self.configs['ngpus']*batch_gpu}")
            assert False

        self.batch_gpu = batch_gpu
        self.multiview_num = multiview_num

        print('=================================')
        print(f'total batch size: {batch_size}')
        print(f'batch_gpu: {batch_gpu}')
        print(f'multiview_num: {multiview_num}')
        print('=================================')

        self.define_label_loader()
        self.set_prompt()
        self.define_optimizer()

        if mode == 'train' and not self.configs['experiment']['debug']:
            self.use_wandb = True
        else:
            self.use_wandb = False

        self.conditioning_cam2world_pose = None
        self.intrinsics = None
        self.cam_pivot = None
        self.cam_radius = None
        self.conditioning_params = None

    def init_wandb(self):
        if self.use_wandb:
            config_dict = {}
            for key, value in self.configs.items():
                config_dict[str(key)] = value

            exp_name = self.configs['experiment']['exp_name']
            entity = self.configs['wandb']['entity']
            log_dir = self.configs['experiment']['exp_dir']
            wandb.init(
                entity = entity,
                project = exp_name,
                dir = log_dir,
                config = config_dict
            )

            wandb.run.name = self.configs['wandb']['wandb_run_name'] + f'{self.configs["experiment"]["exp_dir"]}'


    def set_prompt(self):
        self.edit_prompt_token = None
        self.mask_prompt_token = None
        self.neg_mask_prompt_token = None
        self.neg_edit_prompt_token = None

        edit_prompt = self.configs['prompt']['edit_prompt']
        print(f"edit prompt: {edit_prompt}")
        self.edit_prompt_token = clip.tokenize(edit_prompt).cuda()

        if 'mask_prompt' in self.configs['prompt']:
            mask_prompt = self.configs['prompt']['mask_prompt']
            print(f"mask prompt: {mask_prompt}")
            self.mask_prompt_token = clip.tokenize(mask_prompt).cuda()

        if 'neg_mask_prompt' in self.configs['prompt']:
            neg_mask_prompt = self.configs['prompt']['neg_mask_prompt']
            print(f"negative mask prompt: {neg_mask_prompt}")
            self.neg_mask_prompt_token = clip.tokenize(neg_mask_prompt).cuda()

        if 'neg_edit_prompt' in self.configs['prompt']:
            neg_edit_prompt = self.configs['prompt']['neg_edit_prompt']
            print(f"negative edit prompt: {neg_edit_prompt}")
            self.neg_edit_prompt_token = clip.tokenize(neg_edit_prompt).cuda()

        if 'src_prompt' in self.configs['prompt']:
            src_prompt = self.configs['prompt']['src_prompt']
            print(f"source prompt: {src_prompt}")
            self.src_prompt_token = clip.tokenize(src_prompt).cuda()

        if self.mapper_type == 'ec':
            self.mapper_mlp_ddp.module.embed = self.clip_loss_ddp.module.encode_prompt_token(self.edit_prompt_token).detach()
            self.mapper_mlp_ddp.module.embed.requires_grad = False


    def define_model(self):

        self.G_kwargs, self.common_kwargs, self.neural_rendering_resolution = return_kwargs(self.configs['model']['generator']['model_type'])
        self.img_size = self.G_kwargs['rendering_kwargs']['image_resolution']
        self.generator = dnnlib.util.construct_class_by_name(**self.G_kwargs, **self.common_kwargs).requires_grad_(False).cuda()
        self.generator.neural_rendering_resolution = self.neural_rendering_resolution

        self.use_mask = self.configs['model']['masknet'].get('use_mask', True)
        self.use_masked_cliploss = self.configs['lambda'].get('use_masked_cliploss', False)
        self.masknet_type = self.configs['model']['masknet'].get('masknet_type', 'default') # default, ws_condition
        self.configs['model']['masknet']['masknet_type'] = self.masknet_type

        if self.use_mask:
            print('====use masknet====')
            print(f"====masknet type: {self.masknet_type}====")
            masknet_kwargs = dnnlib.EasyDict(self.configs['model']['masknet'])

            if self.configs['model']['masknet']['use_coor_input']:
                masknet_kwargs.feat_dim = self.generator.decoder.total_feat_dim + 3
            else:
                masknet_kwargs.feat_dim = self.generator.decoder.total_feat_dim

            self.masknet_mlp = dnnlib.util.construct_class_by_name(**masknet_kwargs).cuda()

        self.use_deform = False
        if 'deform' in self.configs['model']:
            self.use_deform = True
            print('====use deform mlp====')
            deform_kwargs = dnnlib.EasyDict(self.configs['model']['deform'])

            deform_kwargs.feat_dim = 3
            self.deform_mlp = dnnlib.util.construct_class_by_name(**deform_kwargs).cuda()


        mapper_kwargs = dnnlib.EasyDict(self.configs['model']['mapper'])
        mapper_kwargs.num_ws = self.generator.backbone.synthesis.num_ws
        mapper_kwargs.mapper_type = self.configs['model']['mapper'].get('mapper_type', None)
        self.mapper_type = mapper_kwargs.mapper_type
        #print(self.generator)
        self.mapper_mlp = dnnlib.util.construct_class_by_name(**mapper_kwargs).cuda()

        if self.configs['model']['generator']['model_type'] == 'shapenetcars128-64':
            self.clip_vis = CLIP_VIS(ratio=2).cuda()
        else:
            self.clip_vis = CLIP_VIS().cuda()

        self.clip_loss = CLIP_Loss(self.configs['lambda']['clip_augmentation_num'], self.configs['lambda']['sim_scale']).cuda()
        self.arcface_loss = arcface_loss.ArcFaceLoss()

        if self.configs['lambda']['lpips_lambda'] > 0:
            print('load lpips model')
            self.lpips = lpips.exportPerceptualLoss(model="net-lin", net="vgg").cuda()

        self.mseloss = torch.nn.MSELoss()
        self.upsample = torch.nn.Upsample(size=self.img_size, mode='bilinear')

        if not self.configs['experiment']['debug']:
            if 'pti_pkl' in self.configs:
                other = self.configs['pti_pkl']
            else:
                other = None
            self.load_pretrained(other)

        self.generator.requires_grad_(True)
        self.generator_ddp = DDP(self.generator, device_ids=[self.rank])
        if self.use_mask:
            self.masknet_mlp_ddp = DDP(self.masknet_mlp, device_ids=[self.rank])
        if self.use_deform:
            self.deform_mlp_ddp = DDP(self.deform_mlp, device_ids=[self.rank])
        else:
            self.deform_mlp_ddp = None
        self.inv_thres = self.configs['model']['masknet'].get('inv_thres', None)
        self.thres = self.configs['model']['masknet'].get('thres', None)
        if self.inv_thres is not None:
            print(f'====inv_thres====! {self.inv_thres}')
        if self.thres is not None:
            print(f'====thres====! {self.thres}')
        self.mask_lpf = self.configs['model']['masknet'].get('mask_lpf', False)
        if self.mask_lpf:
            print(f'====use mask lpf====!')

        
        self.ws_edit_cond = self.configs['model']['generator'].get('ws_edit_cond', False)
        if self.ws_edit_cond:
            print(f'====use ws edit cond====')

        self.mapper_mlp_ddp = DDP(self.mapper_mlp, device_ids=[self.rank])
        self.clip_vis_ddp = DDP(self.clip_vis, device_ids=[self.rank])
        self.clip_loss_ddp = DDP(self.clip_loss, device_ids=[self.rank], find_unused_parameters=True)
        self.arcface_loss_ddp = DDP(self.arcface_loss, device_ids=[self.rank], find_unused_parameters=True)


    def load_pretrained(self, other=None):
        if other is not None:
            network_pkl = other
            ckpt = torch.load(other)
            self.generator.load_state_dict(ckpt['G_ema'], strict=False)

        else:
            network_pkl = './networks/ffhq512-128.pkl'
            if self.configs['model']['generator']['model_type'] == 'afhqcats512-128':
                network_pkl = './networks/afhqcats512-128.pkl'
            elif self.configs['model']['generator']['model_type'] == 'shapenetcars128-64':
                network_pkl = './networks/shapenetcars128-64.pkl'

            with dnnlib.util.open_url(network_pkl) as f:
                pretrained_data = legacy.load_network_pkl(f)
            for name, module in [('G_ema', self.generator)]:
                misc.copy_params_and_buffers(pretrained_data[name], module, require_all=True)

        print(f'Loading pretrained model from {network_pkl}')

    def define_optimizer(self):
        if self.use_mask:
            masknet_parameters = [p for p in self.masknet_mlp_ddp.parameters()]
        mapper_parameters = [p for p in self.mapper_mlp_ddp.parameters()]
        if self.use_deform:
            mapper_parameters += [p for p in self.deform_mlp_ddp.parameters()]

        if self.use_mask:
            self.optimizer_masknet = torch.optim.Adam(masknet_parameters, lr=self.configs['training']['lr'], betas=self.configs['training']['betas'], weight_decay=self.configs['training']['weight_decay'])
        self.optimizer_mapper = torch.optim.Adam(mapper_parameters, lr=self.configs['training']['lr'], betas=self.configs['training']['betas'], weight_decay=self.configs['training']['weight_decay'])


    def define_label_loader(self):
        dataset_kwargs = dnnlib.EasyDict(self.configs['dataset'])

        dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
        self.dataset = dataset

        self.train_sampler = misc.InfiniteSampler(dataset=dataset, rank=self.rank, num_replicas=self.configs['ngpus'], seed=self.random_seed)
        '''
        if self.configs['ngpus'] > 1:
            #self.train_sampler = utils_data.distributed.DistributedSampler(dataset, drop_last=True)
            shuffle = False
        else:
            self.train_sampler = None
            shuffle = True
        '''

        self.train_set_iterator = iter(utils_data.DataLoader(
                dataset, batch_size=self.batch_gpu*self.multiview_num,
                num_workers=int(self.configs['training']['n_workers']),
                sampler=self.train_sampler))


    def training(self):
        print("Start training !!!!")
        swapping_prob = gpc_reg_prob = self.G_kwargs['rendering_kwargs']['gpc_reg_prob']
        self.step = 0
        progress_bar = tqdm(total=self.configs['training']['max_step'], desc="Progress in steps")

        while True:

            gen_c = next(self.train_set_iterator)
            gen_c = gen_c.cuda()

            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand((gen_c.shape[0], 1), device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)


            self.generator_ddp.train()
            self.mapper_mlp_ddp.train()
            if self.use_mask:
                self.masknet_mlp_ddp.train()

            if self.use_mask:
                loss_masknet, logs_masknet = self.train_masknet(gen_c, c_gen_conditioning)
            loss_mapper, logs_mapper = self.train_mapper(gen_c, c_gen_conditioning)

            if self.use_mask:
                loss_sum = loss_masknet + loss_mapper
                logs_masknet.update(logs_mapper)
                total_logs = logs_masknet
            else:
                loss_sum = loss_mapper
                total_logs = logs_mapper


            if self.rank == 0 and self.step < 20000:
                self.display_save_result(self.step, loss_sum, total_logs)

            self.step += 1
            progress_bar.update(1)

    def display_save_result(self, step, loss, logs):
        if step % 10 == 0:
            tqdm.write(
                    f"[Experiment: {self.configs['experiment']['exp_dir']}]"
                    f"[GPU: {self.rank}] [step: {step}/{self.configs['training']['max_step']}]"
                    f"[loss: {loss}]"
                    )
        if self.use_wandb:
            wandb_log = {
                'Step': step,
                'train_loss_sum': loss,
            }
            if len(logs.keys()) > 0:
                for key, value in logs.items():
                    wandb_log.update({key: value})
            wandb.log(wandb_log)

            if step % self.configs['experiment']['sample_interval'] == 0:
                if self.model_type == 'shapenetcars128-64':
                    self.display_image_to_wandb_cars()
                else:
                    self.display_image_to_wandb()

        if step % self.configs['experiment']['model_save_interval'] == 0:
            self.save_models(step)

    def save_models(self, step):
        if self.use_mask:
            if self.use_deform:
                save_dict = {'masknet': self.masknet_mlp_ddp.module,
                             'mapper': self.mapper_mlp_ddp.module,
                             'deform': self.deform_mlp_ddp.module}
            else:
                save_dict = {'masknet': self.masknet_mlp_ddp.module,
                             'mapper': self.mapper_mlp_ddp.module}
        else:
            save_dict = {'mapper': self.mapper_mlp_ddp.module}


        for key, value in save_dict.items():
            save_name = os.path.join(self.configs['experiment']['exp_dir'], f"{step}-{key}.pth")
            torch.save(value.state_dict(), save_name) 
            print(f"saved model check point: {save_name}")

    def display_image_to_wandb_cars(self):

        N_images = 5
        truncation_psi = 0.7

        self.generator_ddp.eval()
        self.mapper_mlp_ddp.eval()
        if self.use_deform:
            self.deform_mlp_ddp.eval()
        if self.use_mask:
            self.masknet_mlp_ddp.eval()


        if self.intrinsics is None:
            '''
            self.intrinsics = torch.tensor([
                [4.2647, 0.0, 0.5],
                [0.0, 4.2647, 0.5],
                [0.0, 0.0, 1.0]
                ], device=self.rank)
            '''
            self.intrinsics = torch.tensor([
                [1.7074, 0.0, 0.5],
                [0.0, 1.7074, 0.5],
                [0.0, 0.0, 1.0]
                ], device=self.rank)

        if self.cam_pivot is None:
            self.cam_pivot = torch.tensor(self.G_kwargs['rendering_kwargs']['avg_camera_pivot'], device=self.rank)

        if self.cam_radius is None:
            self.cam_radius = self.G_kwargs['rendering_kwargs']['avg_camera_radius']

        if self.conditioning_cam2world_pose is None:
            self.conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, self.cam_pivot, radius=self.cam_radius, device=self.rank)

        if self.conditioning_params is None:
            self.conditioning_params = torch.cat([self.conditioning_cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1,9)], 1)
            if self.G_kwargs['rendering_kwargs']['gpc_reg_prob'] is None:
                self.conditioning_params = torch.zeros_like(self.conditioning_params)

        recover_ns = self.generator_ddp.module.rendering_kwargs['depth_resolution']
        self.generator_ddp.module.rendering_kwargs['depth_resolution'] = recover_ns*2
        recover_nsi = self.generator_ddp.module.rendering_kwargs['depth_resolution_importance']
        self.generator_ddp.module.rendering_kwargs['depth_resolution_importance'] = recover_nsi*2
        image_log = {}
        
        imgs = []
        for _ in range(N_images):
            z = torch.randn([1, self.generator.z_dim]).cuda()
            angle_ps = [-3.14/2+0.5, 0, 3.14/2-0.5, 3.14/2*3-1, 3.14/2*3 + 1.]
            ys = [0] * 5
            ys[-2] = 3.14/2
            ys[-1] = -3.14/2
            for angle_y, angle_p in [(ys[0], angle_ps[0]), (ys[1], angle_ps[1]), (ys[2], angle_ps[2])]:
                with torch.no_grad():
                    cam2world_pose = LookAtPoseSampler.sample(np.pi/2+angle_y, np.pi/2 + angle_p, self.cam_pivot, radius=self.cam_radius, device=self.rank)

                    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1,9)], 1)

                    ws = self.generator_ddp.module.mapping(z, self.conditioning_params, truncation_psi=truncation_psi)
                    ws_delta = self.mapper_mlp_ddp(ws)
                    ws_edit = ws + self.configs['lambda']['ws_delta_lambda']*ws_delta

                    with torch.no_grad():
                        raw_edit_out = self.generator_ddp.module.synthesis_get_mask_features(ws_edit, camera_params, noise_mode='const')
                        delta_raw_img = raw_edit_out['image']

                    if self.use_mask:
                        out = self.generator_ddp.module.synthesis_get_mask_features(ws, camera_params, self.deform_mlp_ddp, ws_condition=ws_edit, noise_mode='const')

                        N, HW, steps, feat_dim = out['all_feats'].shape
                        if self.configs['model']['masknet']['use_coor_input']:
                            mask_input = torch.cat([out['all_feats'].reshape(-1, feat_dim), out['all_coor'].reshape(-1, 3)], dim=-1)
                            if self.masknet_type == 'ws_condition':
                                mask = self.masknet_mlp_ddp(mask_input, ws, ws_edit)
                            elif self.masknet_type == 'attention':
                                mask_input_edit = torch.cat([raw_edit_out['all_feats'].reshape(-1, feat_dim), raw_edit_out['all_coor'].reshape(-1,3)], dim=-1)
                                mask = self.masknet_mlp_ddp(mask_input, mask_input_edit, ws, ws_edit)
                            elif self.masknet_type == 'default':
                                mask = self.masknet_mlp_ddp(mask_input)
                            else:
                                raise NotImplementedError
                        else:
                            mask_input = out['all_feats'].reshape(-1, feat_dim)
                            mask = self.masknet_mlp_ddp(mask_input)
                        mask = mask.reshape(N, HW, steps, 1)

                        if self.mask_lpf:
                            H = W = self.neural_rendering_resolution
                            assert HW == H*W
                            mask = mask.squeeze(-1).reshape(N, H, W, steps).permute(0, 3, 1, 2)
                            mask = transforms.GaussianBlur(17, sigma=15)(mask)
                            mask = mask.reshape(N, steps, HW, 1).permute(0, 2, 1, 3)
                        if self.inv_thres is not None:
                            mask2d = self.generator_ddp.module.render_mask(mask, out)
                            ones = torch.ones_like(mask)
                            mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) > self.inv_thres, ones, mask)
                        if self.thres is not None:
                            mask2d = self.generator_ddp.module.render_mask(mask, out)
                            zeros = torch.zeros_like(mask)
                            mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) < self.thres, zeros, mask)

                        mask2d = self.generator_ddp.module.render_mask(mask, out)
                        mask2d = mask2d.reshape(-1, self.neural_rendering_resolution, self.neural_rendering_resolution, 1).permute(0,3,1,2)
                        mask2d = self.upsample(mask2d)
                        mask2d = mask2d.repeat(1,3,1,1)

                    else:
                        out = self.generator_ddp.module.synthesis(ws, camera_params, noise_mode='const')


                image_relevance = self.clip_vis_ddp(out['image'], self.mask_prompt_token, self.neg_mask_prompt_token) # (N, 1, H, W)
                if self.configs['model']['masknet']['use_enlarged_mask']:
                    image_relevance_edit = self.clip_vis_ddp(delta_raw_img, self.mask_prompt_token, self.neg_mask_prompt_token) # (N, 1, H, W)
                    image_relevance = torch.max(image_relevance, image_relevance_edit)


                with torch.no_grad():

                    image_relevance = image_relevance * 2 - 1. # unnormalize
                    image_relevance = self.upsample(image_relevance)
                    image_relevance = image_relevance.repeat(1,3,1,1)


                    if self.use_mask:

                        out_edit = self.generator_ddp.module.synthesis_with_mask(ws, ws_edit, camera_params, out, mask, mix_layers=self.configs['model']['generator']['mix_layers'], noise_mode='const', ws_edit_cond=self.ws_edit_cond)

                    imgs.append(image_relevance)

                    imgs.append(delta_raw_img)

                    img_origin = out['image']
                    imgs.append(img_origin)

                    if self.use_mask:

                        imgs.append(mask2d)

                        # low res image
                        img_raw = out_edit['image_raw']
                        img_raw = self.upsample(img_raw)
                        imgs.append(img_raw)

                        img = out_edit['image']
                        imgs.append(img)


                        if self.use_masked_cliploss:
                            masked_img = torch.where(mask2d > 0., img, -1*torch.ones_like(img))
                            imgs.append(masked_img)

                #img_int = (img.permute(0,2,3,1)*127.5 + 128).clamp(0,255).to(torch.uint8)
                #imgs_int.append(img)
        img = torch.cat(imgs, dim=0)
        numel = int(len(img) / N_images)
        img = torch.clamp(img, min=-1, max=1)
        img = make_grid(img, normalize=True, nrow=numel)
        save_image(img, os.path.join(self.configs['experiment']['exp_dir'], f"{self.step}.png"))
        image_log['image'] = wandb.Image(img)

        wandb.log(image_log)
        self.generator_ddp.module.rendering_kwargs['depth_resolution'] = recover_ns
        self.generator_ddp.module.rendering_kwargs['depth_resolution_importance'] = recover_nsi


    def display_image_to_wandb(self):

        N_images = 5
        truncation_psi = 0.7

        self.generator_ddp.eval()
        self.mapper_mlp_ddp.eval()
        if self.use_deform:
            self.deform_mlp_ddp.eval()
        if self.use_mask:
            self.masknet_mlp_ddp.eval()


        if self.intrinsics is None:
            self.intrinsics = torch.tensor([
                [4.2647, 0.0, 0.5],
                [0.0, 4.2647, 0.5],
                [0.0, 0.0, 1.0]
                ], device=self.rank)

        if self.cam_pivot is None:
            self.cam_pivot = torch.tensor(self.G_kwargs['rendering_kwargs']['avg_camera_pivot'], device=self.rank)

        if self.cam_radius is None:
            self.cam_radius = self.G_kwargs['rendering_kwargs']['avg_camera_radius']

        if self.conditioning_cam2world_pose is None:
            self.conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, self.cam_pivot, radius=self.cam_radius, device=self.rank)

        if self.conditioning_params is None:
            self.conditioning_params = torch.cat([self.conditioning_cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1,9)], 1)

        image_log = {}
        
        imgs = []
        for _ in range(N_images):
            z = torch.randn([1, self.generator.z_dim]).cuda()
            angle_p = -0.2
            for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
                with torch.no_grad():
                    cam2world_pose = LookAtPoseSampler.sample(np.pi/2+angle_y, np.pi/2 + angle_p, self.cam_pivot, radius=self.cam_radius, device=self.rank)

                    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1,9)], 1)

                    ws = self.generator_ddp.module.mapping(z, self.conditioning_params, truncation_psi=truncation_psi)
                    ws_delta = self.mapper_mlp_ddp(ws)
                    ws_edit = ws + self.configs['lambda']['ws_delta_lambda']*ws_delta

                    with torch.no_grad():
                        raw_edit_out = self.generator_ddp.module.synthesis_get_mask_features(ws_edit, camera_params, noise_mode='const')
                        delta_raw_img = raw_edit_out['image']

                    if self.use_mask:
                        out = self.generator_ddp.module.synthesis_get_mask_features(ws, camera_params, self.deform_mlp_ddp, ws_condition=ws_edit, noise_mode='const')

                        N, HW, steps, feat_dim = out['all_feats'].shape
                        if self.configs['model']['masknet']['use_coor_input']:
                            mask_input = torch.cat([out['all_feats'].reshape(-1, feat_dim), out['all_coor'].reshape(-1, 3)], dim=-1)
                            if self.masknet_type == 'ws_condition':
                                mask = self.masknet_mlp_ddp(mask_input, ws, ws_edit)
                            elif self.masknet_type == 'attention':
                                mask_input_edit = torch.cat([raw_edit_out['all_feats'].reshape(-1, feat_dim), raw_edit_out['all_coor'].reshape(-1,3)], dim=-1)
                                mask = self.masknet_mlp_ddp(mask_input, mask_input_edit, ws, ws_edit)
                            elif self.masknet_type == 'default':
                                mask = self.masknet_mlp_ddp(mask_input)
                            else:
                                raise NotImplementedError
                        else:
                            mask_input = out['all_feats'].reshape(-1, feat_dim)
                            mask = self.masknet_mlp_ddp(mask_input)
                        mask = mask.reshape(N, HW, steps, 1)

                        if self.mask_lpf:
                            H = W = self.neural_rendering_resolution
                            assert HW == H*W
                            mask = mask.squeeze(-1).reshape(N, H, W, steps).permute(0, 3, 1, 2)
                            mask = transforms.GaussianBlur(17, sigma=15)(mask)
                            mask = mask.reshape(N, steps, HW, 1).permute(0, 2, 1, 3)
                        if self.inv_thres is not None:
                            mask2d = self.generator_ddp.module.render_mask(mask, out)
                            ones = torch.ones_like(mask)
                            mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) > self.inv_thres, ones, mask)
                        if self.thres is not None:
                            mask2d = self.generator_ddp.module.render_mask(mask, out)
                            zeros = torch.zeros_like(mask)
                            mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) < self.thres, zeros, mask)

                        mask2d = self.generator_ddp.module.render_mask(mask, out)
                        mask2d = mask2d.reshape(-1, self.neural_rendering_resolution, self.neural_rendering_resolution, 1).permute(0,3,1,2)
                        mask2d = self.upsample(mask2d)
                        mask2d = mask2d.repeat(1,3,1,1)

                    else:
                        out = self.generator_ddp.module.synthesis(ws, camera_params, noise_mode='const')


                image_relevance = self.clip_vis_ddp(out['image'], self.mask_prompt_token, self.neg_mask_prompt_token) # (N, 1, H, W)
                if self.configs['model']['masknet']['use_enlarged_mask']:
                    image_relevance_edit = self.clip_vis_ddp(delta_raw_img, self.mask_prompt_token, self.neg_mask_prompt_token) # (N, 1, H, W)
                    image_relevance = torch.max(image_relevance, image_relevance_edit)


                with torch.no_grad():

                    image_relevance = image_relevance * 2 - 1. # unnormalize
                    image_relevance = self.upsample(image_relevance)
                    image_relevance = image_relevance.repeat(1,3,1,1)


                    if self.use_mask:

                        out_edit = self.generator_ddp.module.synthesis_with_mask(ws, ws_edit, camera_params, out, mask, mix_layers=self.configs['model']['generator']['mix_layers'], noise_mode='const', ws_edit_cond=self.ws_edit_cond)

                    imgs.append(image_relevance)

                    imgs.append(delta_raw_img)

                    img_origin = out['image']
                    imgs.append(img_origin)

                    if self.use_mask:

                        imgs.append(mask2d)

                        # low res image
                        img_raw = out_edit['image_raw']
                        img_raw = self.upsample(img_raw)
                        imgs.append(img_raw)

                        img = out_edit['image']
                        imgs.append(img)


                        if self.use_masked_cliploss:
                            masked_img = torch.where(mask2d > 0., img, -1*torch.ones_like(img))
                            imgs.append(masked_img)

        img = torch.cat(imgs, dim=0)
        numel = int(len(img) / N_images)
        img = torch.clamp(img, min=-1, max=1)
        img = make_grid(img, normalize=True, nrow=numel)
        save_image(img, os.path.join(self.configs['experiment']['exp_dir'], f"{self.step}.png"))
        image_log['image'] = wandb.Image(img)

        wandb.log(image_log)


    def inference_random(self, zs, name, noise_mode, save_origin=False):
        zs = zs.cuda()
        N_images = zs.shape[0]
        #truncation_psi = 0.5
        truncation_psi = 1
        save_dir = f'./evaluation/{self.configs["experiment"]["random_seeds"]}/{name}/edit_random_noisemode{noise_mode}'
        save_nm_dir = f'./evaluation/{self.configs["experiment"]["random_seeds"]}/{name}_nm/edit_random_noisemode{noise_mode}'
        save_origin_dir = f'./evaluation/{self.configs["experiment"]["random_seeds"]}/origin_random_noisemode{noise_mode}'
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_origin_dir, exist_ok=True)
        os.makedirs(save_nm_dir, exist_ok=True)
        print(f'start inference random // save dir: {save_dir}')

        self.generator_ddp.eval()
        self.mapper_mlp_ddp.eval()
        if self.use_deform:
            self.deform_mlp_ddp.eval()

        if self.use_mask:
            self.masknet_mlp_ddp.eval()

        if self.intrinsics is None:
            self.intrinsics = torch.tensor([
                [4.2647, 0.0, 0.5],
                [0.0, 4.2647, 0.5],
                [0.0, 0.0, 1.0]
                ], device=self.rank)

        self.train_set_iterator = iter(utils_data.DataLoader(
                self.dataset, batch_size=1,
                num_workers=int(self.configs['training']['n_workers']),
                sampler=self.train_sampler))


        for id in tqdm(range(N_images)):
            z = zs[id:id+1]
            #z = torch.randn([1, self.generator.z_dim]).cuda()
            c = next(self.train_set_iterator).cuda()

            with torch.no_grad():
                ws = self.generator_ddp.module.mapping(z, c, truncation_psi=truncation_psi)
                ws_delta = self.mapper_mlp_ddp(ws)
                ws_edit = ws + self.configs['lambda']['ws_delta_lambda']*ws_delta

                raw_edit_out = self.generator_ddp.module.synthesis_get_mask_features(ws_edit, c, noise_mode=noise_mode)
                raw_img_edit = raw_edit_out['image']

                if self.use_mask:
                    out = self.generator_ddp.module.synthesis_get_mask_features(ws, c, self.deform_mlp_ddp, ws_condition=ws_edit, noise_mode=noise_mode)

                    N, HW, steps, feat_dim = out['all_feats'].shape
                    if self.configs['model']['masknet']['use_coor_input']:
                        mask_input = torch.cat([out['all_feats'].reshape(-1, feat_dim), out['all_coor'].reshape(-1, 3)], dim=-1)
                        if self.masknet_type == 'ws_condition':
                            mask = self.masknet_mlp_ddp(mask_input, ws, ws_edit)
                        elif self.masknet_type == 'attention':
                            mask_input_edit = torch.cat([raw_edit_out['all_feats'].reshape(-1, feat_dim), raw_edit_out['all_coor'].reshape(-1,3)], dim=-1)
                            mask = self.masknet_mlp_ddp(mask_input, mask_input_edit, ws, ws_edit)
                        elif self.masknet_type == 'default':
                            mask = self.masknet_mlp_ddp(mask_input)
                        else:
                            raise NotImplementedError
                    else:
                        mask_input = out['all_feats'].reshape(-1, feat_dim)
                        mask = self.masknet_mlp_ddp(mask_input)
                    mask = mask.reshape(N, HW, steps, 1)

                    if self.mask_lpf:
                        H = W = self.neural_rendering_resolution
                        assert HW == H*W
                        mask = mask.squeeze(-1).reshape(N, H, W, steps).permute(0, 3, 1, 2)
                        mask = transforms.GaussianBlur(17, sigma=15)(mask)
                        mask = mask.reshape(N, steps, HW, 1).permute(0, 2, 1, 3)
                    if self.inv_thres is not None:
                        mask2d = self.generator_ddp.module.render_mask(mask, out)
                        ones = torch.ones_like(mask)
                        mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) > self.inv_thres, ones, mask)
                    if self.thres is not None:
                        mask2d = self.generator_ddp.module.render_mask(mask, out)
                        zeros = torch.zeros_like(mask)
                        mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) < self.thres, zeros, mask)

                    mask2d = self.generator_ddp.module.render_mask(mask, out)
                    mask2d = mask2d.reshape(-1, self.neural_rendering_resolution, self.neural_rendering_resolution, 1).permute(0,3,1,2)
                    mask2d = self.upsample(mask2d)
                    mask2d = mask2d.repeat(1,3,1,1)

                else:
                    out = self.generator_ddp.module.synthesis(ws, c, noise_mode=noise_mode)


            with torch.no_grad():
                if self.use_mask:

                    out_edit = self.generator_ddp.module.synthesis_with_mask(ws, ws_edit, c, out, mask, mix_layers=self.configs['model']['generator']['mix_layers'], noise_mode=noise_mode, ws_edit_cond=self.ws_edit_cond)


                img_origin = out['image']

                if self.use_mask:
                    img_edit = out_edit['image']


            if save_origin:
                data_name = f'{id:04}.png'
                img_origin = (img_origin.permute(0,2,3,1)*127.5+128).clamp(0,255).to(torch.uint8)
                #img_origin = (img_origin.permute(0,2,3,1)*127.5+128).clamp(0,255)
                PIL.Image.fromarray(img_origin[0].cpu().numpy(), 'RGB').save(os.path.join(save_origin_dir, data_name))

            if self.use_mask:
                data_name = f'{id:04}_edit.png'
                img_edit = (img_edit.permute(0,2,3,1)*127.5+128).clamp(0,255).to(torch.uint8)
                PIL.Image.fromarray(img_edit[0].cpu().numpy(), 'RGB').save(os.path.join(save_dir, data_name))

            data_name = f'{id:04}_nm_edit.png'
            raw_img_edit = (raw_img_edit.permute(0,2,3,1)*127.5+128).clamp(0,255).to(torch.uint8)
            PIL.Image.fromarray(raw_img_edit[0].cpu().numpy(), 'RGB').save(os.path.join(save_nm_dir, data_name))


        # end of inference code for atr, fid, is calculation

    def inference_inversion(self, name, w):

        N_images = len(w)
        #truncation_psi = 0.5
        save_dir = f'./evaluation/{self.configs["experiment"]["random_seeds"]}/{name}/edit_qual'
        os.makedirs(save_dir, exist_ok=True)
        print(f'start inference for inv // save dir: {save_dir}')

        recover_ns = self.generator_ddp.module.rendering_kwargs['depth_resolution']
        #self.generator_ddp.module.rendering_kwargs['depth_resolution'] = recover_ns*2
        recover_nsi = self.generator_ddp.module.rendering_kwargs['depth_resolution_importance']
        #self.generator_ddp.module.rendering_kwargs['depth_resolution_importance'] = recover_nsi*2

        self.generator_ddp.eval()
        self.mapper_mlp_ddp.eval()
        if self.use_deform:
            self.deform_mlp_ddp.eval()

        if self.use_mask:
            self.masknet_mlp_ddp.eval()

        if self.intrinsics is None:
            self.intrinsics = torch.tensor([
                [4.2647, 0.0, 0.5],
                [0.0, 4.2647, 0.5],
                [0.0, 0.0, 1.0]
                ], device=self.rank)

        if self.cam_pivot is None:
            self.cam_pivot = torch.tensor(self.G_kwargs['rendering_kwargs']['avg_camera_pivot'], device=self.rank)

        if self.cam_radius is None:
            self.cam_radius = self.G_kwargs['rendering_kwargs']['avg_camera_radius']

        print(self.cam_pivot, self.cam_radius)

        if self.conditioning_cam2world_pose is None:
            self.conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, self.cam_pivot, radius=self.cam_radius, device=self.rank)

        if self.conditioning_params is None:
            self.conditioning_params = torch.cat([self.conditioning_cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1,9)], 1)

        image_log = {}
        
        imgs = {}
        sum_imgs = []
        for id in tqdm(range(N_images)):
            angle_ps = [-0.2] * 5
            ys = [.3, .15, 0, -0.15, -0.3]
            for angle_i, (angle_y, angle_p) in enumerate([(ys[0], angle_ps[0]), (ys[1], angle_ps[1]), (ys[2], angle_ps[2]), (ys[3], angle_ps[3]), (ys[4], angle_ps[4])]):
                with torch.no_grad():
                    cam2world_pose = LookAtPoseSampler.sample(np.pi/2+angle_y, np.pi/2 + angle_p, self.cam_pivot, radius=self.cam_radius, device=self.rank)

                    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1,9)], 1)

                    #ws = self.generator_ddp.module.mapping(z, self.conditioning_params, truncation_psi=truncation_psi)
                    ws = w
                    ws_delta = self.mapper_mlp_ddp(ws)
                    ws_edit = ws + self.configs['lambda']['ws_delta_lambda']*ws_delta

                    raw_edit_out = self.generator_ddp.module.synthesis_get_mask_features(ws_edit, camera_params, noise_mode='const')
                    delta_raw_img = raw_edit_out['image']

                    if self.use_mask:
                        out = self.generator_ddp.module.synthesis_get_mask_features(ws, camera_params, self.deform_mlp_ddp, ws_condition=ws_edit, noise_mode='const')

                        N, HW, steps, feat_dim = out['all_feats'].shape
                        if self.configs['model']['masknet']['use_coor_input']:
                            mask_input = torch.cat([out['all_feats'].reshape(-1, feat_dim), out['all_coor'].reshape(-1, 3)], dim=-1)
                            if self.masknet_type == 'ws_condition':
                                mask = self.masknet_mlp_ddp(mask_input, ws, ws_edit)
                            elif self.masknet_type == 'attention':
                                mask_input_edit = torch.cat([raw_edit_out['all_feats'].reshape(-1, feat_dim), raw_edit_out['all_coor'].reshape(-1,3)], dim=-1)
                                mask = self.masknet_mlp_ddp(mask_input, mask_input_edit, ws, ws_edit)
                            elif self.masknet_type == 'default':
                                mask = self.masknet_mlp_ddp(mask_input)
                            else:
                                raise NotImplementedError
                        else:
                            mask_input = out['all_feats'].reshape(-1, feat_dim)
                            mask = self.masknet_mlp_ddp(mask_input)
                        mask = mask.reshape(N, HW, steps, 1)

                        if self.inv_thres is not None:
                            mask2d = self.generator_ddp.module.render_mask(mask, out)
                            ones = torch.ones_like(mask)
                            mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) > self.inv_thres, ones, mask)
                        if self.thres is not None:
                            mask2d = self.generator_ddp.module.render_mask(mask, out)
                            zeros = torch.zeros_like(mask)
                            mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) < self.thres, zeros, mask)
                        if self.mask_lpf:
                            H = W = self.neural_rendering_resolution
                            assert HW == H*W
                            mask = mask.squeeze(-1).reshape(N, H, W, steps).permute(0, 3, 1, 2)
                            # for inference, 17,15
                            mask = transforms.GaussianBlur(21, sigma=21)(mask)
                            mask = mask.reshape(N, steps, HW, 1).permute(0, 2, 1, 3)

                        mask2d = self.generator_ddp.module.render_mask(mask, out)
                        mask2d = mask2d.reshape(-1, self.neural_rendering_resolution, self.neural_rendering_resolution, 1).permute(0,3,1,2)
                        mask2d = self.upsample(mask2d)
                        mask2d = mask2d.repeat(1,3,1,1)

                    else:
                        out = self.generator_ddp.module.synthesis(ws, camera_params, noise_mode='const')


                image_relevance = self.clip_vis_ddp(out['image'], self.mask_prompt_token, self.neg_mask_prompt_token) # (N, 1, H, W)
                if self.configs['model']['masknet']['use_enlarged_mask']:
                    image_relevance_edit = self.clip_vis_ddp(delta_raw_img, self.mask_prompt_token, self.neg_mask_prompt_token) # (N, 1, H, W)
                    image_relevance = torch.max(image_relevance, image_relevance_edit)


                with torch.no_grad():

                    image_relevance = image_relevance * 2 - 1. # unnormalize
                    image_relevance = self.upsample(image_relevance)
                    image_relevance = image_relevance.repeat(1,3,1,1)


                    if self.use_mask:

                        out_edit = self.generator_ddp.module.synthesis_with_mask(ws, ws_edit, camera_params, out, mask, mix_layers=self.configs['model']['generator']['mix_layers'], noise_mode='const', ws_edit_cond=self.ws_edit_cond)


                    img_origin = out['image']

                    imgs['ori'] = img_origin
                    imgs['rel'] = image_relevance
                    imgs['edit'] = delta_raw_img

                    if self.use_mask:
                        imgs['m2d'] = mask2d

                        # override delta_raw_img
                        delta_raw_img = out_edit['image']
                        imgs['edit_raw'] = imgs['edit']
                        imgs['edit'] = delta_raw_img

                    imgs['diff'] = torch.sqrt((imgs['edit'] - imgs['ori'])**2) - 1 # -1 ~ 1
                    if 'edit_raw' in imgs:
                        imgs['diff_raw'] = torch.sqrt((imgs['edit_raw'] - imgs['ori'])**2) - 1 # -1 ~ 1

                    if id % 10 == 0:
                        sum_imgs.append(img_origin)
                        sum_imgs.append(delta_raw_img)

                for _type, data in imgs.items():
                    data_name = f'{id:04}_{angle_i}_{_type}.png'
                    data = (data.permute(0,2,3,1)*127.5+128).clamp(0,255).to(torch.uint8)
                    PIL.Image.fromarray(data[0].cpu().numpy(), 'RGB').save(os.path.join(save_dir, data_name))


        sum_imgs = torch.cat(sum_imgs, dim=0)
        sum_imgs = torch.clamp(sum_imgs, min=-1, max=1)
        sum_imgs = make_grid(sum_imgs, normalize=True, nrow=6)
        save_image(sum_imgs, os.path.join(save_dir, f"summary_image.png"))
        # end of inference code for qual results
        self.generator_ddp.module.rendering_kwargs['depth_resolution'] = recover_ns
        self.generator_ddp.module.rendering_kwargs['depth_resolution_importance'] = recover_nsi


    def inference_video(self, name, zs, gen_shape=False, cars=False):
        N_images = len(zs)
        truncation_psi = 0.5
        save_dir = f'./evaluation/{self.configs["experiment"]["random_seeds"]}/videos/{name}'
        os.makedirs(save_dir, exist_ok=True)
        print(f'start inference for qual // save dir: {save_dir}')

        recover_ns = self.generator_ddp.module.rendering_kwargs['depth_resolution']
        #self.generator_ddp.module.rendering_kwargs['depth_resolution'] = recover_ns*2
        recover_nsi = self.generator_ddp.module.rendering_kwargs['depth_resolution_importance']
        #self.generator_ddp.module.rendering_kwargs['depth_resolution_importance'] = recover_nsi*2


        self.generator_ddp.eval()
        self.mapper_mlp_ddp.eval()
        if self.use_deform:
            self.deform_mlp_ddp.eval()

        if self.use_mask:
            self.masknet_mlp_ddp.eval()

        if self.intrinsics is None:
            if cars:
                self.intrinsics = torch.tensor([
                    [1.7074, 0.0, 0.5],
                    [0.0, 1.7074, 0.5],
                    [0.0, 0.0, 1.0]
                    ], device=self.rank)
            else:
                self.intrinsics = torch.tensor([
                    [4.2647, 0.0, 0.5],
                    [0.0, 4.2647, 0.5],
                    [0.0, 0.0, 1.0]
                    ], device=self.rank)

        if self.cam_pivot is None:
            self.cam_pivot = torch.tensor(self.G_kwargs['rendering_kwargs']['avg_camera_pivot'], device=self.rank)

        if self.cam_radius is None:
            self.cam_radius = self.G_kwargs['rendering_kwargs']['avg_camera_radius']

        print(self.cam_pivot, self.cam_radius)

        if self.conditioning_cam2world_pose is None:
            self.conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, self.cam_pivot, radius=self.cam_radius, device=self.rank)

        if self.conditioning_params is None:
            self.conditioning_params = torch.cat([self.conditioning_cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1,9)], 1)
        if cars:
            self.conditioning_params = torch.zeros_like(self.conditioning_params)

        image_log = {}
        
        imgs = {}
        sum_imgs = []
        for id in tqdm(range(N_images)):
            z = zs[id:id+1]
            video_out = imageio.get_writer(os.path.join(save_dir, f'{id}.mp4'), mode='I', fps=60, codec='libx264', bitrate='10M')

            with torch.no_grad():
                ws = self.generator_ddp.module.mapping(z, self.conditioning_params, truncation_psi=truncation_psi)
                ws_delta = self.mapper_mlp_ddp(ws)
                ws_edit = ws + self.configs['lambda']['ws_delta_lambda']*ws_delta

            for frame_idx in tqdm(range(240)):
            #for angle_i, (angle_y, angle_p) in enumerate([(ys[0], angle_ps[0]), (ys[1], angle_ps[1]), (ys[2], angle_ps[2]), (ys[3], angle_ps[3]), (ys[4], angle_ps[4])]):
                pitch_range = 0.25
                yaw_range = 0.35
                with torch.no_grad():
                    #cam2world_pose = LookAtPoseSampler.sample(np.pi/2+angle_y, np.pi/2 + angle_p, self.cam_pivot, radius=self.cam_radius, device=self.rank)
                    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / 240), 3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / 240), self.cam_pivot, radius=self.cam_radius, device=self.rank)
                    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1,9)], 1)


                    raw_edit_out = self.generator_ddp.module.synthesis_get_mask_features(ws_edit, camera_params, noise_mode='const')
                    delta_raw_img = raw_edit_out['image']

                    if self.use_mask:
                        out = self.generator_ddp.module.synthesis_get_mask_features(ws, camera_params, self.deform_mlp_ddp, ws_condition=ws_edit, noise_mode='const')

                        N, HW, steps, feat_dim = out['all_feats'].shape
                        if self.configs['model']['masknet']['use_coor_input']:
                            mask_input = torch.cat([out['all_feats'].reshape(-1, feat_dim), out['all_coor'].reshape(-1, 3)], dim=-1)
                            if self.masknet_type == 'ws_condition':
                                mask = self.masknet_mlp_ddp(mask_input, ws, ws_edit)
                            elif self.masknet_type == 'attention':
                                mask_input_edit = torch.cat([raw_edit_out['all_feats'].reshape(-1, feat_dim), raw_edit_out['all_coor'].reshape(-1,3)], dim=-1)
                                mask = self.masknet_mlp_ddp(mask_input, mask_input_edit, ws, ws_edit)
                            elif self.masknet_type == 'default':
                                mask = self.masknet_mlp_ddp(mask_input)
                            else:
                                raise NotImplementedError
                        else:
                            mask_input = out['all_feats'].reshape(-1, feat_dim)
                            mask = self.masknet_mlp_ddp(mask_input)
                        mask = mask.reshape(N, HW, steps, 1)

                        if self.inv_thres is not None:
                            mask2d = self.generator_ddp.module.render_mask(mask, out)
                            ones = torch.ones_like(mask)
                            mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) > self.inv_thres, ones, mask)
                        if self.thres is not None:
                            mask2d = self.generator_ddp.module.render_mask(mask, out)
                            zeros = torch.zeros_like(mask)
                            mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) < self.thres, zeros, mask)
                        if self.mask_lpf:
                            H = W = self.neural_rendering_resolution
                            assert HW == H*W
                            mask = mask.squeeze(-1).reshape(N, H, W, steps).permute(0, 3, 1, 2)
                            # for inference, 17,15
                            mask = transforms.GaussianBlur(21, sigma=21)(mask)
                            mask = mask.reshape(N, steps, HW, 1).permute(0, 2, 1, 3)


                        mask2d = self.generator_ddp.module.render_mask(mask, out)
                        mask2d = mask2d.reshape(-1, self.neural_rendering_resolution, self.neural_rendering_resolution, 1).permute(0,3,1,2)
                        mask2d = self.upsample(mask2d)
                        mask2d = mask2d.repeat(1,3,1,1)

                    else:
                        out = self.generator_ddp.module.synthesis(ws, camera_params, noise_mode='const')


                image_relevance = self.clip_vis_ddp(out['image'], self.mask_prompt_token, self.neg_mask_prompt_token) # (N, 1, H, W)
                if self.configs['model']['masknet']['use_enlarged_mask']:
                    image_relevance_edit = self.clip_vis_ddp(delta_raw_img, self.mask_prompt_token, self.neg_mask_prompt_token) # (N, 1, H, W)
                    image_relevance = torch.max(image_relevance, image_relevance_edit)


                with torch.no_grad():

                    image_relevance = image_relevance * 2 - 1. # unnormalize
                    image_relevance = self.upsample(image_relevance)
                    image_relevance = image_relevance.repeat(1,3,1,1)


                    if self.use_mask:

                        out_edit = self.generator_ddp.module.synthesis_with_mask(ws, ws_edit, camera_params, out, mask, mix_layers=self.configs['model']['generator']['mix_layers'], noise_mode='const', ws_edit_cond=self.ws_edit_cond)


                    img_origin = out['image']

                    imgs['ori'] = img_origin
                    imgs['rel'] = image_relevance
                    imgs['edit'] = delta_raw_img

                    if self.use_mask:
                        imgs['m2d'] = mask2d

                        # override delta_raw_img
                        delta_raw_img = out_edit['image']
                        imgs['edit_raw'] = imgs['edit']
                        imgs['edit'] = delta_raw_img


                frame = torch.cat([imgs['ori'], imgs['edit']], dim=3)
                frame = (frame * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0)
                frame = frame.permute(1,2,0,).cpu().numpy()
                video_out.append_data(frame)
            video_out.close()
        # end of inference code for qual results
        self.generator_ddp.module.rendering_kwargs['depth_resolution'] = recover_ns
        self.generator_ddp.module.rendering_kwargs['depth_resolution_importance'] = recover_nsi


    # inference code for qual results
    def inference_qual(self, name, zs, gen_shape=False, cars=False):

        N_images = len(zs)
        truncation_psi = 0.5
        save_dir = f'./evaluation/{self.configs["experiment"]["random_seeds"]}/{name}/edit_qual'
        os.makedirs(save_dir, exist_ok=True)
        print(f'start inference for qual // save dir: {save_dir}')

        recover_ns = self.generator_ddp.module.rendering_kwargs['depth_resolution']
        #self.generator_ddp.module.rendering_kwargs['depth_resolution'] = recover_ns*2
        recover_nsi = self.generator_ddp.module.rendering_kwargs['depth_resolution_importance']
        #self.generator_ddp.module.rendering_kwargs['depth_resolution_importance'] = recover_nsi*2

        self.generator_ddp.eval()
        self.mapper_mlp_ddp.eval()
        if self.use_deform:
            self.deform_mlp_ddp.eval()

        if self.use_mask:
            self.masknet_mlp_ddp.eval()

        if self.intrinsics is None:
            if cars:
                self.intrinsics = torch.tensor([
                    [1.7074, 0.0, 0.5],
                    [0.0, 1.7074, 0.5],
                    [0.0, 0.0, 1.0]
                    ], device=self.rank)
            else:
                self.intrinsics = torch.tensor([
                    [4.2647, 0.0, 0.5],
                    [0.0, 4.2647, 0.5],
                    [0.0, 0.0, 1.0]
                    ], device=self.rank)

        if self.cam_pivot is None:
            self.cam_pivot = torch.tensor(self.G_kwargs['rendering_kwargs']['avg_camera_pivot'], device=self.rank)

        if self.cam_radius is None:
            self.cam_radius = self.G_kwargs['rendering_kwargs']['avg_camera_radius']

        print(self.cam_pivot, self.cam_radius)

        if self.conditioning_cam2world_pose is None:
            self.conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, self.cam_pivot, radius=self.cam_radius, device=self.rank)

        if self.conditioning_params is None:
            self.conditioning_params = torch.cat([self.conditioning_cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1,9)], 1)
        if cars:
            self.conditioning_params = torch.zeros_like(self.conditioning_params)

        image_log = {}
        
        imgs = {}
        sum_imgs = []
        for id in tqdm(range(N_images)):
            z = zs[id:id+1]
            angle_ps = [-0.2] * 5
            ys = [.3, .15, 0, -0.15, -0.3]
            if cars:
                angle_ps = [-3.14/2+0.5, 0, 3.14/2-0.5, 3.14/2*3-1, 3.14/2*3 + 1.]
                ys = [0] * 5
                ys[-1] = 3.14/2
                ys[-2] = -3.14/2
            for angle_i, (angle_y, angle_p) in enumerate([(ys[0], angle_ps[0]), (ys[1], angle_ps[1]), (ys[2], angle_ps[2]), (ys[3], angle_ps[3]), (ys[4], angle_ps[4])]):
                with torch.no_grad():
                    cam2world_pose = LookAtPoseSampler.sample(np.pi/2+angle_y, np.pi/2 + angle_p, self.cam_pivot, radius=self.cam_radius, device=self.rank)

                    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1,9)], 1)

                    ws = self.generator_ddp.module.mapping(z, self.conditioning_params, truncation_psi=truncation_psi)
                    ws_delta = self.mapper_mlp_ddp(ws)
                    ws_edit = ws + self.configs['lambda']['ws_delta_lambda']*ws_delta

                    raw_edit_out = self.generator_ddp.module.synthesis_get_mask_features(ws_edit, camera_params, noise_mode='const')
                    delta_raw_img = raw_edit_out['image']

                    if self.use_mask:
                        out = self.generator_ddp.module.synthesis_get_mask_features(ws, camera_params, self.deform_mlp_ddp, ws_condition=ws_edit, noise_mode='const')

                        N, HW, steps, feat_dim = out['all_feats'].shape
                        if self.configs['model']['masknet']['use_coor_input']:
                            mask_input = torch.cat([out['all_feats'].reshape(-1, feat_dim), out['all_coor'].reshape(-1, 3)], dim=-1)
                            if self.masknet_type == 'ws_condition':
                                mask = self.masknet_mlp_ddp(mask_input, ws, ws_edit)
                            elif self.masknet_type == 'attention':
                                mask_input_edit = torch.cat([raw_edit_out['all_feats'].reshape(-1, feat_dim), raw_edit_out['all_coor'].reshape(-1,3)], dim=-1)
                                mask = self.masknet_mlp_ddp(mask_input, mask_input_edit, ws, ws_edit)
                            elif self.masknet_type == 'default':
                                mask = self.masknet_mlp_ddp(mask_input)
                            else:
                                raise NotImplementedError
                        else:
                            mask_input = out['all_feats'].reshape(-1, feat_dim)
                            mask = self.masknet_mlp_ddp(mask_input)
                        mask = mask.reshape(N, HW, steps, 1)

                        if self.inv_thres is not None:
                            mask2d = self.generator_ddp.module.render_mask(mask, out)
                            ones = torch.ones_like(mask)
                            mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) > self.inv_thres, ones, mask)
                        if self.thres is not None:
                            mask2d = self.generator_ddp.module.render_mask(mask, out)
                            zeros = torch.zeros_like(mask)
                            mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) < self.thres, zeros, mask)
                        if self.mask_lpf:
                            H = W = self.neural_rendering_resolution
                            assert HW == H*W
                            mask = mask.squeeze(-1).reshape(N, H, W, steps).permute(0, 3, 1, 2)


                            # for inference, 17,15
                            mask = transforms.GaussianBlur(17, sigma=15)(mask)
                            mask = mask.reshape(N, steps, HW, 1).permute(0, 2, 1, 3)


                        mask2d = self.generator_ddp.module.render_mask(mask, out)
                        mask2d = mask2d.reshape(-1, self.neural_rendering_resolution, self.neural_rendering_resolution, 1).permute(0,3,1,2)
                        mask2d = self.upsample(mask2d)
                        mask2d = mask2d.repeat(1,3,1,1)

                    else:
                        out = self.generator_ddp.module.synthesis(ws, camera_params, noise_mode='const')


                image_relevance = self.clip_vis_ddp(out['image'], self.mask_prompt_token, self.neg_mask_prompt_token) # (N, 1, H, W)
                if self.configs['model']['masknet']['use_enlarged_mask']:
                    image_relevance_edit = self.clip_vis_ddp(delta_raw_img, self.mask_prompt_token, self.neg_mask_prompt_token) # (N, 1, H, W)
                    image_relevance = torch.max(image_relevance, image_relevance_edit)


                with torch.no_grad():

                    image_relevance = image_relevance * 2 - 1. # unnormalize
                    image_relevance = self.upsample(image_relevance)
                    image_relevance = image_relevance.repeat(1,3,1,1)


                    if self.use_mask:

                        out_edit = self.generator_ddp.module.synthesis_with_mask(ws, ws_edit, camera_params, out, mask, mix_layers=self.configs['model']['generator']['mix_layers'], noise_mode='const', ws_edit_cond=self.ws_edit_cond)


                    img_origin = out['image']

                    imgs['ori'] = img_origin
                    imgs['rel'] = image_relevance
                    imgs['edit'] = delta_raw_img

                    if self.use_mask:
                        imgs['m2d'] = mask2d

                        # override delta_raw_img
                        delta_raw_img = out_edit['image']
                        imgs['edit_raw'] = imgs['edit']
                        imgs['edit'] = delta_raw_img

                    imgs['diff'] = torch.sqrt((imgs['edit'] - imgs['ori'])**2) - 1 # -1 ~ 1
                    if 'edit_raw' in imgs:
                        imgs['diff_raw'] = torch.sqrt((imgs['edit_raw'] - imgs['ori'])**2) - 1 # -1 ~ 1

                    if id % 10 == 0:
                        sum_imgs.append(img_origin)
                        sum_imgs.append(delta_raw_img)

                for _type, data in imgs.items():
                    data_name = f'{id:04}_{angle_i}_{_type}.png'
                    data = (data.permute(0,2,3,1)*127.5+128).clamp(0,255).to(torch.uint8)
                    PIL.Image.fromarray(data[0].cpu().numpy(), 'RGB').save(os.path.join(save_dir, data_name))

            if gen_shape:
                # gen shape
                # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
                max_batch=1000000
                shape_res = 512

                samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=self.G_kwargs['rendering_kwargs']['box_warp'] * 1)#.reshape(1, -1, 3)
                samples = samples.to(z.device)
                sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
                masks = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
                transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
                transformed_ray_directions_expanded[..., -1] = -1

                head = 0
                with tqdm(total = samples.shape[1]) as pbar:
                    with torch.no_grad():
                        while head < samples.shape[1]:
                            out = self.generator_ddp.module.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, self.conditioning_params, truncation_psi=truncation_psi, noise_mode='const', return_feats=True)
                            sigma = out['sigma']
                            feats = torch.cat(out['feats_list'], dim=-1)
                            sigmas[:, head:head+max_batch] = sigma
                            ###

                            N, num_points, feat_dim = feats.shape
                            if self.configs['model']['masknet']['use_coor_input']:
                                mask_input = torch.cat([feats.reshape(-1, feat_dim), samples[:,head:head+max_batch].reshape(-1, 3)], dim=-1)
                                if self.masknet_type == 'ws_condition':
                                    mask = self.masknet_mlp_ddp(mask_input, ws, ws_edit)
                                elif self.masknet_type == 'attention':
                                    raw_edit_out = self.generator_ddp.module.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, self.conditioning_params, ws=ws_edit, truncation_psi=truncation_psi, noise_mode='const', return_feats=True)
                                    raw_feats = torch.cat(out['feats_list'], dim=-1)
                                    mask_input_edit = torch.cat([raw_feats.reshape(-1, feat_dim), samples[:,head:head+max_batch].reshape(-1,3)], dim=-1)
                                    mask = self.masknet_mlp_ddp(mask_input, mask_input_edit, ws, ws_edit)
                                elif self.masknet_type == 'default':
                                    mask = self.masknet_mlp_ddp(mask_input)
                                else:
                                    raise NotImplementedError
                            else:
                                mask_input = feats.reshape(-1, feat_dim)
                                mask = self.masknet_mlp_ddp(mask_input)
                            '''
                            mask = mask.reshape(N, HW, steps, 1)

                            if self.mask_lpf:
                                H = W = self.neural_rendering_resolution
                                assert HW == H*W
                                mask = mask.squeeze(-1).reshape(N, H, W, steps).permute(0, 3, 1, 2)
                                # for inference, 17,15
                                mask = transforms.GaussianBlur(61, sigma=130)(mask)
                                mask = mask.reshape(N, steps, HW, 1).permute(0, 2, 1, 3)
                            if self.inv_thres is not None:
                                mask2d = self.generator_ddp.module.render_mask(mask, out)
                                ones = torch.ones_like(mask)
                                mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) > self.inv_thres, ones, mask)
                            if self.thres is not None:
                                mask2d = self.generator_ddp.module.render_mask(mask, out)
                                zeros = torch.zeros_like(mask)
                                mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) < self.thres, zeros, mask)
                            ###
                            '''
                            masks[:, head:head+max_batch] = mask

                            head += max_batch
                            pbar.update(max_batch)

                sig_masks = sigmas * masks

                sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
                sigmas = np.flip(sigmas, 0)

                masks = masks.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
                masks = np.flip(masks, 0)
                sig_masks = sig_masks.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
                sig_masks = np.flip(sig_masks, 0)

                # Trim the border of the extracted cube
                pad = int(30 * shape_res / 256)
                pad_value = -1000
                sigmas[:pad] = pad_value
                sigmas[-pad:] = pad_value
                sigmas[:, :pad] = pad_value
                sigmas[:, -pad:] = pad_value
                sigmas[:, :, :pad] = pad_value
                sigmas[:, :, -pad:] = pad_value

                masks[:pad] = pad_value
                masks[-pad:] = pad_value
                masks[:, :pad] = pad_value
                masks[:, -pad:] = pad_value
                masks[:, :, :pad] = pad_value
                masks[:, :, -pad:] = pad_value

                sig_masks[:pad] = pad_value
                sig_masks[-pad:] = pad_value
                sig_masks[:, :pad] = pad_value
                sig_masks[:, -pad:] = pad_value
                sig_masks[:, :, :pad] = pad_value
                sig_masks[:, :, -pad:] = pad_value

                mrc_name = f'{id:04}'
                mrc_dir = os.path.join(save_dir, 'mrcs')
                os.makedirs(mrc_dir, exist_ok=True)
                with mrcfile.new_mmap(os.path.join(mrc_dir, f'{mrc_name}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                    mrc.data[:] = sigmas
                with mrcfile.new_mmap(os.path.join(mrc_dir, f'{mrc_name}_mask.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                    mrc.data[:] = masks
                with mrcfile.new_mmap(os.path.join(mrc_dir, f'{mrc_name}_sigmasks.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                    mrc.data[:] = sig_masks

        sum_imgs = torch.cat(sum_imgs, dim=0)
        sum_imgs = torch.clamp(sum_imgs, min=-1, max=1)
        sum_imgs = make_grid(sum_imgs, normalize=True, nrow=6)
        save_image(sum_imgs, os.path.join(save_dir, f"summary_image.png"))
        # end of inference code for qual results
        self.generator_ddp.module.rendering_kwargs['depth_resolution'] = recover_ns
        self.generator_ddp.module.rendering_kwargs['depth_resolution_importance'] = recover_nsi


    # mapper, masknet, deformation field
    def load_checkpoints(self, config_path, step):
        _dirname = os.path.dirname(config_path)

        ckpt_mapper = os.path.join(_dirname, f'{step}-mapper.pth')
        self.mapper_mlp_ddp.module.load_state_dict(torch.load(ckpt_mapper))
        print(f'loaded from {ckpt_mapper}')

        if self.use_mask:
            ckpt_mask = os.path.join(_dirname, f'{step}-masknet.pth')
            self.masknet_mlp_ddp.module.load_state_dict(torch.load(ckpt_mask))
            print(f'loaded from {ckpt_mask}')

        if self.use_deform:
            ckpt_deform = os.path.join(_dirname, f'{step}-deform.pth')
            self.deform_mlp_ddp.module.load_state_dict(torch.load(ckpt_deform))
            print(f'loaded from {ckpt_deform}')



    def train_masknet(self, gen_c, c_gen_conditioning):
        z = torch.randn([self.batch_gpu, self.generator.z_dim]).cuda()

        all_logs = {}
        self.optimizer_masknet.zero_grad()
        loss = 0



        # gradient accumulation with same identity
        for msplit in range(self.multiview_num):
            _gc = gen_c[msplit*self.batch_gpu:(msplit+1)*self.batch_gpu]
            _cgc = c_gen_conditioning[msplit*self.batch_gpu:(msplit+1)*self.batch_gpu]

            with torch.no_grad():
                ws = self.generator_ddp.module.mapping(z, _cgc)

                ws_delta = self.mapper_mlp_ddp(ws)
                ws_edit = ws + self.configs['lambda']['ws_delta_lambda']*ws_delta
                # raw edit
                raw_edit_out = self.generator_ddp.module.synthesis_get_mask_features(ws_edit, _gc)

                out = self.generator_ddp.module.synthesis_get_mask_features(ws, _gc, self.deform_mlp_ddp, ws_condition=ws_edit)


            N, HW, steps, feat_dim = out['all_feats'].shape
            if self.configs['model']['masknet']['use_coor_input']:
                mask_input = torch.cat([out['all_feats'].reshape(-1, feat_dim), out['all_coor'].reshape(-1, 3)], dim=-1)
                if self.masknet_type == 'ws_condition':
                    mask = self.masknet_mlp_ddp(mask_input, ws, ws_edit)
                elif self.masknet_type == 'attention':
                    mask_input_edit = torch.cat([raw_edit_out['all_feats'].reshape(-1, feat_dim), raw_edit_out['all_coor'].reshape(-1,3)], dim=-1)
                    mask = self.masknet_mlp_ddp(mask_input, mask_input_edit, ws, ws_edit)
                elif self.masknet_type == 'default':
                    mask = self.masknet_mlp_ddp(mask_input)
                else:
                    raise NotImplementedError
            else:
                mask_input = out['all_feats'].reshape(-1, feat_dim)
                mask = self.masknet_mlp_ddp(mask_input)
            mask = mask.reshape(N, HW, steps, 1)

            if self.mask_lpf:
                H = W = self.neural_rendering_resolution
                assert HW == H*W
                mask = mask.squeeze(-1).reshape(N, H, W, steps).permute(0, 3, 1, 2)
                mask = transforms.GaussianBlur(17, sigma=15)(mask)
                mask = mask.reshape(N, steps, HW, 1).permute(0, 2, 1, 3)
            if self.inv_thres is not None:
                mask2d = self.generator_ddp.module.render_mask(mask, out)
                ### testing
                zeros = torch.zeros_like(mask2d)
                ### testing

                ones = torch.ones_like(mask)
                mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) > self.inv_thres, ones, mask)
            if self.thres is not None:
                mask2d = self.generator_ddp.module.render_mask(mask, out)
                ### testing
                zeros = torch.zeros_like(mask2d)
                ### testing
                zeros = torch.zeros_like(mask)
                mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) < self.thres, zeros, mask)
            #print('mask::::::::::')
            #print(float(mask.max().detach().cpu()), float(mask.min().detach().cpu()), float(out['all_densities'].max().detach().cpu()), float(out['all_densities'].min().detach().cpu()), float(out['all_depths'].max().detach().cpu()), float(out['all_depths'].min().detach().cpu()))
            mask2d =  self.generator_ddp.module.render_mask(mask, out)
            mask2d = mask2d.reshape(-1, self.neural_rendering_resolution, self.neural_rendering_resolution, 1).permute(0,3,1,2)

            image_relevance = self.clip_vis_ddp(out['image'], self.mask_prompt_token, self.neg_mask_prompt_token) # (N, 1, H, W)
            if self.configs['model']['masknet']['use_enlarged_mask']:
                image_relevance_edit = self.clip_vis_ddp(raw_edit_out['image'], self.mask_prompt_token, self.neg_mask_prompt_token) # (N, 1, H, W)
                image_relevance = torch.max(image_relevance, image_relevance_edit)


            dict_for_penalty = {'mask2d': mask2d, 'image_relevance': image_relevance}

            if self.configs['lambda']['masknet_cliploss_lambda'] > 0:
                out_edit = self.generator_ddp.module.synthesis_with_mask(ws, ws_edit, _gc, out, mask, mix_layers=self.configs['model']['generator']['mix_layers'], texture_only=self.configs['model']['generator']['texture_only'], ws_edit_cond=self.ws_edit_cond)
                dict_for_penalty['image'] = out_edit['image']
                dict_for_penalty['image_original'] = out['image']

            all_logs, _loss = self.get_masknet_penalty(all_logs, dict_for_penalty)
            loss += _loss.item()
            _loss.backward()


        self.optimizer_masknet.step()

        return loss, all_logs


    def get_masknet_penalty(self, all_logs, dict_for_penalty):
        loss = 0

        # masknet loss
        mask2d = dict_for_penalty['mask2d'] # mask2d is in range -1 ~ 1
        image_relevance = dict_for_penalty['image_relevance'] # image_relevance is in range 0 ~ 1
        image_relevance = image_relevance * 2 -1. # now -1 ~ 1
        masknet_loss = self.configs['lambda']['mask_lambda']*self.mseloss(mask2d, image_relevance) / self.multiview_num

        if 'masknet_loss' in all_logs:
            all_logs['masknet_loss'] += masknet_loss.item()
        else:
            all_logs['masknet_loss'] = masknet_loss.item()

        loss += masknet_loss

        # CLIP Loss
        if self.configs['lambda']['masknet_cliploss_lambda'] > 0:
            img = dict_for_penalty['image']
            if self.configs['training']['loss_type'] == 'dir_nce':
                img_ori = dict_for_penalty['image_original']
                cliploss = self.configs['lambda']['masknet_cliploss_lambda'] * self.clip_loss_ddp.module.forward_diff(img, img_ori, self.edit_prompt_token, self.src_prompt_token) / self.multiview_num
            elif self.configs['training']['loss_type'] == 'dir_diff':
                img_ori = dict_for_penalty['image_original']
                cliploss = self.configs['lambda']['masknet_cliploss_lambda'] * self.clip_loss_ddp.module.forward_diff_only(img, img_ori, self.edit_prompt_token, self.src_prompt_token) / self.multiview_num
            elif self.configs['training']['loss_type'] == 'default':
                cliploss = self.configs['lambda']['masknet_cliploss_lambda'] * self.clip_loss_ddp(img, self.edit_prompt_token) / self.multiview_num
            else:
                raise NotImplementedError

            if 'masknet_cliploss' in all_logs:
                all_logs['masknet_cliploss'] += cliploss.item()
            else:
                all_logs['masknet_cliploss'] = cliploss.item()

            loss += cliploss

        if self.configs['lambda']['mask2dreg'] > 0:
            mask2dreg = self.configs['lambda']['mask2dreg']*(1 -mask2d**2).mean() / self.multiview_num

            if 'mask2dreg' in all_logs:
                all_logs['mask2dreg'] += mask2dreg.item()
            else:
                all_logs['mask2dreg'] = mask2dreg.item()

            loss += mask2dreg

        assert 'mask2d_topk' in self.configs['lambda']
        assert 'mask2d_botk' in self.configs['lambda']

        if self.configs['lambda']['mask2d_minmax_reg'] > 0:
            bs = mask2d.shape[0]
            mask2d_flat = mask2d.view(bs, -1)
            hw = mask2d_flat.shape[-1]
            mask2d_flat = mask2d_flat[:, torch.randperm(hw)[:5000]]
            tkn = self.configs['lambda']['mask2d_topk']
            bkn = self.configs['lambda']['mask2d_botk']
            topks = torch.topk(mask2d_flat, tkn, dim=-1)[0]
            botks = torch.topk(mask2d_flat, bkn, dim=-1, largest=False)[0]

            mask2d_minmax_reg = self.configs['lambda']['mask2d_minmax_reg'] * (((1.2 - topks)**2).mean() * tkn + ((1 + botks)**2).mean() * bkn) / ((tkn+bkn) * self.multiview_num)

            if 'mask2d_minmax_reg' in all_logs:
                all_logs['mask2d_minmax_reg'] += mask2d_minmax_reg.item()
            else:
                all_logs['mask2d_minmax_reg'] = mask2d_minmax_reg.item()

            loss += mask2d_minmax_reg

        if self.configs['lambda']['mask2d_tv_reg'] > 0:
            h_diff = ((torch.roll(mask2d, 1, dims=2)[:,:,1:] - mask2d[:,:,1:])**2).mean()
            w_diff = ((torch.roll(mask2d, 1, dims=3)[:,:,:,1:] - mask2d[:,:,:,1:])**2).mean()
            mask2d_tv_reg = self.configs['lambda']['mask2d_tv_reg'] * (h_diff + w_diff) / self.multiview_num

            if 'mask2d_tv_reg' in all_logs:
                all_logs['mask2d_tv_reg'] += mask2d_tv_reg.item()
            else:
                all_logs['mask2d_tv_reg'] = mask2d_tv_reg.item()

            loss += mask2d_tv_reg

        return all_logs, loss



    def train_mapper(self, gen_c, c_gen_conditioning):
        z = torch.randn([self.batch_gpu, self.generator.z_dim]).cuda()

        all_logs = {}
        self.optimizer_mapper.zero_grad()

        loss = 0

        # gradient accumulation with same identity
        for msplit in range(self.multiview_num):
            _gc = gen_c[msplit*self.batch_gpu:(msplit+1)*self.batch_gpu]
            _cgc = c_gen_conditioning[msplit*self.batch_gpu:(msplit+1)*self.batch_gpu]

            with torch.no_grad():
                ws = self.generator_ddp.module.mapping(z, _cgc)

            ws_delta = self.mapper_mlp_ddp(ws)
            ws_edit = ws + self.configs['lambda']['ws_delta_lambda']*ws_delta

            dict_for_penalty = {}

            if self.use_mask:
                with torch.no_grad():
                    out = self.generator_ddp.module.synthesis_get_mask_features(ws, _gc, self.deform_mlp_ddp, ws_condition=ws_edit)
                    raw_edit_out = self.generator_ddp.module.synthesis_get_mask_features(ws_edit, _gc)

                    N, HW, steps, feat_dim = out['all_feats'].shape
                    if self.configs['model']['masknet']['use_coor_input']:
                        mask_input = torch.cat([out['all_feats'].reshape(-1, feat_dim), out['all_coor'].reshape(-1, 3)], dim=-1)
                        if self.masknet_type == 'ws_condition':
                            mask = self.masknet_mlp_ddp(mask_input, ws, ws_edit)
                        elif self.masknet_type == 'attention':
                            mask_input_edit = torch.cat([raw_edit_out['all_feats'].reshape(-1, feat_dim), raw_edit_out['all_coor'].reshape(-1,3)], dim=-1)
                            mask = self.masknet_mlp_ddp(mask_input, mask_input_edit, ws, ws_edit)
                        elif self.masknet_type == 'default':
                            mask = self.masknet_mlp_ddp(mask_input)
                        else:
                            raise NotImplementedError
                    else:
                        mask_input = out['all_feats'].reshape(-1, feat_dim)
                        mask = self.masknet_mlp_ddp(mask_input)
                    mask = mask.reshape(N, HW, steps, 1)

                    if self.mask_lpf:
                        H = W = self.neural_rendering_resolution
                        assert HW == H*W
                        mask = mask.squeeze(-1).reshape(N, H, W, steps).permute(0, 3, 1, 2)
                        mask = transforms.GaussianBlur(17, sigma=15)(mask)
                        mask = mask.reshape(N, steps, HW, 1).permute(0, 2, 1, 3)
                    if self.inv_thres is not None:
                        mask2d = self.generator_ddp.module.render_mask(mask, out)
                        ones = torch.ones_like(mask)
                        mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) > self.inv_thres, ones, mask)

                        mask2d = self.generator_ddp.module.render_mask(mask, out)
                    if self.thres is not None:
                        mask2d = self.generator_ddp.module.render_mask(mask, out)
                        zeros = torch.zeros_like(mask)
                        mask = torch.where(mask2d.unsqueeze(2).expand_as(mask) < self.thres, zeros, mask)
                        mask2d = self.generator_ddp.module.render_mask(mask, out)

                out_edit = self.generator_ddp.module.synthesis_with_mask(ws, ws_edit, _gc, out, mask, mix_layers=self.configs['model']['generator']['mix_layers'], texture_only=self.configs['model']['generator']['texture_only'], ws_edit_cond=self.ws_edit_cond)


            # no mask
            else:
                with torch.no_grad():
                    out = self.generator_ddp.module.synthesis(ws, _gc)
                # raw edit 
                raw_edit_out = out_edit = self.generator_ddp.module.synthesis(ws_edit, _gc)

            dict_for_penalty.update(out_edit) #(image, image_raw, image_depth, mask2d) # (N, C, H, W)
            dict_for_penalty['ws_delta'] = ws_delta
            
            # Log original image for ArcFace loss
            dict_for_penalty['image_original'] = out['image'].detach()

            dict_for_penalty['raw_edit'] = raw_edit_out['image']

            if self.use_deform:
                dict_for_penalty['all_coor_deform'] = out['all_coor_deform']
                dict_for_penalty['all_coor'] = out['all_coor']
                dict_for_penalty['mask'] = mask.detach()

            all_logs, _loss = self.get_mapper_penalty(all_logs, dict_for_penalty)

            loss += _loss.item()
            _loss.backward()


        self.optimizer_mapper.step()

        return loss, all_logs

    def get_mapper_penalty(self, all_logs, dict_for_penalty):
        loss = 0

        # CLIP Loss
        img = dict_for_penalty['image']
        img_ori = dict_for_penalty['image_original']

        if self.use_mask and self.use_masked_cliploss:
            mask2d = dict_for_penalty['mask2d']
            mask2d = self.upsample(mask2d)
            mask2d = mask2d.repeat(1,3,1,1)
            img = torch.where(mask2d > 0., img, -1*torch.ones_like(img))
            img_ori = torch.where(mask2d > 0., img_ori, -1*torch.ones_like(img_ori))

        if self.configs['training']['loss_type'] == 'dir_nce':
            cliploss = self.configs['lambda']['cliploss_lambda'] * self.clip_loss_ddp.module.forward_diff(img, img_ori, self.edit_prompt_token, self.src_prompt_token) / self.multiview_num
        elif self.configs['training']['loss_type'] == 'dir_diff':
            img_ori = dict_for_penalty['image_original']
            cliploss = self.configs['lambda']['cliploss_lambda'] * self.clip_loss_ddp.module.forward_diff_only(img, img_ori, self.edit_prompt_token, self.src_prompt_token) / self.multiview_num
        elif self.configs['training']['loss_type'] == 'default':
            cliploss = self.configs['lambda']['cliploss_lambda'] * self.clip_loss_ddp(img, self.edit_prompt_token) / self.multiview_num
        else:
            raise NotImplementedError


        if 'cliploss' in all_logs:
            all_logs['cliploss'] += cliploss.item()
        else:
            all_logs['cliploss'] = cliploss.item()

        loss += cliploss

        # ws_Delta reg. loss
        ws_delta = dict_for_penalty['ws_delta']
        edit_latent_reg = self.configs['lambda']['ws_reg_lambda'] * torch.norm(ws_delta, dim=-1).mean() / self.multiview_num

        if 'edit_latent_reg' in all_logs:
            all_logs['edit_latent_reg'] += edit_latent_reg.item()
        else:
            all_logs['edit_latent_reg'] = edit_latent_reg.item()

        loss += edit_latent_reg

        # ArcFace Loss
        af_loss = self.configs['lambda']['arcface_lambda'] * self.arcface_loss_ddp(img, dict_for_penalty['image_original']) / self.multiview_num
        if 'arcface_loss' in all_logs:
            all_logs['arcface_loss'] += af_loss.item()
        else:
            all_logs['arcface_loss'] = af_loss.item()
        loss += af_loss

        af_raw_edit_loss = self.configs['lambda']['delta_arcface_lambda'] * self.arcface_loss(img.detach(), dict_for_penalty['raw_edit']) / self.multiview_num
        if 'arcface_raw_edit_loss' in all_logs:
            all_logs['arcface_raw_edit_loss'] += af_raw_edit_loss.item()
        else:
            all_logs['arcface_raw_edit_loss'] = af_raw_edit_loss.item()
        loss += af_raw_edit_loss

        if self.use_deform:
            all_coor = dict_for_penalty['all_coor']
            all_coor_deform = dict_for_penalty['all_coor_deform']
            diff=torch.mean((all_coor - all_coor_deform)**2, dim=-1).reshape(all_coor.shape[0], -1) # (N, HW*steps)
            mask = dict_for_penalty['mask'].reshape(all_coor.shape[0], -1)
            soft_mask = torch.nn.functional.softmax(mask, dim=-1) # (N, HW*steps)
            deform_reg = self.configs['lambda']['deform_reg_lambda']*torch.mean(diff*soft_mask) / self.multiview_num

            if 'deform_reg' in all_logs:
                all_logs['deform_reg'] += deform_reg.item()
            else:
                all_logs['deform_reg'] = deform_reg.item()
            loss += deform_reg

        return all_logs, loss


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size
