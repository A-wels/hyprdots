import random
import numpy as np
import sys
import os
import torch.nn.functional as F
import torch
import argparse
import h5py
import time
from PIL import Image
def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def readFlow3d(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def convert_mvf_to_flo(mvf, mvf_name) -> str:
    mvf = np.reshape(mvf, [*[344,127],2], order='F')
    flo_name = mvf_name.replace('.mvf', '.flo')
    writeFlow(flo_name, mvf)
    return flo_name

def convert_3dmvf_to_flo(mvf, device="cpu", reduce_size=False):
    mvf = torch.from_numpy(np.reshape(mvf, [*[344,344,127],3], order='F'))
    # remove every other element in all dimensions
    if reduce_size:
      #  mvf = mvf[::2,::2,::2,:]
        mvf = mvf[20:-20,40:-40, :]
    return mvf
def read_gen(file_name, pil=False, reduce_size=False, cpu=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    elif ext == '.v':
        v = np.fromfile(file_name, 'float32')
        is_3d = len(v) == 344*344*127
        if is_3d:
            # create image from generated data
            if cpu:
                data =  torch.from_numpy(np.reshape(v,[344,344,127], order='F')).float()
            else:
                data =  torch.from_numpy(np.reshape(v,[344,344,127], order='F')).float()
            # normalize data to [0, 1]
            data = ((data - data.min()) * (1/(data.max() - data.min())))
            # remove every other row, column, and depth to reduce size
            if reduce_size:
             #   data = data[::2, ::2, ::2]       
                data = data[20:-20,40:-40, :]

            return data
        else:
            # create image from generated data
            data =  torch.from_numpy(np.reshape(np.fromfile(file_name, 'float32'),[344,127], order='F'))
            # scale sum of data to 6
            sum = torch.sum(data).to(device)
            target = torch.tensor(6.0).to(device)
            data = data * (target/sum)

            # scale to [0,1]
            data = ((data - data.min()) * (1/(data.max() - data.min())))

            return data
    elif ext == '.mvf':
        mvf = np.fromfile(file_name, dtype=np.float32)
        is_3d = len(mvf) == 344*344*127*3
        if is_3d:
           return convert_3dmvf_to_flo(mvf, device, reduce_size)
        else:
            flo_name = file_name.replace('.mvf', '.flo')
            if os.path.exists(flo_name):
                flow = readFlow(flo_name).astype(np.float32)
                return flow
            else:
                 return read_gen(convert_mvf_to_flo(mvf, file_name))
    return []

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def writeFlow3D(filename, uvw,shape, v=None):
    # if uvw is tensor, convert to numpy
    if isinstance(uvw, torch.Tensor):
        uvw = uvw.cpu().numpy()
        
    # write to file
    uvw = uvw.reshape(shape, order='A')
    uvw.tofile(filename)
    
    


def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("model_name", help="Name of model.", type=str)

    # Optional arguments
    
    # How often to display the losses
    parser.add_argument("-v", "--verbose_iters", 
                        help="Number of batch  iters after which to evaluate val set and display output.", 
                        type=int, default=10000)
    
    # How often to display save the model
    parser.add_argument("-ct", "--cp_time", 
                        help="Number of minutes after which to save a checkpoint.", 
                        type=float, default=15)
    # Alternate data directory than cycgan/data/
    parser.add_argument("-dd", "--data_dir", 
                        help="Different data directory from ml/data.", 
                        type=str, default=None)
    
    # Parse arguments
    args = parser.parse_args()
    
    return args

class XCAT3DDataset(torch.utils.data.Dataset):
    
    MAX_IMAGES = 10
    def __init__(self, split='training', root='/scratch/tmp/a_wels03/dataset_002_3d'):
        self.image_list = []
        self.flow_list = []
        error_list = []
        # iterate over all files in root, sorted by name
        files = sorted(os.listdir(root))
        img_list = [f for f in files if f.endswith(".v")]
        flow_list = [os.path.join(root,f) for f in files if f.endswith(".mvf")]

        scenes = []
        last_scence = ""
        for f in flow_list:
            scene_id = f.split("/")[-1].split("_")[0]
            if scene_id != last_scence:
                scenes.append([scene_id])
            scenes[-1].append(f)
            last_scence = scene_id
        for scene in scenes:
            if len(scene) != 8:
                error_list.append(scene[0])

        # iterate over all images, add them to self.image_list fixed if their scene id is not in error_list. Additionally, add the corresponding flow file to self.flow_list
        for f in img_list:
            scene_id = f.split("/")[-1].split("_")[0]
            if scene_id not in error_list:
                end_idx = f[-3]
                if end_idx != "1":
                    self.image_list.append([ os.path.join(root,f[:-3]+"1.v"), os.path.join(root,f),])
                    flow_path = re.sub(r'_noise_\d+', '', f)
                    flow_path = flow_path.replace(".v", ".mvf")
                    self.flow_list.append(os.path.join(root,flow_path))

                    
        # first 90% as training data
        amount_of_scenes = len(scenes)
        index_split = int(0.9*amount_of_scenes)
        if split=="training":
            for f in self.flow_list.copy():
                scene_id = int(f.split("/")[-1].split("_")[0])
                if scene_id > index_split:
                    self.flow_list.remove(f)
            for f in self.image_list.copy():
                scene_id = int(f[0].split("/")[-1].split("_")[0])
                if scene_id > index_split:
                    self.image_list.remove(f)
        else:
            for f in self.flow_list.copy():
                scene_id = int(f.split("/")[-1].split("_")[0])
                if scene_id <= index_split:
                    self.flow_list.remove(f)
            for f in self.image_list.copy():
                scene_id = int(f[0].split("/")[-1].split("_")[0])
                if scene_id <= index_split:
                    self.image_list.remove(f)

        #self.image_list = self.image_list[0:self.MAX_IMAGES]
        #self.flow_list = self.flow_list[0:self.MAX_IMAGES]


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
            index = index % len(self.image_list)
            valid = None
           
            flow = read_gen(self.flow_list[index], reduce_size=True)
            img1 = read_gen(self.image_list[index][0], reduce_size=True)
            img2 = read_gen(self.image_list[index][1], reduce_size=True)
            img1 = img1.permute((2, 0, 1))
            img2 = img2.permute((2, 0, 1))
            flow = flow.permute((3, 2, 0, 1))

            # add noise to images
            max_noise = random.randint(0,20)/100.0
            img1 = (img1 + torch.from_numpy(np.random.normal(0, max_noise, img1.shape))).float()
            img2 = (img2 + torch.from_numpy(np.random.normal(0, max_noise, img2.shape))).float()
            # reset to 1 if value is >1
            img1[img1 > 1] = 1
            img2[img2 > 1] = 1
            # reset to 0 if value is <0
            img1[img1 < 0] = 0
            img2[img2 < 0] = 0
            

            # scale to [0,1]
            #img1 = ((img1 - img1.min()) * (1/(img1.max() - img1.min())))
            #img2 = ((img2 - img2.min()) * (1/(img2.max() - img2.min())))
            

            if valid is not None:
                valid = torch.from_numpy(valid)
            else:
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
            input_images = torch.stack([img1,img2], dim=0)
            return input_images, flow, valid.float()


def create_normalize(x_mean, x_std):
    
    def normalize(img):
        return (img - x_mean) / x_std
    
    return normalize
    
def charbonnier(x, alpha=0.25, epsilon=1.0e-9):
    return torch.pow(torch.pow(x, 2) + epsilon**2, alpha)

def smoothness_loss(flow):
    b, c, d, h, w = flow.size()
    # Compute the charbonnier loss between subsequent flow values
    v_translated = torch.cat((flow[:, :, 1:], torch.zeros(b, c, 1, h, w, device=flow.device)), dim=-3)
    h_translated = torch.cat((flow[:, :, :, 1:], torch.zeros(b, c, d, 1, w, device=flow.device)), dim=-2)
    u_translated = torch.cat((flow[:, :, :, :, 1:], torch.zeros(b, c, d, h, 1, device=flow.device)), dim=-1)
    s_loss = charbonnier(flow - v_translated) + charbonnier(flow - h_translated) + charbonnier(flow - u_translated)
    return torch.mean(s_loss)

def photometric_loss(warped, target_img, alpha):
    d, h, w = warped.shape[2:]
    target_img = F.interpolate(target_img, (d, h, w), mode='trilinear', align_corners=False)
    p_loss = charbonnier(warped - target_img, alpha)
    return torch.mean(p_loss)

def inv_loss(flow1, flow2, grid0):
    '''Invertible loss.'''
    
    b, c, d, h, w = flow1.size()
    

    grid0 = grid0.unsqueeze(0)
    grid0 = grid0.repeat(b, 1, 1, 1, 1)
    
    # Adjust original grid with flow in one direction
    grid1 = grid0 + flow1.permute(0,2,3,4,1)

    # Adjust original grid with flow in the other direction
    grid2 = grid0 + flow2.permute(0,2,3,4,1)

    # If flow1 is invertible and flow2 is its inverse, then 
    # grid2 should return grid1 back to the original grid
    grid0_cycle = torch.nn.functional.grid_sample(grid1.permute(0,4,1,2,3), 
                                                  grid2, mode='bilinear', padding_mode='border', 
                                                  align_corners=True).permute(0,2,3,4,1)
    
    # Convert to units of pixels which helps the even the scaling of the loss along each axis
    factor = torch.FloatTensor([[[[w, h, d]]]]).view((-1,3)).to(grid0_cycle.device)
    factor = factor/torch.mean(factor)
    grid0 = grid0 * factor
    grid0_cycle = grid0_cycle * factor
    
    return torch.nn.MSELoss()(grid0_cycle, grid0)


def unsup_loss(pred_flows1, warped_imgs, pred_flows2, grids, target_img, smooth_weight, inv_weight,
               weights=(0.4, 0.6, 0.8, 1.0), sample_weight=1., alpha=0.25):

    bce_total = 0
    smooth_total = 0
    inv_total = 0
    loss = 0
    
    # Loop through the different resolutions
    for w, output_img, flow1, flow2, grid0 in zip(weights, warped_imgs, pred_flows1, pred_flows2, grids):    
        bce = photometric_loss(output_img, target_img, alpha)
        if smooth_weight>0:
            smooth = smoothness_loss(flow1)
        else:
            smooth = 0.
        inv = inv_loss(flow1, flow2, grid0)
        loss += w * (bce + smooth_weight*smooth + sample_weight*inv_weight*inv)
        bce_total += float(bce)
        smooth_total += float(smooth)
        inv_total += float(inv)
    
    return loss, bce_total, smooth_total, inv_total

def run_iter(model, input_img, target_img, loss_fnc, smooth_weight, inv_weight, res_weights, optimizer, 
             lr_scheduler, losses_cp, cur_iter, batchsize, mode='train', sample_weight=1.,
             photo_alpha=0.25):
    
    if mode=='train':
        model.train()
    else:
        model.eval()
    # Predict flows and warp images
    pred_flows1, warped_imgs = model(torch.cat((input_img, target_img), 1))
    # Predict flows in opposite direction
    pred_flows2 = model.predictor(torch.cat((target_img, input_img), 1))
        
    # Compute losses
    loss, bce_loss, smooth_loss, i_loss = loss_fnc(pred_flows1, warped_imgs, pred_flows2, model.grids,
                                                       model.gaussian_blur(target_img), 
                                                       smooth_weight, inv_weight,
                                                       res_weights, sample_weight, photo_alpha)
    
    if mode=='train':        
        # Update the gradients
        loss = 1/batchsize * loss
        loss.backward()
                
        # Save losses
        losses_cp['train_photo_loss'].append(float(bce_loss))
        losses_cp['train_smooth_loss'].append(float(smooth_loss))
        losses_cp['train_inv_loss'].append(float(i_loss))
        
        if (cur_iter%batchsize==0):
            # Adjust network weights
            optimizer.step()
            # Reset gradients
            optimizer.zero_grad(set_to_none=True)
            # Adjust learning rate
            lr_scheduler.step()
            # Free up GPU memory
            #torch.cuda.empty_cache()
            
    else:
        # Calculate average of shift in x, y, z to evaluate progress
        dx_avg, dy_avg, dz_avg = torch.mean(torch.abs(pred_flows1[0]),dim=(0,2,3,4))
        # Convert to pixels
        dx_avg = float(dx_avg * (input_img.shape[4]-1) / 2)
        dy_avg = float(dy_avg * (input_img.shape[3]-1) / 2)
        dz_avg = float(dz_avg * (input_img.shape[2]-1) / 2)
        
        # Save losses
        losses_cp['val_photo_loss'].append(float(bce_loss))
        losses_cp['val_smooth_loss'].append(float(smooth_loss))
        losses_cp['val_inv_loss'].append(float(i_loss))
        losses_cp['val_dx'].append(dx_avg)
        losses_cp['val_dy'].append(dy_avg)
        losses_cp['val_dz'].append(dz_avg)
    
    del input_img
    del target_img 
    del pred_flows1
    del pred_flows2
    del loss
                
    return model, optimizer, lr_scheduler, losses_cp

def load_imgs(val_dataset, pat_num, device, AP_expansion=None, collect_flows=True):

    with h5py.File(val_dataset.data_file, "r") as F: 

        # Load original frames
        inp_patient_nums = F['Patient Number %s' % val_dataset.dataset][:]
        inp_imgs = F['Activity Sample %s' % val_dataset.dataset]
        
        # Load ground truth target image
        gt_patient_nums = F['Patient Number %s GT' % val_dataset.dataset][:]
        gt_imgs = F['Activity %s GT' % val_dataset.dataset]
        if collect_flows:
            gt_flows = F['Flow Maps %s GT' % val_dataset.dataset]
            gt_flow_masks = F['Flow Map Masks %s GT' % val_dataset.dataset]
        
        if ('Breathing Phase %s' % val_dataset.dataset) in F.keys():
            inp_phases = F['Breathing Phase %s' % val_dataset.dataset][:]
            gt_phases = F['Breathing Phase %s GT' % val_dataset.dataset][:]
        else:
            inp_phases = F['Breathing Bin %s' % val_dataset.dataset][:]
            gt_phases = F['Breathing Bin %s GT' % val_dataset.dataset][:]
        
        # Index into original frames
        if AP_expansion is None:
            inp_indices = np.where((inp_patient_nums==pat_num))[0]
        else:
            inp_AP_expansions = F['AP Expansion %s' % val_dataset.dataset][:]
            inp_indices = np.where((inp_patient_nums==pat_num)&(inp_AP_expansions==AP_expansion))[0]
        inp_phases = inp_phases[inp_indices]
        inp_imgs = np.array([inp_imgs[i] for i in inp_indices])
        
        # Index into ground truth data
        if AP_expansion is None:
            gt_index = np.where((gt_patient_nums==pat_num))[0][0]
        else:
            gt_AP_expansions = F['AP Expansion %s GT' % val_dataset.dataset][:]
            gt_index = np.where((gt_patient_nums==pat_num)&(gt_AP_expansions==AP_expansion))[0][0]
        tgt_phase = gt_phases[gt_index]
        tgt_gt = torch.from_numpy(gt_imgs[gt_index].astype(np.float32)).unsqueeze(0)
        if collect_flows:
            gt_flows = torch.from_numpy(gt_flows[gt_index].astype(np.float32))
            gt_flow_masks = torch.from_numpy(gt_flow_masks[gt_index].astype(np.float32))
                
        # Index again into original frames to separate the frames
        target_index = np.where((inp_phases==tgt_phase))[0]
        input_indices = np.where((inp_phases!=tgt_phase))[0]
                
        tgt_img = torch.from_numpy(inp_imgs[target_index].astype(np.float32))
        inp_imgs = torch.from_numpy(inp_imgs[input_indices].astype(np.float32))
        
        # Normalize grond truth target to have same sum as inputs
        tgt_gt = tgt_gt * (torch.sum(inp_imgs) + torch.sum(tgt_img))/torch.sum(tgt_gt)
    if collect_flows:
        return (tgt_img.unsqueeze(1).to(device),
                inp_imgs.unsqueeze(1).to(device), 
                tgt_gt.unsqueeze(1).to(device),
                gt_flows.unsqueeze(1).to(device),
                gt_flow_masks.unsqueeze(1).to(device),
                tgt_phase)
    else:
        return (tgt_img.unsqueeze(1).to(device),
                inp_imgs.unsqueeze(1).to(device), 
                tgt_gt.unsqueeze(1).to(device),
                tgt_phase)

def predict_flow(model, val_dataset, input_img, target_img):
    # Normalize
    input_img = val_dataset.normalize(input_img)
    target_img = val_dataset.normalize(target_img)
    
    # Run model
    flow_predictions = model.predictor(torch.cat((input_img, 
                                               target_img), 1))
    
    # Return high-res flow
    return flow_predictions[0]

def EPE(flow_pred, flow_true, flow_mask):
    # Calculate difference between prediction and groud-truth
    flow_diff = flow_pred - flow_true
    return torch.norm(flow_diff[flow_mask], 2).mean()

def eval_sum(model, val_dataset, device, pat_num=None, return_data=False, AP_expansion=None, collect_flows=True):
    
    model.eval()
    
    if pat_num is None:
        # Select all patients
        with h5py.File(val_dataset.data_file, "r") as f:    
            pat_nums = np.unique(f['Patient Number %s' % val_dataset.dataset][:])
    else:
        pat_nums = [pat_num]
    
    loss = 0 
    flow_loss = 0
    flow_epe = 0
    # Loop through the validation patients 
    for pat_num in pat_nums:
    
        # Load target and inputs
        target_img, input_imgs, target_gt, gt_flows, gt_flow_masks, tgt_phase = load_imgs(val_dataset,
                                                                                           pat_num, device,
                                                                                          AP_expansion,
                                                                                         collect_flows=collect_flows)
        # Add to non-blurred target to compute sum
        output_sum = target_img
        if return_data:
            input_sum = target_img
            output_imgs = []
            target_imgs = []
            flows = []
        # Loop through all inputs
        for i in range(len(input_imgs)):

            # Select current frame
            input_img = input_imgs[i:i+1]

            # Normalize target to have same sum as input
            target_img = target_img * torch.sum(input_img)/torch.sum(target_img)

            # Predict flow
            flow = predict_flow(model, val_dataset, input_img.to(device), target_img.to(device))

            # Apply flow to original input img
            output_img = model.warp_frame(flow, input_img, interp_mode='nearest')
            
            # Add to sums
            output_sum = output_sum + output_img
            
            # Compare flow to ground truth
            # only considering pixels that have a ground-truth flow and have activity
            flow_mask = torch.where((gt_flow_masks[i]!=0) & (target_gt>0).repeat(1,3,1,1,1))
            flow_loss = flow_loss + 1/len(pat_nums) * 1/len(input_imgs) *  float(torch.median(torch.abs(flow[flow_mask] - 
                                                                                                      gt_flows[i][flow_mask])))
            # Also compute end-point error
            flow_epe = flow_epe + 1/len(pat_nums) * 1/len(input_imgs) * float(EPE(flow, gt_flows[i], flow_mask))
            
            if return_data:
                input_sum = input_sum + input_img
                output_imgs.append(output_img)
                target_imgs.append(target_img)
                flows.append(flow)
        
        loss = loss + 1/len(pat_nums) * float(torch.mean(torch.abs(output_sum - target_gt)))
        
    if return_data:
        return input_sum, output_sum, target_gt, input_imgs, output_imgs, target_imgs, tgt_phase, flows, gt_flows, gt_flow_masks
    else:
        return loss, flow_loss, flow_epe
    
def eval_binned(model, val_dataset, device, pat_num=None, return_data=False, AP_expansion=None):
    
    model.eval()
    
    if pat_num is None:
        # Select all patients
        with h5py.File(val_dataset.data_file, "r") as f:    
            pat_nums = np.unique(f['Patient Number %s' % val_dataset.dataset][:])
    else:
        pat_nums = [pat_num]
    
    loss = 0 
    # Loop through the validation patients 
    for pat_num in pat_nums:
    
        # Load target and inputs
        target_img, input_imgs, target_gt, tgt_phase = load_imgs(val_dataset, pat_num, 
                                                                 device, AP_expansion, collect_flows=False)
        
        # Add to non-blurred target to compute sum
        output_sum = target_img
        if return_data:
            input_sum = target_img
            output_imgs = []
            target_imgs = []
            flows = []
        # Loop through all inputs
        for i in range(len(input_imgs)):

            # Select current frame
            input_img = input_imgs[i:i+1]

            # Normalize target to have same sum as input
            target_img = target_img * torch.sum(input_img)/torch.sum(target_img)

            # Predict flow
            flow = predict_flow(model, val_dataset, input_img.to(device), target_img.to(device))

            # Apply flow to original input img
            output_img = model.warp_frame(flow, input_img, interp_mode='nearest')
            
            # Add to sums
            output_sum = output_sum + output_img
            
            if return_data:
                input_sum = input_sum + input_img
                output_imgs.append(output_img)
                target_imgs.append(target_img)
                flows.append(flow)
        
        loss = loss + 1/len(pat_nums) * float(torch.mean(torch.abs(output_sum - target_gt)))
        
    if return_data:
        return input_sum, output_sum, target_gt, input_imgs, output_imgs, target_imgs, tgt_phase, flows
    else:
        return loss