from flownet import FlowNetPET
from utils import unsup_loss, parseArguments, run_iter, XCAT3DDataset, eval_sum

import configparser
import time
import numpy as np
import torch
from collections import defaultdict
import os
# tensorboard imports for logging
from torch.nn.parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

np.random.seed(1)
torch.manual_seed(1)

print('Using Torch version: %s' % (torch.__version__))

print('Using a %s device' % (device))

# Collect the command line arguments
args = parseArguments()
model_name = args.model_name
verbose_iters = args.verbose_iters
cp_time = args.cp_time
data_dir = args.data_dir

# Directories
cur_dir = os.path.dirname(__file__)
config_dir = os.path.join(cur_dir, 'configs/')
model_dir = os.path.join(cur_dir, 'models/')
progress_dir = os.path.join(cur_dir, '/scratch/tmp/a_wels03/progress/')
if args.data_dir is None:
    data_dir = os.path.join(cur_dir, 'data/')

# Model configuration
config = configparser.ConfigParser()
config.read(config_dir+model_name+'.ini')
architecture_config = config['ARCHITECTURE']
print('\nCreating model: %s'%model_name)

print('\nConfiguration:')
for key_head in config.keys():
    if key_head=='DEFAULT':
        continue
    print('  %s' % key_head)
    for key in config[key_head].keys():
        print('    %s: %s'%(key, config[key_head][key]))
        
# TRAINING PARAMETERS
batchsize = 1
learning_rate = 0.0003
lr_decay_batch_iters = 8000
lr_decay = 0.7
total_batch_iters = 150000
smooth_weight = 0.0
res_weights = [1.0, 1.0, 1.0, 1.0]
l2_weight = 0.0
inv_weight = 1100.0
photo_alpha = 0.3

# Check for pre-trained weights
model_filename =  os.path.join('/scratch/tmp/a_wels03/checkpoints/optical_flow_3d.pt')
model_filename_save =  os.path.join('/scratch/tmp/a_wels03/checkpoints/optical_flow_3d_finetuned.pt')
if os.path.exists(model_filename):
    fresh_model = False
else:
    fresh_model = True

# Construct Network
print('\nBuilding networks...')
model = DataParallel(FlowNetPET(), device_ids=[0,1]).cuda()

# Display model architectures
print('\n\nARCHITECTURE:\n')
print(model.predictor)

# Construct optimizer
optimizer = torch.optim.Adam(model.parameters(), learning_rate,
                             weight_decay=l2_weight, betas=(0.9, 0.999))

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                               step_size=lr_decay_batch_iters, 
                                               gamma=lr_decay)

# Loss for training
loss_fnc = unsup_loss

# Create data loaders
train_dataset = XCAT3DDataset(split='training')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4,
                                               pin_memory=True)
val_dataset = XCAT3DDataset(split='validation')
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4,
                                            pin_memory=True)

print('The training set consists of %i paired frames...' % (len(train_dataset)))

# Model state
if fresh_model:
    # this should not happen
    exit()
    print('\nStarting fresh model to train...')
    cur_iter = 1
    losses = defaultdict(list)
else:
    print('\nLoading saved model to continue training...')
    # Load model info
    model.load_state_dict(torch.load('/scratch/tmp/a_wels03/checkpoints/optical_flow_3d.pt'))
    # we start after iteration 1000
    cur_iter = 1001

    
    # Load optimizer states
   # optimizer.load_state_dict(checkpoint['optimizer'])
   # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    

def train_network(model, optimizer, lr_scheduler, cur_iter):
    print('Training the network with a batchsize of %i...' % (batchsize))
    print('Progress will be displayed every %i batch iterations and the model will be saved every %i minutes.'%
          (verbose_iters, cp_time))
    
    # Train the neural networks
    losses_cp = defaultdict(list)
    cp_start_time = time.time()
    writer = SummaryWriter('/scratch/tmp/a_wels03/logs/flownet-pet-finetune')
    while cur_iter < (total_batch_iters*batchsize):
        for train_batch in train_dataloader:    
            (inputs, targets, _) = [x.cuda() for x in train_batch]
            
            
            model, optimizer, lr_scheduler, losses_cp = run_iter(model, 
                                                                inputs,
                                                                targets,
                                                                 loss_fnc, smooth_weight, inv_weight,
                                                                 res_weights, optimizer,
                                                                 lr_scheduler, losses_cp, cur_iter, 
                                                                 batchsize, mode='train',
                                                                 )
            
            #end.record()
            # Waits for everything to finish running
            #torch.cuda.synchronize()
            #print('Iter time: %0.5f s ' % ((start.elapsed_time(end)/1e3)))
            
            # Evaluate validation set and display losses
            if cur_iter % (verbose_iters*batchsize) == 0:
                
                # Only run 200 iters
                i=0
                for val_batch in val_dataloader:

                    with torch.no_grad():
                        model, optimizer, lr_scheduler, losses_cp = run_iter(model, 
                                                                             val_batch['input_img'].to(device), 
                                                                             val_batch['target_img'].to(device),
                                                                             loss_fnc, smooth_weight, inv_weight,
                                                                             res_weights, optimizer, 
                                                                             lr_scheduler, losses_cp, 
                                                                             cur_iter, 
                                                                             batchsize, mode='validation',
                                                                             sample_weight=val_batch['loss_weight'].to(device),
                                                                            photo_alpha=photo_alpha)

                    i+=1
                    if i>200:
                        break
                
                with torch.no_grad():
                    # Compare sum of warped images to ground truth sample for 5 random patients
                    gt_img_loss, gt_flow_loss, gt_flow_epe = eval_sum(model, val_dataset, 
                                                         device=device)
                    losses['val_gt_img'].append(gt_img_loss)
                    losses['val_gt_flow'].append(gt_flow_loss)
                    losses['val_gt_epe'].append(gt_flow_epe)
                
                # Calculate averages
                for k in losses_cp.keys():
                    losses[k].append(np.mean(losses_cp[k]))
                losses['batch_iters'].append(cur_iter/batchsize)

                # Print current status
                print('\nBatch Iterations: %i/%i ' % (cur_iter/batchsize, total_batch_iters))
                print('Losses:')
                print('| Dataset | Photometric |  Smoothness  |   Invertible   |' +
                      ' avg(dx) | avg(dy) | avg(dz) | Image GT | Flow GT | Flow EPE |')
                print('|  Train  |    %0.3f    |     %0.3f    |    %0.2e   |'% (losses['train_photo_loss'][-1],
                                  model_filename =  os.path.join('/scratch/tmp/a_wels03/checkpoints/optical_flow_3d.pt')
                                           losses['train_smooth_loss'][-1],
                                                                             losses['train_inv_loss'][-1]))
                print('|   Val   |    %0.3f    |     %0.3f    |    %0.2e   |' % (losses['val_photo_loss'][-1],
                                                                              losses['val_smooth_loss'][-1],
                                                                              losses['val_inv_loss'][-1]) +
                      '   %0.2f  |   %0.2f  |   %0.2f  |    %0.2f  |   %0.2f  |   %0.2f  |' % (losses['val_dx'][-1],
                                                                                               losses['val_dy'][-1],
                                                                                               losses['val_dz'][-1],
                                                                                               losses['val_gt_img'][-1],
                                                                                               losses['val_gt_flow'][-1],
                                                                                               losses['val_gt_epe'][-1]))
                print('\n') 

                # Save losses to file to analyze throughout training. 
                np.save(os.path.join(progress_dir, model_name+'_losses.npy'), losses)
                # write to tensorboard
                writer.add_scalar('train/photometric_loss', losses['train_photo_loss'][-1], cur_iter)
                writer.add_scalar('train/smoothness_loss', losses['train_smooth_loss'][-1], cur_iter)
                writer.add_scalar('train/invertible_loss', losses['train_inv_loss'][-1], cur_iter)
                writer.add_scalar('val/photometric_loss', losses['val_photo_loss'][-1], cur_iter)
                writer.add_scalar('val/smoothness_loss', losses['val_smooth_loss'][-1], cur_iter)
                writer.add_scalar('val/invertible_loss', losses['val_inv_loss'][-1], cur_iter)
                writer.add_scalar('val/dx', losses['val_dx'][-1], cur_iter)
                writer.add_scalar('val/dy', losses['val_dy'][-1], cur_iter)
                writer.add_scalar('val/dz', losses['val_dz'][-1], cur_iter)
                writer.add_scalar('val/image_gt', losses['val_gt_img'][-1], cur_iter)
                writer.add_scalar('val/flow_gt', losses['val_gt_flow'][-1], cur_iter)
                writer.add_scalar('val/flow_epe', losses['val_gt_epe'][-1], cur_iter)


                # Reset checkpoint loss dict
                losses_cp = defaultdict(list)
                # Free some GPU memory
                torch.cuda.empty_cache()

            # Increase the iteration
            cur_iter += 1
            
            # Save periodically
            if time.time() - cp_start_time >= cp_time*60:
                print('Saving network...')

                torch.save({'batch_iters': cur_iter,
                            'losses': losses,
                            'optimizer' : optimizer.state_dict(),
                            'lr_scheduler' : lr_scheduler.state_dict(),
                            'model' : model.state_dict(),
                            'x_mean': val_dataset.x_mean,
                            'x_std': val_dataset.x_std}, 
                            model_filename_save)
                
                cp_start_time = time.time()
                
            if cur_iter>(total_batch_iters*batchsize):
                print('Saving network...')

                torch.save({'batch_iters': cur_iter,
                            'losses': losses,
                            'optimizer' : optimizer.state_dict(),
                            'lr_scheduler' : lr_scheduler.state_dict(),
                            'model' : model.state_dict(),
                            'x_mean': val_dataset.x_mean,
                            'x_std': val_dataset.x_std}, 
                            model_filename_save)
                                
                break
                
# Run the training
if __name__=="__main__":
    train_network(model, optimizer, lr_scheduler, cur_iter)
    print('\nTraining complete.')
