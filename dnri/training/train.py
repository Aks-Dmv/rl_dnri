import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from . import train_utils
import dnri.utils.misc as misc

import time, os

import random
import numpy as np

def train(model, train_data, val_data, params, train_writer, val_writer):
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    val_batch_size = params.get('val_batch_size', batch_size)
    if val_batch_size is None:
        val_batch_size = batch_size
    accumulate_steps = params.get('accumulate_steps')
    training_scheduler = params.get('training_scheduler', None)
    q_training_scheduler = params.get('training_scheduler', None)
    num_epochs = params.get('num_epochs', 100)
    val_interval = params.get('val_interval', 1)
    val_start = params.get('val_start', 0)
    clip_grad = params.get('clip_grad', None)
    clip_grad_norm = params.get('clip_grad_norm', None)
    normalize_nll = params.get('normalize_nll', False)
    normalize_kl = params.get('normalize_kl', False)
    tune_on_nll = params.get('tune_on_nll', False)
    verbose = params.get('verbose', False)
    val_teacher_forcing = params.get('val_teacher_forcing', False)
    continue_training = params.get('continue_training', False)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_data_loader = DataLoader(val_data, batch_size=val_batch_size)
    lr = params['lr']
    wd = params.get('wd', 0.)
    mom = params.get('mom', 0.)
    
    # don't send q_net params to policy optimizer
    for p in model.decoder.q_net.parameters():
        p.requires_grad = False
    for p in model.Q_graph.parameters():
        p.requires_grad = False
    model_params = [param for param in model.parameters() if param.requires_grad]
    if params.get('use_adam', False):
        opt = torch.optim.Adam(model_params, lr=lr, weight_decay=wd)
        q_opt = torch.optim.Adam(list(model.decoder.q_net.parameters()) + list(model.Q_graph.parameters()), lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.SGD(model_params, lr=lr, weight_decay=wd, momentum=mom)
        q_opt = torch.optim.SGD(list(model.decoder.q_net.parameters()) + list(model.Q_graph.parameters()), lr=lr, weight_decay=wd, momentum=mom)

    working_dir = params['working_dir']
    best_path = os.path.join(working_dir, 'best_model')
    checkpoint_dir = os.path.join(working_dir, 'model_checkpoint')
    training_path = os.path.join(working_dir, 'training_checkpoint')
    if continue_training:
        print("RESUMING TRAINING")
        model.load(checkpoint_dir)
        train_params = torch.load(training_path)
        start_epoch = train_params['epoch']
        opt.load_state_dict(train_params['optimizer'])
        q_opt.load_state_dict(train_params['q_optimizer'])
        best_val_result = train_params['best_val_result']
        best_val_epoch = train_params['best_val_epoch']
        print("STARTING EPOCH: ",start_epoch)
    else:
        start_epoch = 1
        best_val_epoch = -1
        best_val_result = 10000000
    
    training_scheduler = train_utils.build_scheduler(opt, params)
    q_training_scheduler = train_utils.build_scheduler(q_opt, params)
    end = start = 0 
    misc.seed(1)
    for epoch in range(start_epoch, num_epochs+1):
        print("EPOCH", epoch, (end-start))
        model.train()
        model.train_percent = epoch / num_epochs
        start = time.time() 
        for batch_ind, batch in enumerate(train_data_loader):
            inputs = batch['inputs']
            if gpu:
                inputs = inputs.cuda(non_blocking=True)
            
            # critic training
            for p in model.decoder.q_net.parameters():
                p.requires_grad = True
            for p in model.Q_graph.parameters():
                p.requires_grad = True
            q_opt.zero_grad()
            opt.zero_grad()
            for _ in range(5):
                loss_critic, loss_nll = model.calculate_loss_q(inputs, is_train=True, return_logits=True)
                loss_critic.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                q_opt.step()
                q_opt.zero_grad()
                opt.zero_grad()
            for p in model.decoder.q_net.parameters():
                p.requires_grad = False
            for p in model.Q_graph.parameters():
                p.requires_grad = False

            # Finally, update Q_target networks by polyak averaging.
            # We only do it for the q_net, as we don't use the target policy
            with torch.no_grad():
                polyak=0.995
                for p, p_targ in zip(model.decoder.q_net.parameters(), model.decoder_targ.q_net.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                
                for p, p_targ in zip(model.Q_graph.parameters(), model.Q_graph_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

            # policy training
            q_opt.zero_grad()
            opt.zero_grad()
            for _ in range(2):
                loss, loss_policy, loss_kl, logits, _ = model.calculate_loss_pi(inputs, is_train=True, return_logits=True)
                loss.backward()       
                opt.step()
                opt.zero_grad()
                q_opt.zero_grad()

            
        if training_scheduler is not None:
            training_scheduler.step()
            q_training_scheduler.step()
        
        if train_writer is not None:
            train_writer.add_scalar('loss', loss.item(), global_step=epoch)
            if normalize_nll:
                train_writer.add_scalar('NLL', loss_nll.mean().item(), global_step=epoch)
            else:
                train_writer.add_scalar('NLL', loss_nll.mean().item()/(inputs.size(1)*inputs.size(2)), global_step=epoch)
            
            train_writer.add_scalar("KL Divergence", loss_kl.mean().item(), global_step=epoch)
        model.eval()
        opt.zero_grad()

        total_nll = 0
        total_kl = 0
        if verbose:
            print("COMPUTING VAL LOSSES")
        with torch.no_grad():
            for batch_ind, batch in enumerate(val_data_loader):
                inputs = batch['inputs']
                if gpu:
                    inputs = inputs.cuda(non_blocking=True)
                loss_critic, loss_nll = model.calculate_loss_q(inputs, is_train=False, teacher_forcing=val_teacher_forcing, return_logits=True)
                loss, loss_policy, loss_kl, logits, _ = model.calculate_loss_pi(inputs, is_train=False, teacher_forcing=val_teacher_forcing, return_logits=True)
                total_kl += loss_kl.sum().item()
                total_nll += loss_nll.sum().item()
                if verbose:
                    print("\tVAL BATCH %d of %d: %f, %f"%(batch_ind+1, len(val_data_loader), loss_nll.mean(), loss_kl.mean()))
            
        total_kl /= len(val_data)
        total_nll /= len(val_data)
        total_loss = model.kl_coef*total_kl + total_nll #TODO: this is a thing you fixed
        if val_writer is not None:
            val_writer.add_scalar('loss', total_loss, global_step=epoch)
            val_writer.add_scalar("NLL", total_nll, global_step=epoch)
            val_writer.add_scalar("KL Divergence", total_kl, global_step=epoch)
        if tune_on_nll:
            tuning_loss = total_nll
        else:
            tuning_loss = total_loss
        if tuning_loss < best_val_result:
            best_val_epoch = epoch
            best_val_result = tuning_loss
            print("BEST VAL RESULT. SAVING MODEL...")
            model.save(best_path)
        model.save(checkpoint_dir)
        torch.save({
                    'epoch':epoch+1,
                    'optimizer':opt.state_dict(),
                    'q_optimizer':q_opt.state_dict(),
                    'best_val_result':best_val_result,
                    'best_val_epoch':best_val_epoch,
                   }, training_path)
        print("EPOCH %d EVAL: "%epoch)
        print("\tCURRENT VAL LOSS: %f"%tuning_loss)
        print("\tBEST VAL LOSS:    %f"%best_val_result)
        print("\tBEST VAL EPOCH:   %d"%best_val_epoch)
        end = time.time()

    