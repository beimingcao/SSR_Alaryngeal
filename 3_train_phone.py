import time
import yaml
import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from utils.models import MyLSTM, SpeechRecognitionModel
from utils.models import RegressionLoss
from utils.models import save_model
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from utils.utils import EarlyStopping, IterMeter, data_processing_DeepSpeech
import torch.nn.functional as F

import random
from utils.transforms import ema_random_rotate, ema_time_mask, ema_freq_mask, ema_sin_noise, ema_random_scale, ema_time_seg_mask
from utils.transforms import apply_delta_deltadelta, Transform_Compose
from utils.transforms import apply_MVN
import numpy as np
import torchaudio

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def augmentation_parsing(config, train_transform):

    random_sin_noise_inj = config['data_augmentation']['random_sin_noise_inj']
    random_rotate_apply = config['data_augmentation']['random_rotate']
    random_time_mask = config['data_augmentation']['random_time_mask']
    random_freq_mask = config['data_augmentation']['random_freq_mask']
    random_scale = config['data_augmentation']['random_scale']
    random_time_seg_mask = config['data_augmentation']['random_time_seg_mask']   
    normalize_input = config['articulatory_data']['normalize_input']    

    if random_sin_noise_inj == True:
        ratio = 0.5
        noise_energy_ratio = 0.05
        noise_freq = 20
        fs = 100
        train_transform.append(ema_sin_noise(ratio, noise_energy_ratio, noise_freq, fs)) 

    if random_rotate_apply == True:
        ratio = 0.5
        angle = [-30, 30]
        train_transform.append(ema_random_rotate(ratio,  angle)) 

    if random_scale == True:
        ratio = 0.5
        scale = 0.8
        train_transform.append(ema_random_scale(ratio, scale)) 
        
    train_transform.append(apply_delta_deltadelta()) 
    
    if normalize_input == True:
        norm_transform = [apply_delta_deltadelta()]
        norm_transforms_all = Transform_Compose(norm_transform)

        train_loader_norm = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=1,
                                    shuffle=True,
                                    collate_fn=lambda x: data_processing_DeepSpeech(x, transforms = norm_transforms_all))
        EMA_all = {}
        i = 0
        for batch_idx, _data in enumerate(train_loader_norm):
            file_id, EMA, labels, input_lengths, label_lengths = _data 
            ema = EMA[0][0].T
            EMA_all[i] = ema
            i+=1

        EMA_block = np.concatenate([EMA_all[x] for x in EMA_all], 0)
        EMA_mean, EMA_std  = np.mean(EMA_block, 0), np.std(EMA_block, 0)
        
        train_transform.append(apply_MVN(EMA_mean, EMA_std))

    if random_time_mask == True:
        ratio = 0.8
        mask_num = 10
        train_transform.append(ema_time_mask(ratio, mask_num))

    if random_freq_mask == True:
        ratio = 0.8
        mask_num = 8
        train_transform.append(ema_freq_mask(ratio, mask_num))

    if random_time_seg_mask == True:
        ratio = 0.5
        mask_num = 5
        mask_length = 2
        train_transform.append(ema_time_seg_mask(ratio, mask_num, mask_length))    
        
    return train_transform, EMA_mean, EMA_std
    


def train_DeepSpeech(test_SPK, train_dataset, valid_dataset, exp_output_folder, args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ### Dimension setup ###
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)
    sel_sensors = config['articulatory_data']['sel_sensors']
    sel_dim = config['articulatory_data']['sel_dim'] 
    delta = config['articulatory_data']['delta']
    d = 3 if delta == True else 1
    D_in = len(sel_sensors)*len(sel_dim)*d
    D_out = 41
    
    ### Model setup ###
    n_cnn_layers = config['deep_speech_setup']['n_cnn_layers']
    n_rnn_layers = config['deep_speech_setup']['n_rnn_layers']    
    rnn_dim = config['deep_speech_setup']['rnn_dim']
    stride = config['deep_speech_setup']['stride']
    dropout = config['deep_speech_setup']['dropout']
    
    ### Training setup ###
    learning_rate = config['deep_speech_setup']['learning_rate']
    batch_size = config['deep_speech_setup']['batch_size']
    epochs = config['deep_speech_setup']['epochs']
    early_stop = config['deep_speech_setup']['early_stop']
    patient = config['deep_speech_setup']['patient']
    train_out_folder = os.path.join(exp_output_folder, 'training')
    if not os.path.exists(train_out_folder):
        os.makedirs(train_out_folder)
    results = os.path.join(train_out_folder, test_SPK + '_train.txt')
    
    ### Model training ###

    
    train_transform = []
    valid_transform = []
    
    train_transform, EMA_mean, EMA_std = augmentation_parsing(config, train_transform)
    
    valid_transform.append(apply_delta_deltadelta())
    valid_transform.append(apply_MVN(EMA_mean, EMA_std))
        


    train_transforms_all = Transform_Compose(train_transform)
    valid_transforms_all = Transform_Compose(valid_transform)
        
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=lambda x: data_processing_DeepSpeech(x, transforms = train_transforms_all))
                                
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=lambda x: data_processing_DeepSpeech(x, transforms = valid_transforms_all))
                                
    model = SpeechRecognitionModel(n_cnn_layers, n_rnn_layers, rnn_dim, D_out, D_in, stride, dropout).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
    criterion = torch.nn.CTCLoss(blank=40).to(device)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=int(len(train_loader)), epochs=epochs, anneal_strategy='linear')
    
    data_len = len(train_loader.dataset)
    if early_stop == True:
        print('Applying early stop.')
        early_stopping = EarlyStopping(patience=patient)
        
    iter_meter = IterMeter()
        
    with open(results, 'w') as r:    
        for epoch in range(epochs):
            model.train()
            loss_train = []
            for batch_idx, _data in enumerate(train_loader):
                file_id, ema, labels, input_lengths, label_lengths = _data 
                                    
                ema, labels = ema.to(device), labels.to(device)
                                       
                output = model(ema)  # (batch, time, n_class)

                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1) # (time, batch, n_class)

                loss = criterion(output, labels, input_lengths, label_lengths)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                iter_meter.step()
                
                loss_train.append(loss.detach().cpu().numpy())
            avg_loss_train = sum(loss_train)/len(loss_train)

            model.eval()
            loss_valid = []
            for batch_idx, _data in enumerate(valid_loader):  
                file_id, ema, labels, input_lengths, label_lengths = _data 
                ema, labels = ema.to(device), labels.to(device)           
                  
                output = model(ema)  # (batch, time, n_class)
                output = F.log_softmax(output, dim=2)
                output = output.transpose(0, 1) # (time, batch, n_class)
                loss = criterion(output, labels, input_lengths, label_lengths)    
                loss_valid.append(loss.detach().cpu().numpy())
            avg_loss_valid = sum(loss_valid)/len(loss_valid) 
            SPK = file_id[0][:3]

            early_stopping(avg_loss_valid)
            if early_stopping.early_stop:
                break

            print('epoch %-3d \t train_loss = %0.5f \t valid_loss = %0.5f' % (epoch, avg_loss_train, avg_loss_valid))
            print('epoch %-3d \t train_loss = %0.5f \t valid_loss = %0.5f' % (epoch, avg_loss_train, avg_loss_valid), file = r)                           
                            
            model_out_folder = os.path.join(exp_output_folder, 'trained_models')
            if not os.path.exists(model_out_folder):
                os.makedirs(model_out_folder)
            if early_stopping.save_model == True:
                save_model(model, os.path.join(model_out_folder, test_SPK + '_DS'))
    r.close()
    print('Training for testing SPK: ' + test_SPK + ' is done.')       
           




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/SSR_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')
    args = parser.parse_args()
    config = yaml.load(open(args.conf_dir, 'r'), Loader=yaml.FullLoader)

    data_path = os.path.join(args.buff_dir, 'data_CV')
    SPK_list = config['data_setup']['spk_list']

    for test_SPK in SPK_list:
        data_path_SPK = os.path.join(data_path, test_SPK)

        tr = open(os.path.join(data_path_SPK, 'train_data.pkl'), 'rb') 
        va = open(os.path.join(data_path_SPK, 'valid_data.pkl'), 'rb')        
        train_dataset, valid_dataset = pickle.load(tr), pickle.load(va)

    #    train_LSTM(test_SPK, train_dataset, valid_dataset, args.buff_dir, args)  
        train_DeepSpeech(test_SPK, train_dataset, valid_dataset, args.buff_dir, args)   



