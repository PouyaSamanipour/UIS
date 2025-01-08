import numpy as np
import torch
from Enum_module_BF import updating_NN



def updating_BF_LV(NN,h,enumerate_poly,D):
    enumerate_poly_new=[]
    model=torch.jit.load(h)
    model.eval()
    model_original=torch.jit.load(NN)
    model_original.eval()
    hyperplanes=[]
    biases=[]
    hyperplanes=model.fc1.weight
    # all_hyperplanes=model.fc1.weight
    # all_bias=model.fc1.bias
    biases=model.fc1.bias
    W=model.out.weight
    weights=model_original.out.weight
    c=model.out.bias
    new_hype=[] 
    new_bias=[]
    A_dyn=[]
    B_dyn=[]
    all_hyperplanes=[]
    all_bias=[]
    with torch.no_grad():
        for j,i in enumerate(enumerate_poly):
            vertices=i
            vertices=torch.FloatTensor(vertices).cuda()
            h_val=model(vertices)
            if torch.max(h_val)>1e-8 and torch.min(h_val)<-1e-8:
                enumerate_poly_new.append(i)
                new_hype.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                new_bias.extend((W@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())
                all_hyperplanes.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                all_bias.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())
                A_dyn.append((weights@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                B_dyn.append((weights@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())
            elif torch.min(h_val)>-1e-10:
                enumerate_poly_new.append(i)
                A_dyn.append((weights@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                B_dyn.append((weights@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())
                all_hyperplanes.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@hyperplanes).detach().cpu().numpy())
                all_bias.append((W@torch.diag(torch.FloatTensor(D[j]).cuda())@biases+c).detach().cpu().numpy())

                # all_hyperplanes=torch.cat([all_hyperplanes,new_hype],0)
                # all_bias=torch.cat([all_bias,new_bias],0)
                # weights=torch.cat([weights,torch.FloatTensor(np.zeros((2,len(new_hype)))).cuda()],1)


    # NN,_,_,_,_=updating_NN(NN,all_hyperplanes.size()[1],all_hyperplanes,all_bias,weights,c)
    return enumerate_poly_new,np.array(new_hype).squeeze(1),np.array(new_bias),A_dyn,B_dyn,np.array(all_hyperplanes).squeeze(1),np.array(all_bias)


    
            
