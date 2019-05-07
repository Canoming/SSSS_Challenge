import torch,math
import numpy as np
from scipy.linalg import sqrtm

B=torch.tensor(sqrtm(np.array([[np.e,1/np.e],[1/np.e,np.e]])),dtype=torch.float64,device=torch.device('cpu')) # the Boltzmann matrix

sB=torch.einsum("il,jl,kl->ijk",B,B,B)

Pen = torch.einsum("iab,jbc,kcd,lde,mea->ijklm",sB,sB,sB,sB,sB)

Top = torch.einsum("pqrst,aopkb,ckqld,elrmf,gmsnh,intoj->abcdefghij",Pen,Pen,Pen,Pen,Pen,Pen)

Total = torch.einsum("abcdefghij,defghijabc->",Top,Top)
