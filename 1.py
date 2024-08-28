import torch

a=torch.tensor([1,2,3,4]).to('cuda')

mask = a!=1

b=a[mask]
c=torch.tensor([1,2,3,4])
d=torch.cat((c,a))
import pdb
pdb.set_trace()
