


# class inferencer:
#     def __init__(self,model,dataloader) -> None:

from torch.utils.data import Dataset
import multiprocessing as mp
import os


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


class test_dataset(Dataset):
    def __init__(self,data) -> None:
        self.data=data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

def foo(dd):
    a=[1,2,3,4]

    d=test_dataset(a)
    print('hello',os.getpid())
    q.put(d[0:2])
if __name__=='__main__':
    mp.set_start_method('fork')
    q = mp.Queue()
    p = mp.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    p.join()