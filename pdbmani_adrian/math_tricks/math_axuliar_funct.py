import numpy as np

def smoothListGaussian(list,degree=5):
     window=degree*2-1
     weight=np.array([1.0]*window)
     weightGauss=[]
     for i in range(window):
         i=i-degree+1
         frac=i/float(window)
         gauss=1/(np.exp((4*(frac))**2))
         weightGauss.append(gauss)
     weight=np.array(weightGauss)*weight
     smoothed=[0.0]*(len(list)-window)
     for i in range(len(smoothed)):
         smoothed[i]=sum(np.array(list[i:i+window])*weight)/sum(weight)
     return smoothed

def smooth_to_plot(x_axis,y_axis,degree=5):
    x_dat = x_axis[degree:(-1)*(degree-1)]
    y_dat = smoothListGaussian(y_axis,degree)
    return x_dat, y_dat

def norma_array(data):
    max_val = data.max()
    return data/float(max_val)

def z_score_array(data):
    mean = np.average(data)
    std = np.std(data)
    return (data-mean)/std

#######
def center_of_mass(data):
    """Actually Geometric center"""
    assert isinstance(data, np.ndarray )
    data = np.array(data)
    gc = data.mean(axis=0)
    return gc 
