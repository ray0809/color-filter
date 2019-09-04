import numba
import numpy as np



@numba.jit(nopython=True)
def CountPixelJit(img, mask, hist, Bin1, Bin2, Bin3):
    '''统计某通道像素落在每个桶内的频率
    输入：  
    @img：np.ndarray图像  
    @mask：图像掩码  
    @hist：空的直方图向量  
    @Binx：分别是三个通道的划分  
    输出：  
    @hist：统计好了的直方图向量
    '''
    channel1, channel2, channel3 = img[:,:,0], img[:,:,1], img[:,:,2]
    h, w ,c = img.shape
    for i in range(len(Bin1) - 1):
        for j in range(len(Bin2) - 1):
            for k in range(len(Bin3) - 1):   
                for m in range(h):
                    for n in range(w):
                        if ((Bin1[i] <= channel1[m, n] < Bin1[i+1]) & \
                            (Bin2[j] <= channel2[m, n] < Bin2[j+1]) & \
                            (Bin3[k] <= channel3[m, n] < Bin3[k+1]) & \
                            (mask[m, n])):
                            hist[i, j, k] += 1
                        
    
    return hist


def SelfHist(img, mask, Bins, max_value, channel_type='HSV'):
    '''自定义直方图统计
    输入：  
    @img：输入图像，rgb 或者 hsv，与channel_type公用  
    @bins：可以自定义区间，也可以等分区间, 大小呢  
    @max_value：每个通道的像素值的最大值，内部会加1操作  
    @mask：图像掩码  
    @channel_type：图像颜色空间类型，如果是HSV，由于它的H通道的最后一个桶和第一个桶都是同类，需要合并  
    输出：
    @hist：图像的直方图统计向量（归一化）
    '''
    new_Bins = []
    new_Lens = []
    if mask is None:
        mask = np.ones((img.shape[0], img.shape[1]))
    mask = mask > 0
    
    #三通道图
    assert len(Bins) == 3
    
    for i, Bin in enumerate(Bins):
        if isinstance(Bin, int):
            new_Lens.append(Bin)
            Bin = np.linspace(0, max_value[i]+1, Bin + 1)
            new_Bins.append(Bin)
            
        else:
            new_Bins.append(np.array(Bin))
            new_Lens.append(len(Bin) - 1)
            
            
    
    # imgs = cv2.split(img)
    hist= np.zeros(new_Lens)
    
    hist = CountPixelJit(img, mask, hist, new_Bins[0], new_Bins[1], new_Bins[2])
    if channel_type == 'HSV':
        hist[0, :, :] += hist[-1, :, :]
        hist = hist[:-1, :, :].copy()
        
    hist = hist.reshape(-1)
    hist /= hist.sum()

    return hist