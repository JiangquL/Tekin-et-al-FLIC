def biexpfit( xdata, ydata):

    """
    Fit sum of 2 exp with a = 0
    """
    
    import sys
    import argparse

    import numpy as np
    from math import sqrt
    from scipy.linalg import lstsq
    from scipy.optimize import curve_fit

    x = np.array(xdata)
    y = np.array(ydata)
    S = np.empty_like(y)
    S[0] = 0
    S[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    
    SS = np.empty_like(y)
    SS[0] = 0
    SS[1:] = np.cumsum(0.5 * (S[1:] + S[:-1]) * np.diff(x))
    
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2

    M = [[sum(SS*SS),  sum(SS*S), sum(SS*x), sum(SS)],
         [sum(SS*S),   sum(S*S),  sum(S*x2), sum(S) ],
         [sum(SS*x),   sum(S*x),  sum(x2),  sum(x) ],
         [sum(SS),     sum(S),    sum(x),   len(x) ]]
    Ycol = np.array( [ sum(SS*y), sum(S*y), sum(x*y), sum(y) ] )
    (A,B,D,E),residues,rank,singulars = list( lstsq( M, Ycol ) )


    """
    Minv = np.linalg.inv(M)    
    A,B,D,E = list( np.matmul(Minv,Ycol) )
    """

    p = (1/2)*(B + sqrt(B*B+4*A))
    q = (1/2)*(B - sqrt(B*B+4*A))
    

    beta = np.exp(p*x)
    eta = np.exp(q*x)

    betaeta = beta * eta

    L = [
          [ sum(beta*beta), sum(betaeta) ],
          [ sum(betaeta), sum(eta*eta)] ]

    Ycol = np.array( [ sum(beta*y), sum(eta*y) ] )

    (b,c),residues,rank,singulars = list( lstsq( L, Ycol ) )    

    '''
    Linv = np.linalg.inv(L)
    b,c = list( np.matmul( Linv, Ycol ) )
    '''

    # sort in ascending order (fastest negative rate first)
    (b,p),(c,q) = sorted( [[b,p],[c,q]], key=lambda x: x[1])

    return b,c,p,q


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xlsxwriter
    
def InitAnalysis(root, datafolder, figfolder, fitfolder, start, fitnum):
    writer = pd.ExcelWriter(os.path.join(root,'Summary.xlsx'), engine='xlsxwriter')
    workbook = writer.book
    datapath = os.path.join(root, datafolder)
    fnames = [i for i in os.listdir(datapath) if '.csv' in i]
    figpath = os.path.join(root, figfolder)
    fitpath = os.path.join(root, fitfolder)
        
    for i in range(0,len(fnames)):
            
    #import data
        fname = fnames[i]
        print(fname)
        rawdata = pd.read_csv(os.path.join(datapath,fname))
        end = start + fitnum - 1
        df = rawdata.loc[start:end,:].copy()
        #print(type(df))
        for j in range(1, df.shape[1]):
        #fit data
            b,c,p,q = biexpfit(np.array(df['Timestamp']), np.array(df.iloc[:, j]))
            
        #get yhat
            rawdata['yhat'+'j'] = b * np.exp(p*rawdata['Timestamp']) + c * np.exp(q * rawdata['Timestamp'])
            
        #get y/yhat
            rawdata['y/yhat'+'j'] = rawdata['Region'+'j']/rawdata['yhat'+'j']
            
        #get Delta F/F
            F0 = np.mean(rawdata['y/yhat'+'j'].tolist()[1700-start:1800-start])
            rawdata['Delta F/F'+'j'] = [(F0 - i)/F0 for i in rawdata['y/yhat'+'j'].tolist()]

        #get smoothened Delta F/F by substrating moving average
            def moving_average(data, window_size):
                # Pad the data with NaN values to handle edges
                padded_data = np.pad(data, (window_size - 1, 0), mode='constant', constant_values=np.nan)   
                # Create the moving window view of the data
                shape = (data.size, window_size)
                strides = (data.itemsize, data.itemsize)
                windowed_data = np.lib.stride_tricks.as_strided(padded_data, shape=shape, strides=strides) 
                # Calculate the moving average
                return np.nanmean(windowed_data, axis=1)

            #define window size
            window_size = 30
            #transfer pandas series to Numpy arrays
            #print(type(rawdata['Delta F/F']))
            NpDelta = np.array(rawdata['Delta F/F'+'j'])
            #print(type(NpDelta))
            #calculate moving average of Delta F/F
            ave = moving_average(NpDelta, window_size)

            rawdata['flattened Delta F/F'+'j'] = rawdata['Delta F/F'+'j']-ave
            rawdata.to_csv(os.path.join(fitpath, fname[:-4] + '_fit.csv'), index = False)

        #plot original data
                                        
            plt.figure(figsize=(8,4), dpi = 200)
            plt.plot(rawdata['Timestamp'], rawdata['Region'+'j'],label = "y")
            plt.legend()
            plt.title(label = 'Original Data')
            fig1 = os.path.join(figpath, fname[:-4] + '_orig' +'j'+'.png')
            plt.savefig(fig1)
            plt.clf()
            
        # #plot y and yhat on fitted data
                            
        #     plt.figure(figsize=(8,4), dpi = 200)
        #     plt.plot(rawdata['Timestamp'][start:start+fitnum], rawdata['Region'+'j'][start:start+fitnum],label = "y")
        #     plt.plot(rawdata['Timestamp'][start:start+fitnum], rawdata['yhat'+'j'][start:start+fitnum], label = "yhat")
        #     plt.legend()
        #     plt.title(label = ' y and yhat - fitted start from ' + str(start) + ' to ' + (str(start + fitnum)))
        #     fig2 = os.path.join(figpath, fname[:-4] + '_yhat' +'j'+'.png')
        #     plt.savefig(fig2)
        #     plt.clf()
                            
        # #plot Delta F/F
        #     plt.ylim([-0.5,0.5])
        #     plt.plot(rawdata['Timestamp'][start:start+fitnum], rawdata['Delta F/F'][start:start+fitnum], label = "delta F/F", color = 'black')
        #     plt.plot(rawdata['Timestamp'][start:start+fitnum], np.zeros(fitnum), color = 'red')
        #     plt.plot(rawdata['Timestamp'][start:start+fitnum], [0.05]*fitnum, color = 'red', linestyle = 'dashed')
        #     plt.plot(rawdata['Timestamp'][start:start+fitnum], [-0.05]*fitnum, color = 'red', linestyle = 'dashed')
        #     plt.legend()
        #     plt.title(label = 'Delta F/F')
        #     fig3 = os.path.join(figpath, fname[:-4] + '_deltaF' +'.png')
        #     plt.savefig(fig3)
        #     plt.clf()     
                            
        # #plot all Delta F/F
        #     plt.ylim([-0.5,0.5])
        #     plt.plot(rawdata['Timestamp'], rawdata['Delta F/F'], label = "delta F/F", color = 'black')
        #     plt.title(label = 'all Delta F/F')
        #     fig4 = os.path.join(figpath, fname[:-4] + '_deltaF' + 'all' +'.png')
        #     plt.savefig(fig4)
        #     plt.clf()

        #plot flattened Delta F/F
            plt.ylim([-0.5,0.5])
            plt.plot(rawdata['Timestamp'], rawdata['flattened Delta F/F'+'j'], label = "flattened Delta F/F", color = 'black')
            plt.title(label = 'flattened Delta F/F')
            fig5 = os.path.join(figpath, fname[:-4] + '_flattened deltaF' + 'all' +'j' +'.png')
            plt.savefig(fig5)
            plt.clf()

            plt.close('all')   


        


        # save xlsx file
        
            rawdata.to_excel(writer, sheet_name=fname[:-4],  index=False)
            worksheet = writer.sheets[fname[:-4]] 
            # worksheet.insert_image('K2', fig1)
            # worksheet.insert_image('K30', fig2)
            # worksheet.insert_image('K60', fig3)
                
        
    #writer.save()
    workbook.close()

os.chdir(r"D:\2P_fly_proboscis")
root = r'D:\2P_fly_proboscis'
datafolder = 'original data'
figfolder = 'fig_20230705'
if not os.path.exists(figfolder):
    os.makedirs(figfolder)
fitfolder = 'fig_20230705'
if not os.path.exists(fitfolder):
    os.makedirs(fitfolder)
start = 500
fitnum = 1500
InitAnalysis(root, datafolder, figfolder, fitfolder, start, fitnum)

exit(1)


#below are testing codes
import pandas as pd
import numpy as np
#fname = '10-1_ctrl_all_buffer_177.csv'
#fname = 'ctrl hom rec 20mM D-glu 3-165_Cycle00001-botData.csv'
fname = '9-30_ctrl_all_buffer_183.csv'

rawdata = pd.read_csv(fname)

#rawdataMax = np.argmax(rawdata['Region 1'])


#start = rawdataMax
start = 100
length = 3000
end = start + length - 1

df = rawdata.loc[start:end,:].copy()
#df = rawdata.loc[start:,:].copy()
df.shape

df['Region 1']/df['Timestamp']

df = rawdata.loc[start:end,:].copy()
xdata = np.array(df['Timestamp'])
ydata = np.array(df['Region 1'])

b,c,p,q = biexpfit(xdata, ydata)

print(b,c,p,q)

# yhat = b * exp(px) + c * exp(qx)

yhat = b * np.exp(p*df['Timestamp']) + c * np.exp(q * df['Timestamp'])
df["yhat"] = yhat

yhat_raw = b * np.exp(p*rawdata['Timestamp']) + c * np.exp(q * rawdata['Timestamp'])
rawdata["yhat"] = yhat_raw

import matplotlib.pyplot as plt
plt.figure(figsize=(8,6), dpi = 200)
plt.plot(df['Timestamp'], df['Region 1'],label = "y")
plt.plot(df['Timestamp'], yhat, label = "yhat")
plt.legend()
plt.show()

import matplotlib.pyplot as plt
plt.figure(figsize=(8,6), dpi = 200)
plt.plot(df['Timestamp'], df['Region 1'],label = "y")
plt.plot(df['Timestamp'], yhat, label = "yhat")
plt.legend()
plt.savefig('test1.png')

plt.clf()
num1 = 0
num2 = 3000
#plt.plot(df['Timestamp'][num1:num2], df['Region 1'][num1:num2],label = "y")
#plt.plot(df['Timestamp'][num1:num2], yhat[num1:num2], label = "yhat")
plt.ylim([0.5,2])
plt.plot(df['Timestamp'][num1:num2], df['Region 1'][num1:num2]/yhat[num1:num2], label = "y/yhat", color = 'black')
plt.plot(df['Timestamp'][num1:num2], np.ones(df.shape[0])[num1:num2], color = 'red')
plt.legend()
plt.savefig('test2.png')

import matplotlib.pyplot as plt
plt.figure(figsize=(8,4), dpi = 200)
num1 = 0
num2 = 3000
#plt.plot(df['Timestamp'][num1:num2], df['Region 1'][num1:num2],label = "y")
#plt.plot(df['Timestamp'][num1:num2], yhat[num1:num2], label = "yhat")
plt.ylim([0.5,2])
plt.plot(df['Timestamp'][num1:num2], df['Region 1'][num1:num2]/yhat[num1:num2], label = "y/yhat", color = 'black')
plt.plot(df['Timestamp'][num1:num2], np.ones(df.shape[0])[num1:num2], color = 'red')
plt.legend()
plt.show()


target[np.argmax(target)+start]

import matplotlib.pyplot as plt
plt.figure(figsize=(8,4), dpi = 200)


num1 = 100
num2 = 3000

target = df['Region 1'][num1:num2]/yhat[num1:num2].copy()
target[1700-start:1800-start]

F0 = np.mean(target[1700-start:1800-start])
F1 = [(i - F0)/F0 for i in target]


plt.ylim([-0.5,0.5])
plt.plot(df['Timestamp'][num1:num2], F1, label = "delta F/F", color = 'black')
plt.plot(df['Timestamp'][num1:num2], np.zeros(df.shape[0])[num1:num2], color = 'red')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
length2 = 3000
plt.figure(figsize=(8,6), dpi = 200)
plt.plot(rawdata['Timestamp'][start:length2+start-1], rawdata['Region 1'][start:length2+start-1],label = "y")
plt.plot(rawdata['Timestamp'][start:length2+start-1], yhat_raw[start:length2+start-1] , label = "yhat")
plt.legend()
plt.show()