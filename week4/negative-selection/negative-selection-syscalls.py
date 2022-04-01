#%%
#import all needed libraries
import re
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

#%%
def chunk(inputfile, n, outputfile):
    """
    A function that returns a [[]] structure where the outer list represents one column, 
    Inner lists represents strings, in this case whole system calls, and the elements in these lists are the chunks.
    The functions writes the chunked data to a file ready to be used. 
    
    inputfile = string: data that needs to be chunked in a string
    n = int: integer for how big we would like the chunks to be 
    outputfile = string: file where data is outputted
    """
    #read in data
    file = open(inputfile)
    data = file.readlines()
    data = [[data[i]] for i in range(0,len(data))]

    #initialize local variables
    chunks = []
    regex = f'.{{{n}}}'

    #chunkize:
    for i in data: 
        chunks.append(re.findall(regex, i[0]))
    
    #write on outputfile
    f = open(outputfile, 'w')
    f.write(("\n".join(str(e) for e in sum(chunks, []))))
    f.close()

    return chunks


#%%
def auc_calculation(scorefile, labels, chunks, rpar, npar):
    """
    Function that returns an auc value 
    scorefile = string: path to the scorefile 
    labels = string: path to label file
    chunks = [[]]: the chunks that need to processed
    rpar = int: size of -r parameter 
    npar = int: size of -n parameter
    """
    # create a dataframe where the substrings' index shows to which string they belonged
    pandadata = {"chunks":chunks}
    df = pd.DataFrame(pandadata)
    df = df.explode("chunks")
    print(len(df))
    # add scores to df 
    scores = pd.read_csv(scorefile, header = None, sep = '\n')   
    scores = scores[0].values
    print(len(scores))
    df['scores'] = scores

    #store the index that shows to which string a substring belongs
    df['id'] = df.index

    #average the scores per string
    processed_data = df.groupby("id").mean()

    #add labels to dataframe
    labels = pd.read_csv(labels, header=None, sep='\n')
    labels = labels[0].values
    processed_data['labels'] = labels

    auc = metrics.roc_auc_score(y_true=processed_data["labels"], y_score=processed_data["scores"])
    print(auc)

#%%
#generate data: 
chunks = chunk("C:/Users/yanas/OneDrive/Masters/Untitled Folder 1/NatCom2022/week4/negative-selection/syscalls/snd-unm/snd-unm.3.test", 7, "snd-unm-3-7.test")
#%%
#generate auc value 
auc_calculation("C:/Users/yanas/OneDrive/Masters/Untitled Folder 1/NatCom2022/week4/negative-selection/scores-snd-unm-3-7-3.csv", 
"C:/Users/yanas/OneDrive/Masters/Untitled Folder 1/NatCom2022/week4/negative-selection/syscalls/snd-unm/snd-unm.3.labels", 
chunks, 
2, 
7)
# r = 3 0.48 

# %%

#The following code is an own implementation of AUC value because we didn't believe the values in the first place

# #create the unique values list
    # co_val = processed_data['scores'].unique()

    # #calculate sensitivity: 
    # sensitivity = []
    # for i in range(0,len(co_val)):        #iterate over cut off value list
    #     df_mask=processed_data['scores']>co_val[i]  #create boolean mask
    #     df_pos = processed_data[df_mask]            #select values lower than cut off values
    #     sensitivity.append(len(df_pos.loc[df_pos['labels']==1])) #append the # of values that is higher than cutoff value and is anomaly
    # print(df_pos)
    # sensitivity = [value/len(processed_data.loc[processed_data['labels']==1]) for value in sensitivity] #calculate the sensitivity percentage

    # #calculate specificity 
    # specificity = []

    # for i in range(0,len(co_val)):       #iterate over cut off value list
    #     df_mask=processed_data['scores']<co_val[i] #create boolean mask
    #     df_ts = processed_data[df_mask]            #select values lower than cut off values
    #     specificity.append(len(df_ts.loc[df_ts['labels']==0])) #append the # of values that is lower than cutoff value and is normal
    #     print(df_ts)
    # specificity = [value/len(processed_data.loc[processed_data['labels']==1]) for value in specificity] #calculate specificity

    # #calculate false positive rate: 
    # fpr = [1 - value for value in specificity] #calculate false positive rate

    # #convert to numpy array
    # fpr = np.array(fpr)
    # sensitivity = np.array(sensitivity)

    # #sort the fpr array and the sensitivity array for plotting purposes
    # ix = fpr.argsort()
    # fpr_sorted = fpr[ix]
    # sen_sorted = sensitivity[ix]

    # #calculate the auc and print
    # auc = metrics.auc(fpr_sorted, sen_sorted)
    # result = f"AUC value is {auc}"
    # print(result)

    # #plot the roc curve
    # plt.figure()
    # plt.plot(fpr_sorted, sen_sorted)
    # plt.xlabel('fpr')
    # plt.ylabel('sensitivity')
    # plt.title(f'ROC curve with -r {rpar} and -n {npar}')
    # plt.figtext(0.5, -0.1, result)
    # plt.show()

