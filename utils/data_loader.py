import numpy as np
from pandas import DataFrame
from torch.utils.data import Dataset
import torch

class DDIDataset(Dataset):
    def __init__(self,x,y):
        self.len=len(x)
        self.x_data=torch.from_numpy(x)
        self.y_data=torch.from_numpy(y)
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len

class MDDIData:
    # feature_list = ["drugbank", "smile", "MACCS"]
    def __init__(self, df_drug, feature_list, mechanism, action, drugA, drugB):
        self.df_drug = df_drug
        self.feature_list = feature_list
        self.mechanism = mechanism
        self.action = action
        self.drugA = drugA
        self.drugB = drugB
        
        # ndarray, ndarray, int, list
        self.feature, self.label, self.event_num, self.each_event_num = self.prepare(mechanism, action)
        
    
    def feature_vector(self, feature_name, df):
        def Jaccard(matrix):
            matrix = np.mat(matrix)

            numerator = matrix * matrix.T

            denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T

            return numerator / denominator
        
        all_feature = []
        drug_list = np.array(df[feature_name]).tolist()
        # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
        for i in drug_list:
            for each_feature in i.split('|'):
                if each_feature not in all_feature:
                    all_feature.append(each_feature)  # obtain all the features
        feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
        df_feature = DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
        for i in range(len(drug_list)):
            for each_feature in df[feature_name].iloc[i].split('|'):
                df_feature[each_feature].iloc[i] = 1
        
        df_feature = np.array(df_feature)
        sim_matrix = np.array(Jaccard(df_feature))
        
        print(feature_name+" len is:"+ str(len(sim_matrix[0])))
        return sim_matrix
    
    def prepare(self, df_drug, feature_list,mechanism,action,drugA,drugB):
        d_label = {}
        d_feature = {}

        # Transfrom the interaction event to number
        d_event=[]
        for i in range(len(mechanism)):
            d_event.append(mechanism[i]+" "+action[i])

        count={}
        for i in d_event:
            if i in count:
                count[i]+=1
            else:
                count[i]=1
        event_num=len(count)
        list1 = sorted(count.items(), key=lambda x: x[1],reverse=True)
        each_event_num=[]
        for i in range(len(list1)):
            d_label[list1[i][0]]=i
            each_event_num.append(list1[i][1])

        vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)  #vector=[]
        for i in feature_list:
            # TODO: need to update
            #vector = np.hstack((vector, feature_vector(i, df_drug, vector_size)))#1258*1258
            tempvec=self.feature_vector(i, df_drug)
            vector = np.hstack((vector,tempvec))
        # Transfrom the drug ID to feature vector
        for i in range(len(np.array(df_drug['name']).tolist())):
            d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i]

        # Use the dictionary to obtain feature vector and label
        new_feature = []
        new_label = []

        for i in range(len(d_event)):
            temp=np.hstack((d_feature[drugA[i]],d_feature[drugB[i]]))
            new_feature.append(temp)
            new_label.append(d_label[d_event[i]])

        new_feature = np.array(new_feature) #323539*....
        new_label = np.array(new_label)  #323539

        return new_feature, new_label, event_num, each_event_num
    
    
    # def prepare(self, df_drug, feature_list,mechanism,action,drugA,drugB):
    #     d_label = {}
    #     d_feature = {}

    #     # Transfrom the interaction event to number
    #     d_event=[]
    #     for i in range(len(mechanism)):
    #         d_event.append(mechanism[i]+" "+action[i])

    #     count={}
    #     for i in d_event:
    #         if i in count:
    #             count[i]+=1
    #         else:
    #             count[i]=1
    #     event_num=len(count)
    #     list1 = sorted(count.items(), key=lambda x: x[1],reverse=True)
    #     each_event_num=[]
    #     for i in range(len(list1)):
    #         d_label[list1[i][0]]=i
    #         each_event_num.append(list1[i][1])

    #     vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)  #vector=[]
    #     for i in feature_list:
    #         #vector = np.hstack((vector, feature_vector(i, df_drug, vector_size)))#1258*1258
    #         tempvec=self.feature_vector(i, df_drug)
    #         vector = np.hstack((vector,tempvec))
    #     # Transfrom the drug ID to feature vector
    #     for i in range(len(np.array(df_drug['name']).tolist())):
    #         d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i]

    #     # Use the dictionary to obtain feature vector and label
    #     new_feature = []
    #     new_label = []

    #     for i in range(len(d_event)):
    #         temp=np.hstack((d_feature[drugA[i]],d_feature[drugB[i]]))
    #         new_feature.append(temp)
    #         new_label.append(d_label[d_event[i]])

    #     new_feature = np.array(new_feature) #323539*....
    #     new_label = np.array(new_label)  #323539

    #     return new_feature, new_label,event_num,each_event_num