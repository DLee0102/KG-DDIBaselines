import numpy as np
from pandas import DataFrame
from torch.utils.data import Dataset
import torch
from collections import defaultdict

class KGInfo:
    def __init__(self, **kwargs):
        self.kg = kwargs["kg"]
        self.tail_len = kwargs["tail_len"]
        self.relation_len = kwargs["relation_len"]
    
    def get(self):
        return self.kg, self.tail_len, self.relation_len

class DDIDataset(Dataset):
    def __init__(self,x,y):
        self.len=len(x)
        self.x_data=torch.from_numpy(x)
        self.y_data=torch.from_numpy(y)
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len

class MultiDataLoader:
    # feature_list = ["drugbank", "smile", "MACCS"]
    def __init__(self, df_drug, mechanism, action, drugA, drugB):
        self.df_drug = df_drug
        self.mechanism = mechanism
        self.action = action
        self.drugA = drugA
        self.drugB = drugB
        
        self.drug_nameid_dict = self.getDrugIdDict(self.df_drug)
        
        self.dataset1_kg = self.read_dataset(self.drug_nameid_dict, 1)
        self.dataset2_kg = self.read_dataset(self.drug_nameid_dict, 2)
        self.dataset3_kg = self.read_dataset(self.drug_nameid_dict, 3)
        self.dataset4_kg = self.read_dataset(self.drug_nameid_dict, 4)
        
        # ndarray, ndarray, int, list
        self.feature, self.label, self.event_num, self.each_event_num = self.prepare(self.df_drug, self.mechanism, self.action, self.drugA, self.drugB)
        
    def getDrugIdDict(self, df_drug):
        tempdict = {}
        for i in df_drug["name"]:
            tempdict[i] = len(tempdict)
        return tempdict
    
    def read_dataset(self, drug_name_id, num):
        # kg: key: drug_id, value: list of (tail_id, relation_id)
        kg = defaultdict(list)
        # tails: key: tail_name, value: tail_id
        tails = {}
        # relations: key: relation_name, value: relation_id 
        relations = {}
        
        drug_list=[]
        filename = "/data/dingli/mydata/KG-DDIBaselines/datasets/dataset"+str(num)+".txt"
        with open(filename, encoding="utf8") as reader:
            for line in reader:
                string= line.rstrip().split('//',2)
                head=string[0]
                tail=string[1]
                relation=string[2]
                drug_list.append(drug_name_id[head])
                if tail not in tails:
                    tails[tail] = len(tails)
                if relation not in relations:
                    relations[relation] = len(relations)
                    
                # 对于DDI Matrix数据集需要做对称处理，这里这样处理同时还考虑了tail打标签要和head保持一致，因此没有用tails[tail]
                if num==3:
                    kg[drug_name_id[head]].append((drug_name_id[tail], relations[relation]))
                    kg[drug_name_id[tail]].append((drug_name_id[head], relations[relation]))
                else:
                    kg[drug_name_id[head]].append((tails[tail], relations[relation]))
        
        return KGInfo(kg=kg, tail_len=len(tails), relation_len=len(relations))

    def getFeatureVec(self):
        def find_dif(raw_matrix):
            sim_matrix4 = np.zeros((572, 572), dtype=float)
            for i in range(572):
                for j in range(572):
                    for k in range(20):
                        if i==j:
                            sim_matrix4[i,j] = 0
                            break
                        else:
                            if raw_matrix[i,k] == raw_matrix[j,k]:
                                sim_matrix4[i, j] += 1
                            else:
                                sim_matrix4[i, j] += 0
            return sim_matrix4
        def Jaccard(matrix):
            matrix = np.mat(matrix)

            numerator = matrix * matrix.T

            denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T

            return numerator / denominator
        dataset={}
        dataset["dataset1"],dataset["dataset2"],dataset["dataset3"],dataset["dataset4"] = self.dataset1_kg.kg, self.dataset2_kg.kg, self.dataset3_kg.kg, self.dataset4_kg.kg

        temp_kg = [defaultdict(list) for i in range(4)]
        for p,kg in enumerate(dataset):
            for i in dataset[kg].keys():
                for j in dataset[kg][i]:
                    temp_kg[p][i].append(j[0])
        # temp_kg[0] like: [{drug_id1: [tail_id1, tail_id2, ...], ...}, ...]

        
        # feature_matrix3 = np.zeros((572, 572), dtype=float)
        feature_matrix1 = np.zeros((572, self.dataset1_kg.tail_len), dtype=float)
        feature_matrix2 = np.zeros((572, self.dataset2_kg.tail_len), dtype=float)
        feature_matrix3 = np.zeros((572, self.dataset4_kg.tail_len), dtype=float)
        feature_matrix4 = np.zeros((572, 572), dtype=float)

        # db.log(dataset4_kg)
        for i in self.dataset4_kg.kg.keys():
            for p,v in self.dataset4_kg.kg[i]:
                # i: head, p: tail, v: relation; 分子结构矩阵
                feature_matrix3[i][p] = v
                
        # 每行代表一个head对应和哪些tail连同
        for i in temp_kg[0].keys():
            for j in temp_kg[0][i]:
                feature_matrix1[i][j] = 1

        for i in temp_kg[1].keys():
            for j in temp_kg[1][i]:
                feature_matrix2[i][j] = 1

        for i in temp_kg[2].keys():
            for j in temp_kg[2][i]:
                feature_matrix4[i][j] = 1

        # 计算各行之间的相似度，并得到相似度矩阵（对称矩阵）
        drug_sim1 = np.array(Jaccard(feature_matrix1))
        # feature_matrix2 = np.mat(feature_matrix2)
        drug_sim2 = np.array(Jaccard(feature_matrix2))
        # feature_matrix4 = np.mat(feature_matrix4)
        
        # ddi matrix jaccard feature
        drug_sim4 = np.array(Jaccard(feature_matrix4))

        drug_sim3 = np.array(find_dif(feature_matrix3))
        
        vector = np.zeros((len(np.array(self.df_drug['name']).tolist()), 0), dtype=float)  #vector=[]
        
        
        vector = np.hstack((vector, drug_sim1))
        vector = np.hstack((vector, drug_sim2))
        vector = np.hstack((vector, drug_sim3))
        
        return vector
    
    def prepare(self, df_drug, mechanism, action, drugA, drugB):
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

        # vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)  #vector=[]
        # for i in feature_list:
        #     # TODO: need to update
        #     #vector = np.hstack((vector, feature_vector(i, df_drug, vector_size)))#1258*1258
        #     tempvec=self.feature_vector(i, df_drug)
        #     vector = np.hstack((vector,tempvec))
        
        vector = self.getFeatureVec()
        
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