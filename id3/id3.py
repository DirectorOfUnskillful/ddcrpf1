import math
import csv

def load_csv(filename):
    lines=csv.reader(open(filename,"r"))
    dataset = list(lines)
    headers = dataset.pop(0)
    return dataset,headers

class Node:
    def __init__ (self,attribute):
        self.attribute=attribute
        self.children=[]
        self.answer=""


def subtables(data,col,delete):  #col is basically a column header
    dic={}
    coldata=[row[col] for row in data]
    
    attr=list(set(coldata))  #set returns only unique values in coldata
    counts=[0]*len(attr)     #create empty list for every unique value
    r=len(data)     #no of rows
    c=len(data[0])  #no of columns in each row
    for x in range(len(attr)):   #no of unique values in the "col" column 
        for y in range(r):
            if data[y][col]==attr[x]:
                counts[x]+=1    
    for x in range(len(attr)):
        dic[attr[x]]=[[0 for i in range(c)] for j in range(counts[x])] #initialing the dictionary items
        pos=0
        for y in range(r):
            if data[y][col]==attr[x]:
                if delete:
                    del data[y][col]   #removing tat particular column (upper in the tree/parent)
                dic[attr[x]][pos]=data[y] #all rows for each unique value
                pos+=1
    return attr,dic   #attr is a list, dic is a set

def entropy(S):
    attr=list(set(S))  #S will basically have last column data(not necessarily of all rows)
    if len(attr)==1:
        return 0    #if there is either only yes/ only no =>entrop is 0
    counts=[0,0]
    for i in range(2):
        counts[i]=sum([1 for x in S if attr[i]==x])/(len(S)*1.0) #find no of yes and no of no
    sums=0
    for cnt in counts:
        sums+=-1*cnt*math.log(cnt,2)  #base 2(second parameter)
    return sums

def compute_gain(data,col): #col  is column-header
    attr,dic = subtables(data,col,delete=False) #here no deletion, we just calculate gain
    total_size=len(data)  # |S| value in formula
    entropies=[0]*len(attr) #entropies of each value
    ratio=[0]*len(attr)   # to maintain |Sv|/|S| values
    total_entropy=entropy([row[-1] for row in data])
    
    for x in range(len(attr)):
        ratio[x]=len(dic[attr[x]])/(total_size*1.0) #len of dic=> |Sv| value
        entropies[x]=entropy([row[-1] for row in dic[attr[x]]])

        total_entropy-=ratio[x]*entropies[x] #acc to formula
    return total_entropy

def build_tree(data,features):
    lastcol=[row[-1] for row in data]
    if(len(set(lastcol)))==1:  #if last column contains either only "yes" or only "no"
        node=Node("")          #we are not building the tree further(so no attribute)
        node.answer=lastcol[0] #it'll be either yes/no
        return node
    n=len(data[0])-1   #-1 boz we dont need the last column values
    gains=[0]*n        # gain is initialized to be 0 for all attributes
    for col in range(n):   
        gains[col]=compute_gain(data,col)  #compute gain of each attribute
    
    split=gains.index(max(gains))          # split will have the index of attribute with "highest gain"
    node=Node(features[split])             # features list will have attribute headings(col names)
                                           # so now we create a subtree (node) with that particular attribute
    fea = features[:split]+features[split+1:] 
    attr,dic=subtables(data,split,delete=True)  #attr will have possible values for tat particular attribute
                                                #dic will have all rows for all those attributes(key: values)
    for x in range(len(attr)):                  
        child=build_tree(dic[attr[x]],fea)     #for each value of the attribute
        node.children.append((attr[x],child))  #again build the tree (but fea exclude the one already taken)
    return node

def print_tree(node,level):
    if node.answer!="":              #if its a leaf node
        print(" "*level,node.answer) #just print "level" no of spaces, followed by answer (yes/no)
        return
    print(" "*level,node.attribute)  #attribute in the node
    for value,n in node.children:
        print(" "*(level+1),value)
        print_tree(n,level+2)        # recursive call to the next node (child)

def classify(node,x_test,features): #features: column headers
    if node.answer!="":      #this will be true only for leaf nodes(answer: yes/no)
        print(node.answer)
        return
    pos=features.index(node.attribute) #node.attribute will have the col header
    for value, n in node.children:   #for every value of that attribute
        if x_test[pos]==value:       # for that particular value go along that value
            classify(n,x_test,features) #go deeper in the tree


dataset,features=load_csv("traintennis.csv")
#lastcol=[row[-1] for row in dataset]
node1=build_tree(dataset,features)
print("The decision tree for the dataset using ID3 algorithm is")
print_tree(node1,0)

testdata,features=load_csv("testtennis.csv")
for xtest in testdata:   #xtest is each row in testdata
    print("The test instance:",xtest)
    print("The label for test instance:",end=" ")
    classify(node1,xtest,features)