import csv
a=[]
print("/n The Given Training Data Set")
with open('p2.csv') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        a.append(row)
        print(row)
num_attributes = len(a[0])-1
print("\n The initial value of hypothesis: ") 
S=['0']*num_attributes
G=['?']*num_attributes 
print("\n The most specific hypothesis S0 : [0,0,0,0,0,0]")
print("\n The most general hypothesis G0 : [?,?,?,?,?,?]")
for j in range(0,num_attributes): 
          S[j]=a[0][j];
print("\n Candidate Elimination algorithm Hypotheses Version Space Computation\n")
temp=[]
for i in range(0,len(a)):
    if a[i][num_attributes]=='Yes':
        for j in range(0,num_attributes):
            if a[i][j]!=S[j]:
                S[j]='?'
        for j in range(0,num_attributes): 
            for k in range(1, len(temp)):
                if temp[k][j]!='?' and temp[k][j]!=S[j]: 
                    del temp[k]

        print("------------------------------------------------------------------------------------------------------------------------")
        print( "For Training Example No :{0} the hypothesis is S{0} ".format(i+1),S) 
        if (len(temp)==0): 
            print(" For Training Example No : {0} the hypothesis is G{0} ".format(i+1),G)
        else:
            print( "For Positive Training Example No : {0} the hypothesis is G{0}".format(i+1), temp) 
    if a[i][num_attributes]=="No":
        for j in range(0,num_attributes):
            if S[j]!= a[i][j] and S[j]!= '?':
                G[j]=S[j]
                temp.append(G)
                G=['?'] * num_attributes
        print("------------------------------------------------------------------------------------------------------------------------")
        print("For Training Example No :{0} the hypothesis is S{0} ".format(i+1),S)
        print(" For Training Example No :{0} the hypothesis is G{0}".format(i+1),temp)         
        
        
    """
    value of a should be 
    a = [['sunny', 'warm', 'normal', 'strong', 'warm', 'same', 'Yes'],
['sunny', 'warm', 'high', 'strong', 'warm', 'same', 'Yes'],
['rainy', 'cold', 'high', 'strong', 'warm', 'change', 'No'],
['sunny', 'warm', 'high', 'strong', 'cool', 'change', 'Yes']]
    
    
    
    """