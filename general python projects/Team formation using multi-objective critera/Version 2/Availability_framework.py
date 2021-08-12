from itertools import combinations

def multiply_arrays(a1, a2):
    x = []
    for i in range(len(a1)):
        x.append((a1[i])*(a2[i]))
    return x    

def similarity(n, m):
    y = multiply_arrays(n,m)
    total = 0
    for i in range(0,len(y)):
        total += y[i]
    return total    

def for_all(arr):
    keys = ['p1+p2','p1+p3','p1+p4','p1+p5','p1+p6','p2+p3','p2+p4',
            'p2+p5','p2+p6','p3+p4','p3+p5','p3+p6','p4+p5','p4+p6',
            'p5+p6']
    values = []
    for i in range(0,len(arr),2):
        variable1 = arr[i]
        variable2 = arr[i+1]
        values.append(similarity(variable1,variable2))
    return dict(zip(keys,values))

def avg_dic(x:dict):
    val = 0
    for i in range(len(x)):
        val += values[i]
    return val/len(x)

'''def possible_teams():
    if a'''
    
person_1 = [0,0,0,1,1,1,0,0,0,0,0,1,0,0,1]
person_2 = [0,1,0,1,1,0,1,1,1,0,1,0,1,0,0]
person_3 = [1,0,0,1,0,1,0,0,0,1,0,0,1,0,1]
person_4 = [1,1,0,1,1,1,1,0,0,1,0,1,0,1,1]
person_5 = [0,1,0,1,1,0,1,0,0,1,0,1,0,1,1]
person_6 = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1]

persons = [person_1, person_2, person_3, person_4, person_5, person_6]
comb = combinations(persons, 3)
teamss = list(comb)
# convert list of tuples into list using list comprehension
out = [item for t in teamss for item in t]
print(out)
        
print(for_all(out))
