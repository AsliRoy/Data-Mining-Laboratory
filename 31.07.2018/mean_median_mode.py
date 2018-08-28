from __future__ import division
#function to find mean 
def get_mean(mylist):
    s = sum(mylist) #finding sum of list
    print('Mean :')
    m = s/len(mylist)
    print(m)
#function to find mode    
def get_mode(l):
    if len(l)>1:
        d = {}
    for value in l:
        if value not in d:
            d[value] = 1 
        else:
            d[value] += 1
    if len(d) == 1:
        print(value)
    else:
        i = 0
        
        for key,value in d.items():
            if i < value:
                i = value
        
        print('Modes:')
        #If more than one mode is present
        for key,value in d.items(): 
            if i == value:
                print(key)  
#function to find median        
def get_median(l):
    #Sorting
    l = sorted(l)
    n = len(l)
    mid = n//2 
    if n%2 == 0:
        return(l[mid] + l[mid-1])/2 
    else:
        return(l[mid])
l = [20, 22, 22, 56, 56,79] 
#Driver function calls
get_mode(l)
get_mean(l)
med = get_median(l)
print('Median:')
print med


