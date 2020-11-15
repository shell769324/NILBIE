
shs=['sphere','cube','cylinder','it']

def splitoff(splitword, words):
    for i in range(0,len(words)):
        if words[i]in splitword:
            if i+1==len(words):
                return (words[0:i],words[i],[])
            return words[0:i],words[i],words[i+1:]
    print(words)
    return None

def fit(temp, sent):
    words=sent.split(' ')
    if temp==1:
        object=words[2:4]
        locs=words[4:]
        return (object,locs)
    elif temp==2:
        object=words[2:4]
        firstrel, w, secondrel=splitoff(['it'],words[4:])
        return (object,firstrel,secondrel)
    elif temp==3:
        object=words[2:4]
        remain=words[4:]
        firstfullrel, w, secondfullrel = splitoff(['and'],remain)
        a,b,c=splitoff(shs,firstfullrel)
        if b=='it':
            obj1=['it']
            desc11=a
            desc12=c
        else:
            obj1=[a[-1],b]
            desc11=a[:-1]
            desc12=c
        #print(splitoff(shs,secondfullrel))
        a, b, c = splitoff(shs, secondfullrel)
        if b == 'it':
            obj2 = ['it']
            desc21 = a
            desc22 = c
        else:
            obj2 = [a[-1], b]
            desc21 = a[:-1]
            desc22 = c
        return (object, obj1,desc11,desc12, obj2,desc21, desc22)


import random

def rewriteobj(obj):
    if obj[0]=='it':
        return obj
    r=random.randint(0,1)
    if r==0:
        return obj
    return [obj[1]]+['that','is']+[obj[0]]

def writebeginning(obj):
    r=random.randint(0,1)
    if r==0:
        return ['Add','a']+obj
    return ['A']+obj+['needs','to','be','added']

def rev(d):
    if len(d)==0:
        return d
    if d[0]=='behind':
        return ['in','front','of']
    if d[0]=='in':
        return ['behind']
    if d[2]=='left':
        return ['on','the','right']
    if d[2]=='right':
        return ['on','the','left']


def writedesc(desc1,obj,desc2,origobj,suchthat,spec,prev,r):
    if suchthat==0:
        prefix=[]
        if spec==1:
            prefix=['such','that']
        if obj[0]=='it':
            obj=prev
        return prefix+['the']+obj+['is']+rev(desc2)+['and']+rev(desc1)+['the']+origobj
    if r==0:
        return desc1+obj+desc2
    if r==1:
        if obj[0]=='it':
            obj=prev
        return desc2+['and']+desc1+obj

def rewrite(fivesents):
    obj,desc=fit(1,fivesents[0])
    #obj=rewriteobj(obj)
    re1=writebeginning(obj)+desc
    prevobj=obj
    obj,desc1,desc2=fit(2,fivesents[1])
    #obj=rewriteobj(obj)
    r = random.randint(0, 1)
    re2=writebeginning(obj)+writedesc(desc1,['it'],desc2,obj,1,0,prevobj,r)
    res=[re1,re2,'','','']
    for i in range(2,5):
        prevobj=obj
        obj, obj1, desc11, desc12, obj2, desc21, desc22 = fit(3,fivesents[i])
        #obj=rewriteobj(obj)
        #obj1=rewriteobj(obj1)
        #obj2=rewriteobj(obj2)
        r=random.randint(0,1)

        r2=random.randint(0,1)
        firstdesc=writedesc(desc11,obj1,desc12,obj,r,1,prevobj,r2)
        seconddesc=writedesc(desc21,obj2,desc22,obj,r,2,prevobj,r2)
        res[i]=writebeginning(obj)+firstdesc+['and']+seconddesc
    return res

filenames=[]
for i in range(0,2000):
    putlen=6-len(str(i))
    bpl=''
    for j in range(putlen):
        bpl += '0'
    filenames.append('text/CLEVR_test_'+bpl+str(i)+'.txt')

for filename in filenames:
    f=open(filename,'r')
    lines=f.readlines()
    for i in range(0,len(lines)):
        if lines[i][-1]=='\n':
            lines[i]=lines[i][:-1]
    outs=rewrite(lines)
    f=open('rephrased_text/rephrased-'+filename,'w+')
    for line in outs:
        f.write(' '.join(line)+'\n')


