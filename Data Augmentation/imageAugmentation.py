from PIL import Image
import random

def noisegen(scale):
    f=random.gauss(0.0,scale)
    ff=random.randint(0,1)
    if ff==0:
        return int(abs(f))
    else:
        return -int(abs(f))

def clip(out):
    if out>255:
        return 255
    if out < 0:
        return 0
    return out

def addnoise(r,scale):
    out=r+noisegen(scale)
    return clip(out)

def randomNoise(src,tgt,scale):
    im = Image.open(src)
    pm = im.load()
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            r,g,b,l = pm[i,j]
            pm[i,j] = (addnoise(r,scale),addnoise(g,scale),addnoise(b,scale),addnoise(l,scale))
    im.save(tgt)
    im.close()

def darkenlighten(src,tgt,scale):
    im = Image.open(src)
    pm = im.load()
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            r,g,b,l = pm[i,j]
            pm[i,j] =(clip(r+scale),clip(g+scale),clip(b+scale),l)
    im.save(tgt)
    im.close()

def blackbox(src,tgt,scale):
    im = Image.open(src)
    pm = im.load()
    a1=random.randint(0,im.size[0]-scale)
    a2=random.randint(0,im.size[1]-scale)
    for i in range(a1,a1+scale):
        for j in range(a2,a2+scale):
            pm[i,j] =(0,0,0,255)
    im.save(tgt)
    im.close()
    
"""
randomNoise('a.png','noise5.png',5)
randomNoise('a.png','noise10.png',10)
randomNoise('a.png','noise20.png',20)
randomNoise('a.png','noise40.png',40)
darkenlighten('a.png','lighten80.png',80)
darkenlighten('a.png','lighten40.png',40)
darkenlighten('a.png','lighten20.png',20)
darkenlighten('a.png','daeken20.png',-20)
darkenlighten('a.png','darken40.png',-40)
darkenlighten('a.png','darken80.png',-80)
"""
blackbox('a.png','blackbox10a.png',10)
blackbox('a.png','blackbox10b.png',10)
blackbox('a.png','blackbox10c.png',10)
blackbox('a.png','blackbox20a.png',20)
blackbox('a.png','blackbox20b.png',20)
blackbox('a.png','blackbox20c.png',20)
