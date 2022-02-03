import pandas as pd
import numpy as np
from IPython.display import display, Markdown
from linearmodels import OLS
import matplotlib.pyplot as plt 
import random


def gradient(c1, c2, porc):
    c1=hex_to_rgb(c1)
    c2=hex_to_rgb(c2)
    nv=[]
    i=0
    while i<3:
        nv.append(c1[i]*porc+c2[i]*(1-porc))
        i+=1
    return nv
    
    
def maxi_gradient(colors, porc):
    i=0
    fragmentos=(len(colors)-1)
    while i<=porc:
        if porc>=i and porc<=(i+(1/fragmentos)):
            i1=int(i*fragmentos)
            i2=int(i*fragmentos)+1
            return gradient(colors[i1], colors[i2], 1-(porc-i)/(1/fragmentos))
        i+=1/fragmentos
    return (7/0)


def getcolors_ui(col=None, ncol=0):
    # import urllib library
    from urllib.request import urlopen

    # import json
    import json
    # store the URL in url as 
    # parameter for urlopen
    url = "https://raw.githubusercontent.com/ghosh/uiGradients/master/gradients.json"

    # store the response of URL
    response = urlopen(url)

    # storing the JSON response 
    # from url in data
    colori = json.loads(response.read())
    
    if col==None:
        colori2=[]
        if ncol>0:
            for s in colori:
                if len(s["colors"])==ncol: colori2.append(s)
            sel=random.choice(colori2)
        else: sel=random.choice(colori)
        print("\""+sel["name"]+"\"")
        return sel["colors"]
    else:
        for k in colori:
            if k["name"]==col:
                return k["colors"]
        return np.nan

    
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16)/255.0 for i in range(0, lv, lv // 3))


def savereg(listaregs,reg, name="", depvar="", panel=False, panel_r=None, logit=False, indvars=None, conditional_logit=False):
    nd={}
    nd["name"]=name
    nd["Dep. var"]=depvar
    nd["coefs"]={}
    if len(listaregs)>0:
        for k in listaregs[-1]["coefs"]:
            nd["coefs"][k]=(np.nan, np.nan, np.nan)
    if indvars==None: indvars=reg.params.keys()
    for k in list(indvars):
        c=reg.params[k]
        pval=reg.pvalues[k]
        if (logit or conditional_logit) : se=reg.bse[k]
        else: se=reg.std_errors[k]
        nd["coefs"][k]=(c, se, pval)
    if logit: nd["rsq"]=reg.prsquared
    elif panel: 
        if panel_r==None: nd["rsq"]=reg.rsquared_within
        if panel_r=="inclusive": nd["rsq"]=reg.rsquared_inclusive
    elif conditional_logit: nd["rsq"]=0
    else: nd["rsq"]=reg.rsquared
    nd["N"]=reg.nobs
    for p in range(len(listaregs)):
        for k in nd["coefs"]:
            if k not in listaregs[p]["coefs"]:
                listaregs[p]["coefs"][k]=(np.nan, np.nan, np.nan)
    listaregs.append(nd)
    return listaregs

def euclidean(a, b):
    tot=np.sum([(aa-bb)**2 for aa, bb in zip(a, b)])**0.5
    return tot
    
def euclidean_one(aa):
    a, b = aa
    tot=np.sum([(aa-bb)**2 for aa, bb in zip(a, b)])**0.5
    return tot

def printregs(listaregs, dig_coef=4, dropcols=[], rsq_dig=3):
    finals="|.|"
    for k in listaregs:
        finals+=k["name"]+"|"
        
    finals=finals+"\n|---|"
    for k in listaregs:
        finals+=("---|")
        
    finals+="\n|<b>Dependent variable|"
    for k in listaregs:
        finals+="<b>"+k["Dep. var"]+"|"
    coefs=listaregs[0]["coefs"]
    
    for k in coefs:
        if k not in dropcols:
            finals=finals+"\n|"+k+"|"
            for p in listaregs:
                stars=""
                if np.isnan(p["coefs"][k][0]):
                    finals=finals+"-|"
                else:
                    for vali in [0.1, 0.05, 0.01]:
                        if vali>p["coefs"][k][2]: stars+="*"
                    digits=dig_coef-len(str(int(p["coefs"][k][0])))
                    if digits >0:
                        finals=finals+str(round(p["coefs"][k][0],digits))+stars+"|"
                    else:
                        finals=finals+str(int(p["coefs"][k][0]))+stars+"|"
            finals=finals+"\n|"+"SE|"
            for p in listaregs:
                if np.isnan(p["coefs"][k][1]):
                    finals=finals+"-|"
                else:
                    digits=dig_coef-len(str(int(p["coefs"][k][1])))
                    if digits >0:
                        finals=finals+"("+str(round(p["coefs"][k][1],digits))+")|"
                    else:
                        finals=finals+"("+str(int(p["coefs"][k][1]))+")|"
    finals=finals+"\n|---|"
    for k in listaregs:
        finals+=("---|")
    finals=finals+"\n|<b> rsq|"
    for p in listaregs:
            finals=finals+str(round(p["rsq"],rsq_dig))+"|"
    finals=finals+"\n|<b> N|"
    for p in listaregs:
            finals=finals+str(int(p["N"]))+"|"
    display(Markdown(finals))
        
        
        
def saveregs_tex(listaregs,file, dig_coef=4, names=False, extrarows={}):
    finals="\\begin{tabular}"
    cols="{l|"+"".join(["c" for c in listaregs])+"}"
    finals+=cols+"\n"
    if names:
        finals+=" "
        for k in listaregs:
             finals+="& \textbf{"+k["name"]+"}"
        finals+="\\ \n"
    
    finals+=" Dependent variable "
    for k in listaregs:
        finals+="& "+k["Dep. var"]+""
    coefs=listaregs[0]["coefs"]
    finals+="\\\\ \hline "
    for k in coefs:
        finals+=" \\\\ \n "+k.replace("_", "\_")+" "
        for p in listaregs:
            stars=""
            if np.isnan(p["coefs"][k][0]):
                finals+=" & "
            else:
                for vali in [0.1, 0.05, 0.001]:
                    if vali>p["coefs"][k][2]: stars+="*"
                digits=dig_coef-len(str(int(p["coefs"][k][0])))
                if digits >0:
                    finals=finals+" & "+str(round(p["coefs"][k][0],3))+stars
                else:
                    finals=finals+" & "+str(int(p["coefs"][k][0]))+stars
        finals+="\\\\ \n  "
        for p in listaregs:
            if np.isnan(p["coefs"][k][1]):
                finals+=" & "
            else:
                digits=3-len(str(int(p["coefs"][k][1])))
                if digits >0:
                    finals+="& ("+str(round(p["coefs"][k][1],3))+") "
                else:
                    finals+="& ("+str(int(p["coefs"][k][1]))+") "
    finals+="\\\\ \n \hline "
    finals+=" rsq "
    for p in listaregs:
            finals+=" & "+str(round(p["rsq"],3))
            
    finals=finals+"\\\\ \n N"
    for p in listaregs:
            finals+=" & "+str(int(p["N"]))+""
            
    for key in extrarows:
        finals=finals+"\\\\ \n "+str(key)
        for val in extrarows[key]:
                finals+=" & "+str(val)+""
    finals+="\end{tabular}"
    with open(file, "w") as fil: 
        fil.write(finals)
        

def lolipops(allregs, title, tit="Baseline", dicton2={}, scalev={}):
    points={}
    colors={}
    varsin={}
    plt.figure(figsize=(10,4))

    dicton={"log_distance_noff_1":"Log distance Non-FF (km)", 
           "log_distance_ff_1":"Log distance FF (km)", 
            "log_good_options":"Log Non-FF alternatives (<500m)",
            "log_ff_options":"Log FF alternatives (<500m)", 
            "log_closer_alts":"Log Number of Non-FF closer than closest FF branch"
           }

    for k in allregs:
        if k['Dep. var']!='Distance to lunch given outingfdgfsgh':
            if k['Dep. var'] not in points: 
                points[k['Dep. var']]=len(points)
            for co in k["coefs"]:
                if not (co in ["constant", "logMedianincome", "is.fast.food2before_rmv"]):
                    coef=k["coefs"][co][0]
                    stderror=k["coefs"][co][1]
                    scale=scalev.get(k['Dep. var'], 1)
                    #print(k['Dep. var'])
                    #print(scale)
                    co={"log_distance_noff":"log_distance_noff_1", 
                   "log_distance_ff":"log_distance_ff_1",
                       "log_options":"log_good_options", 
                       "log_FF_options":"log_ff_options"}.get(co, co)
                    if not np.isnan(coef):
                        flag=1
                        if co not in varsin: 
                            varsin[co]=len(varsin)/10.0
                            colors[co]=(random.random(), random.random(), random.random())
                            flag=0
                        valu=0
                        color=colors[co]
                        x=points[k['Dep. var']]+varsin[co]
                        if flag==1: 
                            plt.scatter([x], [coef*scale], s=15, color=color)
                        else: 
                            plt.scatter([x], [coef*scale], s=15, color=color, label=dicton.get(co, co))
                        plt.scatter([x], [coef*scale], s=15,  color=[c+(1-c)*0.8 for c in color], zorder=100)
                        plt.scatter([x], [coef*scale], s=5,  color=[c for c in color], zorder=110)
                        plt.plot([x, x], [coef*scale-scale*stderror*1.96, coef*scale+scale*stderror*1.96], color=color, lw=5)

    plt.legend(bbox_to_anchor=(1,1))
    labs=[k for k in points]
    labs=[dicton2.get(k,k) for k in labs]
    print(labs)
    plt.xticks([points[k]+0.3 for k in points], labs, rotation=15) 
    plt.plot([-10, 10],[0,0], color="gray", linestyle="--")
    plt.xlim([-0.12, len(points)-0.12])
    plt.savefig("lollipop_"+tit+".pdf", bbox_inches='tight')
    plt.show()
    
    
    
        
def plotcoefs(listaregs, drop=["constant"], xcor=10, savefig=""):
    plt.figure(figsize=(xcor,5))
    numregs=len(listaregs)
    r=listaregs[0]
    xval={}
    i=0
    vals=[]
    xvals=[]
    for k in r["coefs"]:
        if k not in drop:
            xval[k]=i
            vals.append(i+numregs/(2*(numregs+1)))
            xvals.append(k)
            i+=1
            
    j=0
    for r in listaregs:
        color=(random.random(), random.random(), random.random())
        flag=1
        for c in r["coefs"]:
            if c not in drop:
                coef=r["coefs"][c][0]
                se=r["coefs"][c][1]
                if not np.isnan(coef):
                    if flag==0: 
                        plt.bar([xval[c]+j/(numregs+1)], coef, width=1/(numregs+1), color=color)
                    else:
                        plt.bar([xval[c]+j/(numregs+1)], coef, width=1/(numregs+1), color=color, label=r["name"])
                        flag=0
                    plt.plot([xval[c]+j/(numregs+1), xval[c]+j/(numregs+1)], [coef-se*1.96, coef+se*1.96], linewidth=3, color="black")
                    
        j+=1
    plt.xticks(vals, xvals, rotation=45)
    plt.title("Size of coefficients")
    plt.legend()
    if savefig!="": plt.savefig(savefig)
    plt.show()
    
    
def plotTreatmentEffect(listaregs, effects=["treatment"], xcor=10, labels=["control","treatment"], constant_name="Constant", title="Efecto de tutorías" ):
    plt.figure(figsize=(xcor,5))
    colors=[(random.random(), random.random(), random.random())]
    j=1
    for e in effects:
        color2=[]
        for c in color:
            color2.append(c+(1-c)*j/(len(effects)+2))
        colors.append(color2)
        j+=1

    index=0
    xlabels=[]
    for r in listaregs:
        
        constant=r["coefs"][constant_name][0]
        if index==0:
            plt.bar([index], constant, width=1/3, color=color, label=labels[0])
        else: plt.bar([index], constant, width=1/3, color=color)
        j=0
        for e in effects:
            treatment=constant+r["coefs"][e][0]
            treatment_se=r["coefs"][e][1]
        
            if index==0: plt.bar([index+1/(len(effects)+2)], treatment, width=1/(len(effects)+1), color=colors[j], label=labels[j] )
            else: plt.bar([index+1/(len(effects)+2)], treatment, width=1/(len(effects)+1), color=colors[j])
        
            plt.plot([index+1/3, index+1/3], [treatment-treatment_se*1.96, treatment+treatment_se*1.96], linewidth=3, color="black")
        index+=1
        xlabels.append(r["name"])
    plt.xticks(range(len(listaregs)), xlabels, rotation=45)
    plt.title(title)
    plt.legend()
    plt.show()
        
    
    
    
    
    
    
