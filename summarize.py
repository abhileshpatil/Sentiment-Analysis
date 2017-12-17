import os.path
textfiles=['users.txt','tweets.txt','cluster.txt','classification.txt']
s= open("summary.txt","w+",encoding="utf-8")
for i in range(0,len(textfiles)):
    with open(textfiles[i],encoding="utf-8") as f:
    # content = f.readlines()
        if(i==0):
            lines = len(f.readlines())
            s.write("Number of users collected: "+str(lines))
            s.write("\n")
        if(i==1):
            lines = len(f.readlines())
            s.write("Number of messages collected: "+str(lines))
            s.write("\n")
        if(i==2):
            content=f.readlines()
            for x in content:
                s.write(x)
            s.write("\n")
        if(i==3):
            lines=f.readlines()
            s.write("Number of instances per class found:\n")
            j=0
            for x in lines:
                if(j<2):
                    j=j+1
                    s.write("     "+x)
                if(j>2):
                    if(j==3):
                        s.write("One example from each class:\n")
                    j=j+1
                    s.write("     "+x)
                if(j>1):
                    j=j+1
