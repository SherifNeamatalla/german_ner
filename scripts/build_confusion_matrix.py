import os
import sys

tags = set()
counts = {}
for l in sys.stdin:
    if len(l.strip())==0:
        continue
    ls = l.strip().split()
    gold = ls[-1].replace("B-","I-")
    pred = ls[-2].replace("B-","I-")
    if not gold in tags:
        tags.add(gold)
    if not pred in tags:
        tags.add(pred)
    counts[gold+pred]=counts.get(gold+pred,0)+1
#print(tags)
for t2 in sorted(tags):
    l+="\t"+t2
print (l)
for t1 in sorted(tags):
    l = ""
    l=t1
    for t2 in sorted(tags):
        if t1+t2 not in counts:
            l+="\t0"
        else:
            l+="\t"+str(counts[t1+t2])
    print (l)
