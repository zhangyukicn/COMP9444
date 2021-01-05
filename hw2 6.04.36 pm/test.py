import re

a = "By far the cleanest restaurant I've been to (including kitchen). The owner is a fantastic guy and always very attentive to the needs of the customers. Food is quite good and consistent."
c = "This is the worst place for massages, the owner speaks loudly to him no matter which one wants to pay for massages and relaxing. Yesterday I had to go because someone else kept talking and I complained and the owner does not care. Worst Foot Massage Las Vegas. never return.\
    I have been coming to Pita Jungle for a while now. Lately it has not been the same. The pita that comes with my meal is always rock hard and I have to ask for a replacement. The table and chairs are dirty the last 2 times I have come there. Today when I went to the restroom I walked right out. It was terrible"

b = a.split(" ")
print(b)

new_list = []
cop = re.compile("[^a-zA-Z\s\d]")
for i in c :
    i = cop.sub(' ', i)
    if (len(i) > 1):
        new_list.append(i)

print(new_list)
