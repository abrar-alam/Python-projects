person_1 = ['python']
person_2 = ['Mpython']
person_3 = ['Mhon']
person_4 = ['M']

team = person_1 + person_2 + person_3 + person_4

performance = 0
ptr = []

for i in range(len(team)):
  target= team[i]
  if target not in ptr:
    ptr.append(target)
    count = 0
    for j in range(len(team)):
      if target == team[j]:
        count += 1
    if count>1:
      performance += count

def normalize(performance):
  return performance/len(team)

print(normalize(performance))
print(team)
