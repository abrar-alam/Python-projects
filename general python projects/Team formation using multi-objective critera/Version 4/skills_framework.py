from collections import Counter

person_1 = ['ML', 'English', 'French']
person_2 = ['NLP', 'French']
person_3 = ['Alchemy', 'NLP']
person_4 = ['ML', 'word', 'NLP']

team = person_1 + person_2 + person_3 + person_4
performance = 0
ptr = []
for i in range(len(team)):
    target = team[i]
    if target not in ptr:
        ptr.append(target)
        count = 0
        for j in range(len(team)):
            if target == team[j]:
                count += 1
        if count > 1:
            performance += count


def normalize(performance):
    return performance/len(team)


print(normalize(performance))
print(team)

ls = [person_1, person_2, person_3, person_4]
team = person_1 + person_2 + person_3 + person_4
performance = 0
ptr = []
for i in range(len(team)):
    target = team[i]
    if target not in ptr:
        ptr.append(target)
        count = 0
        for j in range(len(team)):
            if target == team[j]:
                count += 1
        if count > 1:
            performance += count

"""My added to test stuff"""


def skills_performance() -> float:

    temp = []
    t = 0
    for user in ls:
        temp = temp + user
    counter = Counter(temp)
    if (len(counter) == 0):
        return t
    for key in counter:
        t += (counter[key]-1)
    return t


def normalize(performance):
    return performance/len(team)


print(normalize(performance))
print(performance)
print(skills_performance())
print(team)
