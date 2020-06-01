import numpy as np

db = open("data.txt","r")
items = db.readlines()

letterList = []
for_make_dictionary = []
for item in items:
    for i in item.split() :
        for_make_dictionary.append(i)
        letterList.append(list(i))

English_letter_lowercase = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
English_letter_uppercase = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
main_list = []
for letList in letterList:
    fake_list = []
    for lt in letList :
        if lt in English_letter_lowercase :
            index = English_letter_lowercase.index(lt)
            fake_list.append(index)
        if lt in English_letter_uppercase :
            index = English_letter_uppercase.index(lt)
            fake_list.append(index)

    main_list.append(fake_list)

print(letterList)
print(main_list)


def vectorize_sequence(sequences , dimension=26):
    results = np.zeros((len(sequences),dimension))
    for i , sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

arra = [[19, 7, 17, 14, 20, 6, 7], [19, 7, 4]]
word_vector = vectorize_sequence(main_list)
print(word_vector[0])

my_dictionary = dict()
k=0
for md in for_make_dictionary :
    my_dictionary[md] = word_vector[k]
    k = k+1

print(my_dictionary)
