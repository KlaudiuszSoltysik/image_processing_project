import json

# Load the contents of the two JSON files
with open('data/train.json', 'r') as file:
    train_file = json.load(file)

with open('results.json', 'r') as file:
    result_file = json.load(file)

numerator = 0
denominator = 0
denominator2 = 0

for image_key in train_file:
    print('Train  ' + str(train_file[image_key]))
    print('Result ' + str(result_file[image_key]))
    print()
    for leaf_key in train_file[image_key]:
        numerator += abs(train_file[image_key][leaf_key] - result_file[image_key][leaf_key])
        denominator += train_file[image_key][leaf_key]
        denominator2 += result_file[image_key][leaf_key]

MARPE = 100 / len(train_file) * (numerator / denominator)

print(str(MARPE) + " %")
print('Powinno być: ' + str(denominator))
print('Jest:        ' + str(denominator2))
