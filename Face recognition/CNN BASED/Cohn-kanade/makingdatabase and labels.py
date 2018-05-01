import sys
import os
import pickle
from PIL import Image

path1 = 'F:\\database\\cohn-kanade-images'

path2 = 'F:\\database\\cohn-kanade-images_resized'

listing = os.listdir(path1)
num_classes = len(listing)
print(num_classes)#no_of_classes
y = []
img_rows = 90
img_cols = 90

for folder in listing :
    path = path1 + "\\" + folder
    listing2 = os.listdir(path)
    for files in listing2:
        path3 = path + "\\" + files
        listing3 = os.listdir(path3)
        for file in listing3:
            #im = Image.open(path3 + '\\' + file)
            #img = im.resize((img_rows, img_cols))
            #gray = img.convert('L')
            # need to do some more processing here
            #gray.save(path2 + '\\' + file, "JPEG")
            y.append(folder)

imlist = os.listdir(path2)
print((imlist))
print(y)
#y is the class list


with open('outfile', 'wb') as fp:
    pickle.dump(y, fp)

#storing class information in a file name outfile in current location