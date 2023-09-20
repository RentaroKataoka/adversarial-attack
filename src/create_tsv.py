import csv
import os
dirname = "data/GoogleFonts/val_reg"
alphabet_list = [chr(a + 65) for a in range(26)]
for alphabet in alphabet_list:
    data1 = []
    for i in range(2000):
        for j in os.listdir(dirname + "/" + alphabet):
            if os.path.exists(dirname + "/" + alphabet + "/" + j + "/{}.png".format(i)):
                data1.append(["{}.png".format(i), int(j)])

    # for i in range(1201, 1501):
    #     for j in os.listdir(dirname + "/" + alphabet):
    #         if os.path.exists(dirname + "/" + alphabet + "/" + j + "/{}.png".format(i)):
    #             data2.append(["{}.png".format(i), int(j)])

    f = open("data/GoogleFonts/val_reg/" + alphabet + "/reg_" + alphabet +  "_val.tsv", "w")
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(data1)
    f.close()
# f = open('normalPGD_reg_dataset/' + alphabet + '/reg_' + alphabet + '_test.tsv', 'w')
# writer = csv.writer(f, delimiter='\t')
# writer.writerows(data2)
# f.close()