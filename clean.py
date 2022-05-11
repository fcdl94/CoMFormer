import os
base = "output_inc"

for sett_dir in os.listdir(base):
    for i, exp in enumerate(os.listdir(base + "/" + sett_dir)):
        print(exp)
        for j, step in enumerate(os.listdir(base + "/" + sett_dir + "/" + exp)):
            print(step)
            if step != "step0":
                for k, file in enumerate(os.listdir(base + "/" + sett_dir + "/" + exp + "/" + step)):
                    if "tmp" in file:
                        os.remove(base + "/" + sett_dir + "/" + exp + "/" + step + "/" + file)
                        print(file)