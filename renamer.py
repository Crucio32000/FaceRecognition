import os, sys



if __name__ == "__main__":
    if len(sys.argv) == 3:
        index = sys.argv[2]
    else:
        index = 0
    for (dp, dn, fn) in os.walk("./faces"):
        c_name = dp.split(os.sep)[-1]   # os.sep = "\\" WINDOWS, os.sep = "/" LINUX
        for file in fn:
            new_name = os.path.join(dp, c_name+" - "+str(index)+".jpg")
            while (os.path.isfile(new_name)):
                index += 1
                new_name = os.path.join(dp, c_name+" - "+str(index)+".jpg")
            os.rename(os.path.join(dp,file), new_name)
            index += 1  
