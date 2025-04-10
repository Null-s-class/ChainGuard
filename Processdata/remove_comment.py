import re
import os
import uuid


def remove_comment(inputFile, outputFile):
    fdr = open(inputFile, 'r', encoding="utf-8")
    fdw = open(outputFile, 'w', encoding="utf-8")
    _map = {}
    outstring = ''

    line = fdr.readline()
    while line:
        while True:
            m = re.compile('\".*\"', re.S)
            _str = m.search(line)
            if _str is None:
                outstring += line
                break
            key = str(uuid.uuid1())
            outtmp = re.sub(m, key, line, 1)
            line = outtmp
            _map[key] = _str.group(0)
        line = fdr.readline()

    # Remove comments
    outstring = re.sub(r'//.*', ' ', outstring)
    outstring = re.sub(r'/\*.*?\*/', ' ', outstring, flags=re.S)

    # Remove [SEP] at the beginning of lines
    outstring = re.sub(r'^\[SEP\]', '', outstring, flags=re.M)

    for key in _map.keys():
        outstring = outstring.replace(key, _map[key])

    fdw.write(outstring)
    fdw.close()


if __name__ == '__main__':
    original_dir = "../Data/sourcecode/"
    output_dir = "../Data/dataclean/"

    dir = os.listdir(original_dir)
    for i in dir:
        print(i)
        remove_comment(original_dir + i, output_dir + i)
