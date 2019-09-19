import numpy as np
import keras as K
from keras.models import load_model
import keras.backend as K
import sys, getopt
import re
import pandas as pd
import os

_MOL_WEIGHTS = {
    '-': 0.0,
    'A': 71.03711,
    'C': 103.00919 + 57.02146,  # Add fixed CME modification to the Cys mass.
    'E': 129.04259,
    'D': 115.02694,
    'G': 57.02146,
    'F': 147.06841,
    'I': 113.08406,
    'H': 137.05891,
    'K': 128.09496,
    'M': 131.04049,
    'L': 113.08406,
    'N': 114.04293,
    'Q': 128.05858,
    'P': 97.05276,
    'S': 87.03203,
    'R': 156.10111,
    'T': 101.04768,
    'W': 186.07931,
    'V': 99.06841,
    'Y': 163.06333,
    'M(ox)': 15.994915,
    'P(ox)': 15.994915,
    'W(ox)': 15.994915,
    'S(ph)': 79.966331,
    'T(ph)': 79.966331,
    'Y(ph)': 79.966331,
    'H(hiasp)': -22.031969,
    'H(higlu)': -25.016319,
    '(gl)E': -18.010565,
    '(gl)Q': -17.026549,
    'R(ar)': -42.021798,
    '(gl)Q(de)': 67,
    'C(ca)': 57.0214637236,
    'N(de)': 0.9840155848,
    'Q(de)': 0.9840155848,
    'M(di)': 31.9898292442,
    'W(di)': 31.9898292442}

#_RES = re.compile(r'(\((\w+)\))[A-Z]?')
_RESIDUE = re.compile(r'(\(gl\))Q(\(de\))|[A-Z](\((\w+)\))?|(\((\w+)\))[A-Z]?')
mol_weights = pd.Series(_MOL_WEIGHTS)
alphabet = [k for k in mol_weights.keys()]
one_hot_encoding = pd.get_dummies(alphabet).astype(int).to_dict(orient='list')
one_hot_encoding_re = dict((one_hot_encoding[i].index(1), i) for i in one_hot_encoding)


class createWindowData(object):
    _daa = dict()
    _sequence = ""
    _yIons = []
    _bIons = []
    _yIonsNorm = []
    _bIonsNorm = []
    _matrix = []

    def __init__(self, sequence, ions, intensities):
        self._sequence = sequence
        # Creating Amino Matrix
        seq = []
        for residue in re.finditer(_RESIDUE, self._sequence):
            # print(residue.group())
            sequence_list = residue.group()
            seq.append(sequence_list)

        self.yIons = [0] * len(seq)
        self.bIons = [0] * len(seq)
        lstions = ions.split(";")
        lstintensities = intensities.split(";")

        yIonsreg = re.compile('^y[0-9]+$')
        bIonsreg = re.compile('^b[0-9]+$')

        self._yIons = [0] * len(seq)
        self._bIons = [0] * len(seq)

        for index, ion in enumerate(lstions):
            if yIonsreg.match(ion):
                self._yIons[int(ion.split("y")[1])] = float(lstintensities[index])
            if bIonsreg.match(ion):
                self._bIons[int(ion.split("b")[1])] = float(lstintensities[index])
        list_b_ions = self._bIons[1:]
        list_b_ions.append(0)
        list_y_ions = self._yIons[::-1]
        self._bIons = list_b_ions
        self._yIons = list_y_ions
        self._matrix = []

    def GenerateMatrix(self, size):
        dictFeature = dict()
        order = []
        size = int(size)
        for index in range(((size) // 2) - 1):
            key = "Sj-" + str(index + 1)
            dictFeature[key] = ["?"] * 20
            order.append((index + 1) * -1)
        ##############
        dictFeature["Sj"] = ["?"] * 20
        ##############
        for index in range((size) // 2):
            key = "Sj+" + str(index + 1)
            dictFeature[key] = ["?"] * 20
            order.append(index + 1)
            ##########################
        dictFeature["Sj+1"] = ["?"] * 20
        dictFeature["Sj+2"] = ["?"] * 20
        ############################
        dictFeature["S1"] = ["?"] * 20
        dictFeature["SN"] = ["?"] * 20
        dictFeature["length"] = "?"
        dictFeature["Dist-1"] = "?"
        dictFeature["Dist-N"] = "?"
        Sj_Positive = re.compile('^Sj\+')
        Sj_Negative = re.compile('^Sj-')
        order.append(0)
        order = sorted(order)
        sequence = []
        for residue in re.finditer(_RESIDUE, self._sequence):
            sequence_list = residue.group()
            sequence.append(sequence_list)
        for index, aa in enumerate(sequence):
            for k in dictFeature:
                val = 0
                if Sj_Negative.match(k):
                    val = int(k[3:]) * -1

                if Sj_Positive.match(k):
                    val = int(k[3:])

                if (index + val) < 0 or (index + val) >= len(sequence):
                    dictFeature[k] = one_hot_encoding["-"]
                else:
                    dictFeature[k] = one_hot_encoding[sequence[index+val]]
            # dictFeature["Sj"]
            dictFeature["Sj"] = one_hot_encoding[sequence[index]]
            # dictFeature["S1"]
            dictFeature["S1"] = one_hot_encoding[sequence[0]]
            # dictFeature["SN"]
            dictFeature["SN"] = one_hot_encoding[sequence[len(sequence)-1]]
            # dictFeature["length"]
            dictFeature["length"] = len(sequence)
            # dictFeature["Dist-1"]="?"
            dictFeature["Dist-1"] = index
            # dictFeature["Dist-N"]="?"
            dictFeature["Dist-N"] = len(sequence) - index - 1
            col = []
            for i in order:
                if (i < 0):
                    col += list(dictFeature["Sj" + str(i)])
                if (i == 0):
                    col += list(dictFeature["Sj"])
                if (i > 0):
                    col += list(dictFeature["Sj+" + str(i)])

            col += list(dictFeature["S1"]) + list(dictFeature["SN"]) + [dictFeature["length"]] + [dictFeature["Dist-1"] + 1] + [dictFeature["Dist-N"]]
            res = [int(i) for i in col]
            self._matrix.append(res)

    def nomalizeIones(self):
        self._yIonsNorm = [float(i) / max(self._yIons + self._bIons) for i in self._yIons]
        self._bIonsNorm = [float(i) / max(self._yIons + self._bIons) for i in self._bIons]

    def GenerateDataset(self, removeZeros):
        self.nomalizeIones()
        data = {'featureMatrix': [], 'target_Y': [], 'target_B': []}
        for index, line in enumerate(self._matrix):
            if removeZeros == True:
                if (self._yIonsNorm[index] != 0):
                    data['featureMatrix'].append(line)
                    data['target_Y'].append(np.log2(1 + (10000 * (self._yIonsNorm[index]))))

                if (self._bIonsNorm[index] != 0):
                    data['target_B'].append(np.log2(1+(10000*(self._bIonsNorm[index]))))
            else:
                data['featureMatrix'].append(line)
                data['target_Y'].append(np.log2(1 + (10000 * (self._yIonsNorm[index]))))
                # data['featureMatrix_B'].append(line)
                data['target_B'].append(np.log2(1+(10000*(self._bIonsNorm[index]))))
        data['target_Y'] = list(map(float, data['target_Y']))
        data['target_B'] = list(map(float, data['target_B']))
        # Correction Bug
        data['featureMatrix'] = data['featureMatrix'][:-1]
        data['target_Y'] = data['target_Y'][:-1]
        data['target_B'] = data['target_B'][:-1]
        return data


def pearson_correlation(y_true, y_pred):
    return 1


def batch_size_shape(y_true, y_pred):
    return K.shape(y_true)[1]


def TestNN(model_name, feature_matrix):
    model_1 = load_model(model_name, custom_objects={'pearson_correlation': pearson_correlation,
                                                     'batch_size_shape': batch_size_shape})  # load the saved model
    y_score = model_1.predict(feature_matrix, batch_size=32)
    return (y_score)


def mean(listofNumpyArrays):
    return np.array(listofNumpyArrays).mean(axis=0)

def Predict(model_file, lstsequences,fragmentation_method, charge, outputfile):
    # General variables, depending on the fragmentation method the length of the sequences and the SVR models are differents.
    suffix = ".h5"
    if (fragmentation_method == "CID" and charge <= "2"):
        max_length = 50
        max_D1 = 50
        max_DN = 50
        Models = [
            os.path.join(model_file,"model_cid2"+suffix)
        ]
    elif (fragmentation_method == "CID" and charge >= "3"):
        max_length = 50
        max_D1 = 50
        max_DN = 50
        Models = [
             os.path.join(model_file,"model_cid3"+suffix)
        ]
    elif (fragmentation_method == "HCD" and charge <= "2"):
        max_length = 50
        max_D1 = 50
        max_DN = 50
        Models = [
            os.path.join(model_file,"model_hcd2"+suffix)
        ]
    elif (fragmentation_method == "HCD" and charge >= "3"):
        max_length = 50
        max_D1 = 50
        max_DN = 50
        Models = [
            os.path.join(model_file, "model_hcd3" + suffix)
        ]

    feature_matrix = []
    result_matrix = []
    target_matrix = []
    for seq in lstsequences:
        myTrainingMatrix = createWindowData(seq, "b1","1")  # (seq,"b1","1") -> in order to create a window.. it is needed to provide at least One Ion, does not matter is not real (We don't need intensities)
        myTrainingMatrix.GenerateMatrix(24)
        d = myTrainingMatrix.GenerateDataset(removeZeros=False)
        dm = d['featureMatrix']
        dmy = d['target_Y']
        dmb = d['target_B']
        zipped = zip(dmy, dmb)
        list_c = list(zipped)
        for ix, ra in enumerate(list_c):
            target_matrix.append(ra)
        for index, r in enumerate(dm):
            res = [seq, fragmentation_method, charge, "b+" + str(r[-2]), "y+" + str(r[-1])]
            r[-3] = r[-3] * 1.0/max_length
            r[-2] = r[-2] * 1.0/max_D1
            r[-1] = r[-1] * 1.0/max_DN
            feature_matrix.append(r)
            result_matrix.append(res)
    m_yb = []


    for m in Models:
        Col_BYIons = TestNN(m, np.array(feature_matrix)).tolist()
        m_yb.append(Col_BYIons)
    for idx, r in enumerate(result_matrix):
        r.append(Col_BYIons[idx][1])
        r.append(Col_BYIons[idx][0])
    dictOutput = dict()

    for row in result_matrix:
        seq1 = []
        k = str(row[0] + "-" + row[1] + "-" + row[2])
        bindex = int(row[3].split("+")[1]) - 1
        yindex = int(row[4].split("+")[1]) - 1
        bintensity = row[5]
        yintensity = row[6]
        for residue in re.finditer(_RESIDUE, row[0]):
            sequence_list = residue.group()
            seq1.append(sequence_list)
        if k not in dictOutput.keys():
            dictOutput[k] = ([0] * (len(seq1) - 1), [0] * (len(seq1) - 1),[0] * (len(seq1) - 1),[0] * (len(seq1) - 1))
            dictOutput[k][0][bindex] = bintensity
            dictOutput[k][1][yindex] = yintensity
        else:
            dictOutput[k][0][bindex] = bintensity
            dictOutput[k][1][yindex] = yintensity

    with open(outputfile, 'a') as file:
        for k in dictOutput:
            listb = ["bXXX_charge1-noloss"] * len(dictOutput[k][0])
            listy = ["yXXX_charge1-noloss"] * len(dictOutput[k][0])
            for idx, val in enumerate(listb):
                listb[idx] = val.replace("XXX", str(idx + 1))
            for idx, val in enumerate(listy):
                listy[idx] = val.replace("XXX", str(idx + 1))
            IntensitiesTypes = ';'.join(map(str, (dictOutput[k][0]) + dictOutput[k][1]))
            aux = "\t".join(k.split("-")) + "\t" + IntensitiesTypes + "\t" + ';'.join(listb + listy)
            file.write(aux)
            file.write('\n')

def main(argv):
    inputfile = ''
    outputfile = ''
    modelpath = ''
    sequenceCol = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:d:s:", ["ifile=", "ofile=","dfile","sequence_col"])
    except getopt.GetoptError:
        print('PredictSequences.exe -i <inputfile> -o <outputfile> -d <modelpath> -s <sequenceCol>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("PredictSequences.exe -i <inputfile> -o <outputfile> -d <modelpath> -s <sequence_col>")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-d", "--dfile"):
            modelpath = arg
        elif opt in ("-s","--sequence_col"):
            sequenceCol = arg

    print('Input file is ', inputfile)
    print('Output file is ', outputfile)
    print('model path is ', modelpath)
    print('sequence column is ', sequenceCol)

    with open(outputfile, "w") as file:
        columns = "Sequence\tFragmentation\tCharge\tFragmentIntensities\tFragmentIons"
        file.write(columns)
        file.write('\n')
    print("\t===================================================================")
    print("\t=====  wiNNer Peptide Intensities Prediction - Window 24 =========")
    print("\t===================================================================")
    print("\n\n>>Predictions starting... It might takes several minutes, please wait")
    read = pd.read_csv(inputfile, delimiter='\t')
    dictSeq = {}
    for index, row in read.iterrows():
        if sequenceCol == "ModifiedSequence":
            k = str(row["ModifiedSequence"])

        elif sequenceCol == "Sequence":
            k = str(row["Sequence"])
        if k not in dictSeq.keys():
            dictSeq[k] = [[], []]
            dictSeq[k][0].append(row["Fragmentation"])
            dictSeq[k][1].append(row["Charge"])
        else:
            dictSeq[k][0].append(row["Fragmentation"])
            dictSeq[k][1].append(row["Charge"])
    print("Loading input...done.")
    for k in dictSeq:
        Fragmentation = "".join(str(x) for x in dictSeq[k][0])
        Charge = "".join(str(x) for x in dictSeq[k][1])
        ListofSequences = [k]
        print("Starting Prediction->", Fragmentation, Charge)
        Predict(modelpath, ListofSequences, Fragmentation, Charge, outputfile)
        print("Done:->", Fragmentation, Charge)

if __name__ == "__main__":
    main(sys.argv[1:])