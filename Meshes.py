import pennylane as qml

def ReckMesh4Qumode_encode(x):
    qml.Squeezing(x[0], x, wires=0)
    qml.Squeezing(x[1], x, wires=1)
    qml.Squeezing(x[2], x, wires=2)
    qml.Squeezing(x[3], x, wires=3)

    qml.Beamsplitter(x[4], x[5], wires=[2,3])

    qml.Beamsplitter(x[6], x[7], wires=[1,2])

    qml.Beamsplitter(x[8], x[9], wires=[0,1])
    qml.Beamsplitter(x[10], x[11], wires=[2,3])

    qml.Rotation(x[12], wires=0)
    qml.Beamsplitter(x[13], x[14], wires=[1,2])

    qml.Displacement(x[15], x, wires=0)
    qml.Rotation(x[16], wires=1)
    qml.Beamsplitter(x[17], x[18], wires=[2,3])

    qml.Kerr(x[19], wires=0)
    qml.Displacement(x[20], x, wires=1)
    qml.Rotation(x[21], wires=2)
    qml.Rotation(x[22], wires=3)

    qml.Kerr(x[23], wires=1)
    qml.Displacement(x[24], x, wires=2)
    qml.Displacement(x[25], x, wires=3)

    qml.Kerr(x[26], wires=2)
    qml.Kerr(x[27], wires=3)

    #qml.draw()
    #exit()


def ReckMesh4Qumode_layer(v):

    qml.Beamsplitter(v[28], v[29], wires=[2,3])

    qml.Beamsplitter(v[30], v[31], wires=[1,2])

    qml.Beamsplitter(v[32], v[33], wires=[0,1])
    qml.Beamsplitter(v[34], v[35], wires=[2,3])

    qml.Rotation(v[36], wires=0)
    qml.Beamsplitter(v[37], v[38], wires=[1,2])

    qml.Squeezing(v[39], v, wires=0)
    qml.Rotation(v[40], wires=1)
    qml.Beamsplitter(v[41], v[42], wires=[2,3])

    qml.Kerr(v[43], wires=0)
    qml.Squeezing(v[44], v, wires=1)
    qml.Rotation(v[45], wires=2)
    qml.Rotation(v[46], wires=3)

    qml.Kerr(v[47], wires=1)
    qml.Squeezing(v[48], v, wires=2)
    qml.Squeezing(v[49], v, wires=3)

    qml.Kerr(v[50], wires=2)
    qml.Kerr(v[51], wires=3)

    qml.Beamsplitter(v[52], v[53], wires=[2,3])

    qml.Beamsplitter(v[54], v[55], wires=[1,2])

    qml.Beamsplitter(v[56], v[57], wires=[0,1])
    qml.Beamsplitter(v[58], v[59], wires=[2,3])

    qml.Rotation(v[60], wires=0)
    qml.Beamsplitter(v[61], v[62], wires=[1,2])

    qml.Displacement(v[63], v, wires=0)
    qml.Rotation(v[64], wires=1)
    qml.Beamsplitter(v[65], v[66], wires=[2,3])

    qml.Kerr(v[67], wires=0)
    qml.Displacement(v[68], v, wires=1)
    qml.Rotation(v[69], wires=2)
    qml.Rotation(v[70], wires=3)

    qml.Kerr(v[71], wires=1)
    qml.Displacement(v[72], v, wires=2)
    qml.Displacement(v[73], v, wires=3)

    qml.Kerr(v[74], wires=2)
    qml.Kerr(v[75], wires=3)

    #qml.draw()
    #exit()


def ClementsMesh4Qumode_encode(x):
    qml.Squeezing(x[0], x, wires=0)
    qml.Squeezing(x[1], x, wires=1)
    qml.Squeezing(x[2], x, wires=2)
    qml.Squeezing(x[3], x, wires=3)

    qml.Beamsplitter(x[4], x[5], wires=[0,1])
    qml.Beamsplitter(x[6], x[7], wires=[2,3])

    qml.Beamsplitter(x[8], x[9], wires=[1,2])

    qml.Beamsplitter(x[10], x[11], wires=[0,1])
    qml.Beamsplitter(x[12], x[13], wires=[2,3])

    qml.Rotation(x[14], wires=0)
    qml.Beamsplitter(x[15], x[16], wires=[1,2])
    qml.Rotation(x[17], wires=3)

    qml.Displacement(x[18], x, wires=0)
    qml.Rotation(x[19], wires=1)
    qml.Rotation(x[20], wires=2)
    qml.Displacement(x[21], x, wires=3)

    qml.Kerr(x[22], wires=0)
    qml.Displacement(x[23], x, wires=1)
    qml.Displacement(x[24], x, wires=2)
    qml.Kerr(x[25], wires=3)

    qml.Kerr(x[26], wires=1)
    qml.Kerr(x[27], wires=2)


def ClementsMesh4Qumode_layer(v):

    qml.Beamsplitter(v[28], v[29], wires=[0,1])
    qml.Beamsplitter(v[30], v[31], wires=[2,3])

    qml.Beamsplitter(v[32], v[33], wires=[1,2])

    qml.Beamsplitter(v[34], v[35], wires=[0,1])
    qml.Beamsplitter(v[36], v[37], wires=[2,3])

    qml.Rotation(v[38], wires=0)
    qml.Beamsplitter(v[39], v[40], wires=[1,2])
    qml.Rotation(v[41], wires=3)

    qml.Squeezing(v[42], v, wires=0)
    qml.Rotation(v[43], wires=1)
    qml.Rotation(v[44], wires=2)
    qml.Squeezing(v[45], v, wires=3)

    qml.Kerr(v[46], wires=0)
    qml.Squeezing(v[47], v, wires=1)
    qml.Squeezing(v[48], v, wires=2)
    qml.Kerr(v[49], wires=3)

    qml.Kerr(v[50], wires=1)
    qml.Kerr(v[51], wires=2)

    qml.Beamsplitter(v[52], v[53], wires=[0,1])
    qml.Beamsplitter(v[54], v[55], wires=[2,3])

    qml.Beamsplitter(v[56], v[57], wires=[1,2])

    qml.Beamsplitter(v[58], v[59], wires=[0,1])
    qml.Beamsplitter(v[60], v[61], wires=[2,3])

    qml.Rotation(v[62], wires=0)
    qml.Beamsplitter(v[63], v[64], wires=[1,2])
    qml.Rotation(v[65], wires=3)

    qml.Displacement(v[66], v, wires=0)
    qml.Rotation(v[67], wires=1)
    qml.Rotation(v[68], wires=2)
    qml.Displacement(v[69], v, wires=3)

    qml.Kerr(v[70], wires=0)
    qml.Displacement(v[71], v, wires=1)
    qml.Displacement(v[72], v, wires=2)
    qml.Kerr(v[73], wires=3)

    qml.Kerr(v[74], wires=2)
    qml.Kerr(v[75], wires=3)

    qml.draw()


    def DiamondMesh4Qumode_encode(x):
        qml.Squeezing(x[0],)