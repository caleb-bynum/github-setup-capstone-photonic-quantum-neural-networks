def ReckMesh():
    qml.Squeezing(v[0], 0.0, wires=0)
    qml.Squeezing(v[1], 0.0, wires=1)
    qml.Squeezing(v[2], 0.0, wires=2)
    qml.Squeezing(v[3], 0.0, wires=3)

    qml.Beamsplitter(v[4], v[5], wires=[2,3])

    qml.Beamsplitter(v[6], v[7], wires=[1,2])

    qml.Beamsplitter(v[8], v[9], wires=[0,1])
    qml.Beamsplitter(v[10], v[11], wires=[2,3])

    qml.Rotation(v[12], wires=0)
    qml.Beamsplitter(v[13], v[14], wires=[1,2])

    qml.Displacement(v[15], 0.0, wires=0)
    qml.Rotation(v[16], wires=1)
    qml.Beamsplitter(v[17], v[18], wires=[2,3])

    qml.Kerr(v[19], wires=0)
    qml.Displacement(v[20], 0.0, wires=1)
    qml.Rotation(v[21], wires=2)
    qml.Rotation(v[22], wires=3)

    qml.Kerr(v[23], wires=1)
    qml.Displacement(v[24], 0.0, wires=2)
    qml.Displacement(v[25], 0.0, wires=3)

    qml.Kerr(v[26], wires=2)
    qml.Kerr(v[27], wires=3)

    qml.Beamsplitter(v[28], v[29], wires=[2,3])

    qml.Beamsplitter(v[30], v[31], wires=[1,2])

    qml.Beamsplitter(v[32], v[33], wires=[0,1])
    qml.Beamsplitter(v[34], v[35], wires=[2,3])

    qml.Rotation(v[36], wires=0)
    qml.Beamsplitter(v[37], v[38], wires=[1,2])

    qml.Squeezing(v[39], 0.0, wires=0)
    qml.Rotation(v[40], wires=1)
    qml.Beamsplitter(v[41], v[42], wires=[2,3])

    qml.Kerr(v[43], wires=0)
    qml.Squeezing(v[44], 0.0, wires=1)
    qml.Rotation(v[45], wires=2)
    qml.Rotation(v[46], wires=3)

    qml.Kerr(v[47], wires=1)
    qml.Squeezing(v[48], 0.0, wires=2)
    qml.Squeezing(v[49], 0.0, wires=3)

    qml.Kerr(v[50], wires=2)
    qml.Kerr(v[51], wires=3)

    qml.Beamsplitter(v[52], v[53], wires=[2,3])

    qml.Beamsplitter(v[54], v[55], wires=[1,2])

    qml.Beamsplitter(v[56], v[57], wires=[0,1])
    qml.Beamsplitter(v[58], v[59], wires=[2,3])

    qml.Rotation(v[60], wires=0)
    qml.Beamsplitter(v[61], v[62], wires=[1,2])

    qml.Displacement(v[63], 0.0, wires=0)
    qml.Rotation(v[64], wires=1)
    qml.Beamsplitter(v[65], v[66], wires=[2,3])

    qml.Kerr(v[67], wires=0)
    qml.Displacement(v[68], 0.0, wires=1)
    qml.Rotation(v[69], wires=2)
    qml.Rotation(v[70], wires=3)

    qml.Kerr(v[71], wires=1)
    qml.Displacement(v[72], 0.0, wires=2)
    qml.Displacement(v[73], 0.0, wires=3)

    qml.Kerr(v[74], wires=2)
    qml.Kerr(v[75], wires=3)
