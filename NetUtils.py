import tensorflow as tf
import numpy as np

def weight_variable(shape, name):
    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
    return var

def bias_variable(shape, name):
    initial = tf.constant_initializer(0.01)
    var = tf.get_variable(name, shape, initializer=initial)
    return var

def weight_variable2(shape, name):
    initial = tf.constant_initializer(np.eye(shape[0],M=shape[1]))
    var = tf.get_variable(name, shape, initializer=initial)
    return var

def bias_variable2(shape, name):
    initial = tf.constant_initializer(0.0)
    var = tf.get_variable(name, shape, initializer=initial)
    return var

def upConv2d(input, weight, output, stride, name):
    return tf.nn.conv2d_transpose(input, weight, output, stride, padding='SAME', data_format='NHWC', name=name)

def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, ss):

    h_pool = tf.nn.max_pool(x, ksize=[1, ss, ss, 1], strides=[1, ss, ss, 1], padding='SAME')

    return h_pool

#---------------------------------------------------------------------------------------------------------------------------------------------------

def upConvLayerNoRelu(nbLayer, input, batchSize, kernelSize, inputFeatureSize, outputFeatureSize, outputSpacialSize, stride):

    with tf.name_scope("UpConvNoRelu-"+nbLayer):

        wName = nbLayer + "-wUpConv"
        bName = nbLayer + "-bUpConv"

        W_upConv1 = weight_variable([kernelSize, kernelSize, outputFeatureSize, inputFeatureSize], wName)
        b_upConv1 = bias_variable([outputFeatureSize], bName)

        out = upConv2d(input, W_upConv1, [batchSize, outputSpacialSize, outputSpacialSize, outputFeatureSize], [1, stride, stride, 1], "UpConv"+nbLayer)
        out = out + b_upConv1

        tf.summary.histogram("histogram-"+wName,W_upConv1)
        tf.summary.histogram("histogram-"+bName,b_upConv1)

    return out

def upConvLayer(nbLayer, input, batchSize, kernelSize, inputFeatureSize, outputFeatureSize, outputSpacialSize, stride):

    with tf.name_scope("UpConv-"+nbLayer):
        out = upConvLayerNoRelu(nbLayer, input, batchSize, kernelSize, inputFeatureSize, outputFeatureSize, outputSpacialSize, stride)

        relu = tf.maximum(0.01 * out, out, name=nbLayer + "-upConvRelu")

    return relu

#---------------------------------------------------------------------------------------------------------------------------------------------------

def fullyLayerNoRelu(nbLayer, input, inputSize, outSize):

    with tf.name_scope("FullyNoRelu-"+nbLayer):
        wName = nbLayer + "-wFc"
        bName = nbLayer + "-bFc"

        W_fc = weight_variable([inputSize, outSize], wName)
        b_fc = bias_variable([outSize], bName)

        score = tf.matmul(input, W_fc) + b_fc

        tf.summary.histogram("histogram-"+wName,W_fc)
        tf.summary.histogram("histogram-"+bName,b_fc)

    return score

def fullyLayer(nbLayer, input, inputSize, outSize):

    with tf.name_scope("Fully-"+nbLayer):
        out = fullyLayerNoRelu(nbLayer, input, inputSize, outSize)
        relu = tf.maximum(0.01 * out, out, name="Fc"+nbLayer)

    return relu

#---------------------------------------------------------------------------------------------------------------------------------------------------

def convPoolLayerNoRelu(nbLayer, inputScore, kernelSize, kernelDeep, outputSize, stride):

    with tf.name_scope("ConvNoRelu-"+nbLayer):
        wName = nbLayer + "-W_convPool"
        bName = nbLayer + "-b_convPool"

        W_conv = weight_variable([kernelSize, kernelSize, kernelDeep, outputSize], wName)
        b_conv = bias_variable([outputSize], bName)

        out = tf.add(tf.nn.conv2d(inputScore, W_conv, strides=[1, stride, stride, 1], padding='SAME'), b_conv)

        tf.summary.histogram("histogram-"+wName,W_conv)
        tf.summary.histogram("histogram-"+bName,b_conv)

    return out

def convPoolLayer(nbLayer, inputScore, kernelSize, kernelDeep, outputSize, stride):

    with tf.name_scope("Conv-"+nbLayer):
        out = convPoolLayerNoRelu(nbLayer, inputScore, kernelSize, kernelDeep, outputSize, stride)
        relu = tf.maximum(0.01 * out, out, name=("convpoolRelu" + nbLayer))

    return relu

#---------------------------------------------------------------------------------------------------------------------------------------------------

def resLayer(nbLayer, inputx, inputSize, outputSize):

    with tf.name_scope("Res-"+ nbLayer):
        residu = inputx

        if (outputSize == (2 * inputSize)):
            residu = tf.concat([inputx, inputx], 3)

        score1 = convPoolLayerNoRelu(nbLayer + "-1", inputx, 3, inputSize, inputSize, 1)
        relu1 = tf.maximum(0.01 * score1, score1, name=("resRelu" + nbLayer + str(0.1)))
        score2 = convPoolLayerNoRelu(nbLayer + "-2", relu1, 3, inputSize, outputSize, 1)
        score3 = tf.add(score2, residu)
        relu2 = tf.maximum(0.01 * score3, score3, name=("resRelu" + nbLayer + str(0.2)))

    return relu2

#---------------------------------------------------------------------------------------------------------------------------------------------------

def rnnCell(nbLayer, inputx, nbNeuron, state):

    with tf.name_scope("RnnCell-" + nbLayer):
        WhxName = nbLayer+"-Whx"
        WhhName = nbLayer+"-Whh"
        WohName = nbLayer+"-Woh"
        bxhName = nbLayer+"-bxh"
        bohName = nbLayer+"-boh"

        Whx = weight_variable([nbNeuron,nbNeuron],WhxName)
        Whh = weight_variable([nbNeuron,nbNeuron],WhhName)
        Woh = weight_variable([nbNeuron,nbNeuron],WohName)
        bxh = bias_variable2([nbNeuron],bxhName)
        boh = bias_variable2([nbNeuron],bohName)

        ht = tf.tanh(tf.matmul(inputx,Whx)+tf.matmul(state,Whh)+bxh)

        ot = tf.tanh(tf.matmul(ht,Woh)+boh)

    return ot, ht

#---------------------------------------------------------------------------------------------------------------------------------------------------

def customCell_1(nbLayer, inputx, nbNeuron, state):

    with tf.name_scope("CustomCell1-"+nbLayer):
        i = fullyLayerNoRelu(nbLayer + "-I", inputx, nbNeuron, nbNeuron)
        i = tf.tanh(i)

        f = fullyLayerNoRelu(nbLayer + "-H", state, nbNeuron, nbNeuron)
        f = tf.tanh(f)

        h = i + f

        y = fullyLayerNoRelu(nbLayer + "-O", h, nbNeuron, nbNeuron)
        y = tf.tanh(y)

    return y , h

#---------------------------------------------------------------------------------------------------------------------------------------------------

def customCell_2(nbLayer, inputx, nbNeuron, state):

    with tf.name_scope("CustomCell2-"+nbLayer):
        combine = inputx + state

        fg = fullyLayerNoRelu(nbLayer + "-G", combine, nbNeuron, nbNeuron)
        fg = tf.sigmoid(fg)

        ig = 1 - fg

        i = ig * inputx
        f = fg * state

        h = i + f

        y = fullyLayerNoRelu(nbLayer+"-O",h,nbNeuron,nbNeuron)
        y = tf.tanh(y)

    return y , h

#---------------------------------------------------------------------------------------------------------------------------------------------------

def gruCell(nbLayer, inputx, nbNeuron, state, drop):

    with tf.name_scope("GruCell-"+nbLayer):
        WzName = nbLayer+"-Wz"
        WrName = nbLayer+"-Wr"
        WcName = nbLayer+"-Wc"
        UzName = nbLayer+"-Uz"
        UrName = nbLayer+"-Ur"
        UcName = nbLayer+"-Uc"
        bzName = nbLayer+"-bz"
        brName = nbLayer+"-br"
        bcName = nbLayer+"-bc"

        Wz = weight_variable([nbNeuron,nbNeuron],WzName)
        Wr = weight_variable([nbNeuron,nbNeuron],WrName)
        Wc = weight_variable([nbNeuron,nbNeuron],WcName)
        Uz = weight_variable([nbNeuron,nbNeuron],UzName)
        Ur = weight_variable([nbNeuron,nbNeuron],UrName)
        Uc = weight_variable([nbNeuron,nbNeuron],UcName)
        bz = bias_variable2([nbNeuron],bzName)
        br = bias_variable2([nbNeuron],brName)
        bc = bias_variable2([nbNeuron],bcName)

        zt = tf.sigmoid(tf.matmul(inputx,Wz) + tf.matmul(state,Uz) + bz)
        rt = tf.sigmoid(tf.matmul(inputx,Wr) + tf.matmul(state,Ur) + br)
        wt = 1 - zt

        rs = rt * state

        sc = tf.tanh(tf.matmul(rs,Wc) + tf.matmul(inputx,Uc) + bc)

        ht = (wt * state) + (zt * sc)
        h_drop = tf.nn.dropout(ht,drop)

        y = fullyLayerNoRelu(nbLayer+"-O",h_drop,nbNeuron,nbNeuron)
        y = tf.tanh(y)

    return y , ht

#---------------------------------------------------------------------------------------------------------------------------------------------------

def descriptorNet(x):

    h_conv = convPoolLayer("1",x,7,3,64,2)

    h_conv = max_pool(h_conv,2)

    h_conv = resLayer("1",h_conv,64,128)

    h_conv = max_pool(h_conv,2)

    h_conv = resLayer("2",h_conv,128,256)

    h_conv = max_pool(h_conv,2)

    h_conv = resLayer("3",h_conv,256,512)

    h_conv = max_pool(h_conv,2)

    h_conv = resLayer("4",h_conv,512,512)

    return h_conv





