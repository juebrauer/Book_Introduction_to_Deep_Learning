import numpy as np

gt_p1 = 1.111
gt_p2 = 2.222
gt_p3 = 3.333
gt_p4 = 4.444
gt_p5 = 5.555
gt_p6 = 6.666

def compute_loss(t,y):
    return (t-y)**2


def f(w1, w2, t, p1,p2,p3,p4,p5,p6):

    # FORWARD STEP

    # hidden neuron 1:
    w3 = p1*w1
    w4 = p2*w2
    w5 = w3+w4

    # hidden neuron 2:
    w6 = p3*w1
    w7 = p4*w2
    w8 = w6+w7

    # output neuron
    w9 = p5*w5
    w10 = p6*w8
    w11 = w9+w10

    # computation of loss/error
    loss = compute_loss(t,w11)

    # BACKWARD STEP
    _w12 = 1.0
    _w11 = _w12*2.0*(t-w11)*(-1.0)

    # left part of the tree
    _w9 = _w11*1.0
    _p5 =  _w9 * w5
    _w5 = _w9*p5
    _w3 = _w5*1.0
    _w4 = _w5*1.0
    _p1 = _w3*w1
    _p2 = _w4*w2

    # right part of the tree
    _w10 = _w11 * 1.0
    _p6  = _w10 * w8
    _w8 = _w10 * p6
    _w6 = _w8 * 1.0
    _w7 = _w8 * 1.0
    _p3 = _w6 * w1
    _p4 = _w7 * w2


    # COMPILE GRADIENT
    gradient = (_p1,_p2,_p3,_p4,_p5,_p6)

    # return function value f(w1,w2)
    # and gradient of error E(w1,w2,t)=(t-w11)^2
    return w11, gradient


def numdiff_dfdp5(w1, w2, t, p1,p2,p3,p4,p5,p6):
    h = 0.0000001
    f1, _ = f(w1, w2, t, p1,p2,p3,p4,p5+h,p6)
    f2, _ = f(w1, w2, t, p1,p2,p3,p4,p5,p6)
    #print("f1=" + str(f1))
    #print("f2=" + str(f2))
    return (f1 - f2) / h


def manualdiff_dfdp5(w1, w2, t, p1,p2,p3,p4,p5,p6):
    y,_ = f(w1,w2,t, p1,p2,p3,p4,p5,p6)
    deriv = 2*(t-y)*(-(p1*w1+p2*w2))
    return deriv



def compute_avg_error( test_data, p1,p2,p3,p4,p5,p6 ):
    sum_losses = 0.0
    NrTestSamples = len(test_data)

    for i in range(0, NrTestSamples):

        # 1. get next test sample
        test_sample = test_data[i]
        w1 = test_sample[0]
        w2 = test_sample[1]
        t  = test_sample[2]

        # 2. forward step
        y,_ = f(w1,w2,t,p1,p2,p3,p4,p5,p6)

        # 3. compute sample loss
        sample_loss = compute_loss(t,y)

        # 4. compute sum of all samples losses
        sum_losses += sample_loss

    # compute average error
    avg_error = sum_losses / NrTestSamples

    return avg_error





def train_model_using_gradient_descent():

    # 1. generate training data
    NrTrainingSamples = 1000
    train_data = []
    for i in range(0, NrTrainingSamples):

        # 1.1 guess a random w1,w2 coordinates
        w1 = np.random.rand()
        w2 = np.random.rand()

        # 1.2 compute corresponding output value y
        y, _ = f(w1, w2, 0.0, gt_p1, gt_p2, gt_p3, gt_p4, gt_p5, gt_p6)

        # 1.3 store that training sample
        train_sample = (w1, w2, y)
        train_data.append(train_sample)


    # 2. generate test data
    NrTestSamples = 1000
    test_data = []
    for i in range(0, NrTestSamples):
        # 1.1 guess a random w1,w2 coordinates
        w1 = np.random.rand()
        w2 = np.random.rand()

        # 1.2 compute corresponding output value y
        y, _ = f(w1, w2, 0.0, gt_p1, gt_p2, gt_p3, gt_p4, gt_p5, gt_p6)

        # 1.3 store that test sample
        test_sample = (w1, w2, y)
        test_data.append(test_sample)


    # 3. TRAINING:
    #    choose a training sample,
    #    forward it and compute the gradient
    #    then change parameters p1,...,p6 such that
    #    we go a small step into the direction of
    #    the negative gradient

    # IMPORTANT!
    # Now, we start with random parameters p1,...,p6
    # and we want to learn the parameters p1,...,p6
    # such that the training inputs are mapped to the
    # teacher output values better and better
    p1 = np.random.uniform(-1.0,1.0)
    p2 = np.random.uniform(-1.0,1.0)
    p3 = np.random.uniform(-1.0,1.0)
    p4 = np.random.uniform(-1.0,1.0)
    p5 = np.random.uniform(-1.0,1.0)
    p6 = np.random.uniform(-1.0,1.0)

    NrTrainingSteps = 1000
    for s in range(0, NrTrainingSteps):

        # 2.1 guess a random sample index
        rnd_idx = np.random.randint(0, NrTrainingSamples)

        # 2.2 get corresponding training triple
        train_sample = train_data[rnd_idx]
        w1 = train_sample[0]
        w2 = train_sample[1]
        t = train_sample[2]

        # 2.3 compute actual output y and
        #     compute gradient of error function
        y, grad = f(w1,w2,t, p1,p2,p3,p4,p5,p6)

        # 2.4 GRADIENT DESCENT:
        #     adapt the parameters p1,...,p6
        #     a little bit into the direction of the
        #     negative gradient
        LEARN_RATE = 0.001
        p1 += -grad[0] * LEARN_RATE
        p2 += -grad[1] * LEARN_RATE
        p3 += -grad[2] * LEARN_RATE
        p4 += -grad[3] * LEARN_RATE
        p5 += -grad[4] * LEARN_RATE
        p6 += -grad[5] * LEARN_RATE

        # 2.4 compute error/loss on test data
        avg_error = compute_avg_error( test_data, p1,p2,p3,p4,p5,p6 )
        print("After training step "+str(s)+" --> avg_error = {0:.5f} ".format(avg_error), end="  ")
        print("p1={0:.3f} ".format(p1) +
              "p2={0:.3f} ".format(p2) +
              "p3={0:.3f} ".format(p3) +
              "p4={0:.3f} ".format(p4) +
              "p5={0:.3f} ".format(p5) +
              "p6={0:.3f} ".format(p6)
              )

def main():

    w1 = 3
    w2 = 4
    t = 0.0
    y, gradient = f(w1, w2, t, gt_p1,gt_p2,gt_p3,gt_p4,gt_p5,gt_p6)
    print("f(3,4) = " + str(y))

    print("Numeric differentiation --> dfdp5(3,4) = " +
          str(numdiff_dfdp5(w1, w2, t, gt_p1, gt_p2, gt_p3, gt_p4, gt_p5, gt_p6)))

    print("Manual differentiation --> dfdp5(3,4) = " +
          str(manualdiff_dfdp5(w1, w2, t, gt_p1,gt_p2,gt_p3,gt_p4,gt_p5,gt_p6)))

    print("Reverse-Mode Autodiff: gradient= " + str(gradient))

    train_model_using_gradient_descent()


main()







