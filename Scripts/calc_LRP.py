"""
Functions are useful untilities for interpretation of ANN
 
Notes
-----
    Author : Zachary Labe
    Date   : 22 July 2020
    
Usage
-----
    [1] deepTaylorAnalysis(model,XtrainS,Ytrain,biasBool,option4,annType,classChunk,startYear)
    [2] def _gradient_descent_for_bwo(cnn_model_object, loss_tensor,
                                      init_function_or_matrices,
                                      num_iterations,learning_rate):
    [3] bwo_for_class(cnn_model_object,target_class,init_function_or_matrices,
                      num_iterations=DEFAULT_NUM_BWO_ITERATIONS,
                      learning_rate=DEFAULT_BWO_LEARNING_RATE)
    [4] optimal_input(model,input_img,target_class,num_iterations=200,
                      learning_rate = 0.01)
"""

###############################################################################
###############################################################################
###############################################################################

def deepTaylorAnalysis(model,XtrainS,Ytrain,biasBool,option4,annType,classChunk,startYear):
    """
    Calculate Deep Taylor for LRP
    """
    ### Import modules
    import numpy as np 
    import innvestigate
    import calc_Stats as SSS
    
    def invert_year_output(ypred,startYear):
        if(option4):
            inverted_years = SSS.convert_fuzzyDecade_toYear(ypred,startYear,
                                                        classChunk)
        else:
            inverted_years = SSS.invert_year_outputChunk(ypred,startYear)
        
        return inverted_years
    
    yearsUnique = np.unique(Ytrain)
    percCutoff = 90
    
    if(option4):
        withinYearInc = 2.
        errTolerance = 2. # originally 2 (8/6/2020)      
    
    # define prediction error
    Ypred = model.predict(XtrainS)
    if(annType=='class'):
        err = Ytrain[:,0] - invert_year_output(model.predict(XtrainS),
                                               startYear)
    elif(annType=='reg'):
        err = (Ypred*Ystd + Ymean) - Ytrain
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Create the innvestigate analyzer instance that we will call for each sample
    # analyzer = innvestigate.create_analyzer('deep_taylor',innvestigate.utils.model_wo_softmax(model))
    # analyzer = innvestigate.create_analyzer('deep_taylor',model)
    if(annType=='class'):
        model_nosoftmax = innvestigate.utils.model_wo_softmax(model)
    else:
        model_nosoftmax = model
    analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlphaBeta(
                                model_nosoftmax, alpha=1, beta=0, bias=biasBool)

    deepTaylorMaps = np.empty(np.shape(XtrainS))
    deepTaylorMaps[:] = np.nan

    # analyze each input via the analyzer
    for ival in np.arange(0,np.shape(XtrainS)[0]):

        # ensure error is small, i.e. model was correct
        if(np.abs(err[ival])<=errTolerance):
            sample = XtrainS[ival]
            analyzer_output = analyzer.analyze(sample[np.newaxis,...])
            deepTaylorMaps[ival] = analyzer_output/np.sum(analyzer_output.flatten())

    print('done with Deep Taylor analyzer normalization')     
    
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ### Compute the frequency of data at each point and the average relevance 
    ### normalized by the sum over the area and the frequency above the 90th 
    ### percentile of the map
    yearsUnique = np.unique(Ytrain)
    summaryDT = np.zeros((len(yearsUnique),np.shape(deepTaylorMaps)[1]))
    summaryDTFreq = np.zeros((len(yearsUnique),np.shape(deepTaylorMaps)[1]))
    summaryNanCount = np.zeros((len(yearsUnique),1))

    for iy, year in enumerate(yearsUnique):
        ### FIND YEARS WITHIN N YEARS OF EACH YEAR
        j = np.where(np.abs(Ytrain-year)<=withinYearInc)[0] 

        ### average relevance
        a = np.nanmean(deepTaylorMaps[j,:],axis=0)
        summaryDT[iy,:] = a[np.newaxis,...]

        ### frequency of non-nans
        nancount = np.count_nonzero(~np.isnan(deepTaylorMaps[j,1]))
        summaryNanCount[iy] = nancount

        ### frequency above percentile cutoff
        count = 0
        for k in j:
            b = deepTaylorMaps[k,:]
            if(~np.isnan(b[0])):
                count = count + 1
                pVal = np.percentile(b,percCutoff)
                summaryDTFreq[iy,:] = summaryDTFreq[iy,:]+np.where(b>=pVal,1,0)
        if(count==0):
            summaryDTFreq[iy,:] = 0
        else:
            summaryDTFreq[iy,:] = summaryDTFreq[iy,:]/count    
          
    return(summaryDT, summaryDTFreq, summaryNanCount)

###############################################################################
###############################################################################
###############################################################################

def _gradient_descent_for_bwo(
        cnn_model_object, loss_tensor, init_function_or_matrices,
        num_iterations, learning_rate):
    """
    Does gradient descent (the nitty-gritty part) for backwards optimization.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param loss_tensor: Keras tensor, defining the loss function to be
        minimized.
    :param init_function_or_matrices: Either a function or list of numpy arrays.
    If function, will be used to initialize input matrices.  See
    `create_gaussian_initializer` for an example.
    If list of numpy arrays, these are the input matrices themselves.  Matrices
    should be processed in the exact same way that training data were processed
    (e.g., normalization method).  Matrices must also be in the same order as
    training matrices, and the [q]th matrix in this list must have the same
    shape as the [q]th training matrix.
    :param num_iterations: Number of gradient-descent iterations (number of
        times that the input matrices are adjusted).
    :param learning_rate: Learning rate.  At each iteration, each input value x
        will be decremented by `learning_rate * gradient`, where `gradient` is
        the gradient of the loss function with respect to x.
    :return: list_of_optimized_input_matrices: length-T list of optimized input
        matrices (numpy arrays), where T = number of input tensors to the model.
        If the input arg `init_function_or_matrices` is a list of numpy arrays
        (rather than a function), `list_of_optimized_input_matrices` will have
        the exact same shape, just with different values.
    """
    ### Import modules
    import numpy as np
    import keras.backend as K
    import copy

    if isinstance(cnn_model_object.input, list):
        list_of_input_tensors = cnn_model_object.input
    else:
        list_of_input_tensors = [cnn_model_object.input]

    num_input_tensors = len(list_of_input_tensors)
    list_of_gradient_tensors = K.gradients(loss_tensor, list_of_input_tensors)

    for i in range(num_input_tensors):
        list_of_gradient_tensors[i] /= K.maximum(
            K.sqrt(K.mean(list_of_gradient_tensors[i] ** 2)),
            K.epsilon()
        )

    inputs_to_loss_and_gradients = K.function(
        list_of_input_tensors + [K.learning_phase()],
        ([loss_tensor] + list_of_gradient_tensors)
    )

    if isinstance(init_function_or_matrices, list):
        list_of_optimized_input_matrices = copy.deepcopy(
            init_function_or_matrices)
    else:
        list_of_optimized_input_matrices = [None] * num_input_tensors

        for i in range(num_input_tensors):
            these_dimensions = np.array(
                [1] + list_of_input_tensors[i].get_shape().as_list()[1:],
                dtype=int
            )

            list_of_optimized_input_matrices[i] = init_function_or_matrices(
                these_dimensions)

    for j in range(num_iterations):
        these_outputs = inputs_to_loss_and_gradients(
            list_of_optimized_input_matrices + [0]
        )

        if np.mod(j, 100) == 0:
            print('Loss after {0:d} of {1:d} iterations: {2:.2e}'.format(
                j, num_iterations, these_outputs[0]
            ))

        for i in range(num_input_tensors):
            list_of_optimized_input_matrices[i] -= (
                these_outputs[i + 1] * learning_rate
            )

    print('Loss after {0:d} iterations: {1:.2e}'.format(
        num_iterations, these_outputs[0]
    ))

    return list_of_optimized_input_matrices

###############################################################################
###############################################################################
###############################################################################

def bwo_for_class(
        cnn_model_object, target_class, init_function_or_matrices,
        num_iterations,learning_rate):
    """
    Does backwards optimization to maximize probability of target class.
    :param cnn_model_object: Trained instance of `keras.models.Model`.
    :param target_class: Synthetic input data will be created to maximize
        probability of this class.
    :param init_function_or_matrices: See doc for `_gradient_descent_for_bwo`.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :return: list_of_optimized_input_matrices: Same.
    """
    ### Import modules
    import numpy as np
    import keras.backend as K

    target_class = int(np.round(target_class))
    num_iterations = int(np.round(num_iterations))

    assert target_class >= 0
    assert num_iterations > 0
    assert learning_rate > 0.
    assert learning_rate < 1.

    num_output_neurons = (
        cnn_model_object.layers[-1].output.get_shape().as_list()[-1]
    )

    if num_output_neurons == 1:
        assert target_class <= 1

        if target_class == 1:
            loss_tensor = K.mean(
                (cnn_model_object.layers[-1].output[..., 0] - 1) ** 2
            )
        else:
            loss_tensor = K.mean(
                cnn_model_object.layers[-1].output[..., 0] ** 2
            )
    else:
        assert target_class < num_output_neurons

        loss_tensor = K.mean(
            (cnn_model_object.layers[-1].output[..., target_class] - 1) ** 2
        )

    return _gradient_descent_for_bwo(
        cnn_model_object=cnn_model_object, loss_tensor=loss_tensor,
        init_function_or_matrices=init_function_or_matrices,
        num_iterations=num_iterations, learning_rate=learning_rate)

###############################################################################
###############################################################################
###############################################################################

def optimal_input(model,input_img,target_class,num_iterations=200,learning_rate = 0.01):
    """ 
    OI
    """
    ### Define modules
    import numpy as np
    import keras.backend as K
    
    ### Need to change the out_loss calculation to use your loss equation
    ### Need to use the target_output variable
    # out_loss = - K.sum(target_output * K.log(model.layers[-1].output))
    out_loss = K.mean(
            (model.layers[-1].output[..., int(target_class)] - 1) ** 2
                )

    ### Calculate the gradients at the input layer WRT your output loss
    grad = K.gradients(out_loss, [model.input])[0]

    ### Create a function to iterate the loss and gradient
    ### Inputs are an image and the learning phase (0 for false)
    ### Outputs are the loss for the output and gradients WRT input layer
    iterate_fcn = K.function([model.input, K.learning_phase()], 
                             [out_loss, grad])

    for iterVal in np.arange(0,num_iterations):

        ### Calculate the loss and the gradients at the input layer based on the 
        ### current stage of the input image
        out_loss, out_grad = iterate_fcn([input_img, 0])

        ### Take a step along gradient WRT input -- 
        ### updates the input slightly towards its optimal input
        input_img -= out_grad*learning_rate
                
    return input_img

###############################################################################
###############################################################################
###############################################################################
