import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

DEFAULT_PADDING = 'SAME'
N_CLASSES = 16


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True, is_training=False, n_classes=20):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        self.setup(is_training, n_classes)

    def setup(self, is_training, n_classes):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, basestring):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self


    def feed_n(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            # if isinstance(fed_layer, basestring):
            try:
                fed_layer = fed_layer
            except KeyError:
                raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape, trainable=self.trainable)

    def make_w_var(self, name, shape):
        '''Creates a new TensorFlow variable.'''
        stddev=0.01
        return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev), trainable=self.trainable)

    def make_b_var(self, name, shape):
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0), trainable=self.trainable)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_w_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_b_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def conv_layer(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_w_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_b_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def atrous_conv(self,
                    input,
                    k_h,
                    k_w,
                    c_o,
                    dilation,
                    name,
                    relu=True,
                    padding=DEFAULT_PADDING,
                    group=1,
                    biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation, padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output
        
    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name)
        
    # @layer
    # def batch_normalization(self, input, name, is_training, activation_fn=None, scale=True):
    #     with tf.variable_scope(name) as scope:
    #         output = slim.batch_norm(
    #             input,
    #             activation_fn=activation_fn,
    #             is_training=is_training,
    #             updates_collections=None,
    #             scale=scale,
    #             scope=scope)
    #         return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)

    @layer
    def batch_normalization(self, input, name = None):
        return tf.layers.batch_normalization(input)

    @layer
    def pad(self, input, padd, name):
        with tf.variable_scope(name) as scope:
            # pad = tf.pad(input, padd, name='pad_1')
            return input

    @layer
    def upsample(self, input, size_h, size_w,name):
        with tf.variable_scope(name) as scope:
            return tf.image.resize_nearest_neighbor(input, size=[size_h, size_w])

    @layer
    def sigmoid(self, input, name):
        with tf.variable_scope(name) as scope:
            return tf.nn.sigmoid(input)
    
    def replicate(self, input, numIn, dim, name):
        with tf.variable_scope(name) as scope:
            repeat = []
            for i in xrange(numIn):
                repeat.append(input)
            return tf.concat(repeat, dim)
    @layer
    def multiply2(self, input, name):
        with tf.variable_scope(name) as scope:
            return tf.multiply(input[0], input[1])

    @layer
    def printLayer(self, input, name):
        with tf.variable_scope(name) as scope:
            print(input)
            return 0
    @layer
    def stack(self, input, axis, name):
        with tf.variable_scope(name) as scope:
            return tf.squeeze(tf.stack(input, axis=axis))

    def conv_block(self, input, numIn, numOut, name):
        with tf.variable_scope(name) as scope:
            bn1 = tf.layers.batch_normalization(input, name = 'bn1')
            relu1 = tf.nn.relu(bn1, name = 'relu1')
            conv1 = self.conv(relu1,1, 1, numOut/2, 1, 1, biased=True, relu=False, name = 'conv1', padding='SAME')
            bn2 = tf.layers.batch_normalization(conv1, name = 'bn2')
            relu2 = tf.nn.relu(bn2, name = 'relu2')
            conv2 = self.conv(relu2,3, 3, numOut/2, 1, 1, biased=True, relu=False, name = 'conv2', padding='SAME')
            bn3 = tf.layers.batch_normalization(conv2, name = 'bn3')
            relu3 = tf.nn.relu(bn3, name = 'relu3')
            conv3 = self.conv(relu3,1, 1, numOut, 1, 1, biased=True, relu=False, name = 'conv3', padding='SAME')        
            return conv3

    def skip_layer(self, input, numIn, numOut, name):
        with tf.variable_scope(name) as scope:
            if numIn == numOut:
                return input
            else:
                conv = self.conv(input, 1, 1, numOut, 1, 1, biased=True, relu=False, name = 'skip', padding='SAME')
                return conv

    def pool_layer(self, input, numIn, numOut, name):
        with tf.variable_scope(name) as scope:
            bn1 = tf.layers.batch_normalization(input, name = 'bn1')
            relu1 = tf.nn.relu(bn1, name = 'relu1')
            pool1 = self.max_pool(relu1, 2, 2, 2, 2, name = 'pool1')
            conv1 = self.conv(pool1, 3, 3, numOut, 1, 1, biased=True, relu=False, name = 'conv1', padding='SAME')
            bn2 = tf.layers.batch_normalization(conv1, name = 'bn2')
            relu2 = tf.nn.relu(bn2, name = 'relu2')
            conv2 = self.conv(relu2,3, 3, numOut, 1, 1, biased=True, relu=False, name = 'conv2', padding='SAME')
            upsample =  tf.image.resize_nearest_neighbor(conv2, size=[int(conv2.get_shape()[1]) * 2, int(conv2.get_shape()[1]) * 2])
            return upsample

    
    def Residual(self, input, numIn, numOut, name):
        with tf.variable_scope(name) as scope:
            conv_b = self.conv_block(input, numIn, numOut, name = 'Conv_Block')
            skip_l = self.skip_layer(input, numIn, numOut, name = 'Skip_Layer')
            add = tf.add_n([conv_b, skip_l])
            return add
            # return tf.add_n([self.conv_b, self.skip_1])

    
    def ResidualPool(self, input, numIn, numOut, name):
        with tf.variable_scope(name) as scope:
            conv_b = self.conv_block(input, numIn, numOut, name = 'Conv_Block')
            skip_l = self.skip_layer(input, numIn, numOut, name = 'Skip_Layer')
            pool_l = self.pool_layer(input, numIn, numOut, name = 'Pool_Layer')
            return tf.add_n([conv_b, skip_l, pool_l])

    def AttentionIter(self, input, numIn, lrnSize, itersize, name):
        with tf.variable_scope(name) as scope:
            U = self.conv(input, 3, 3, 1, 1, 1, biased=True, relu=False, name = 'conv1', padding='SAME')
            # with tf.variable_scope('spConv') as scope:
            #     spConv1 = self.conv(input, lrnSize, lrnSize, 1, 1, 1, biased=True, relu=False, name = 'sp_conv', padding='SAME')
            Q = []
            C = []
            for i in range(0,itersize):
                if i == 0:
                    with tf.variable_scope('spConv', reuse = False):
                        conv = self.conv(U, lrnSize, lrnSize, 1, 1, 1, biased=True, relu=False, name = 'sp_conv', padding='SAME')
                else:
                    with tf.variable_scope('spConv', reuse = True):
                        conv = self.conv(Q[i-1], lrnSize, lrnSize, 1, 1, 1, biased=True, relu=False, name = 'sp_conv', padding='SAME')
                C.append(conv)
                Q_tmp = tf.nn.sigmoid(tf.add_n([C[i], U]))# 
                Q.append(Q_tmp)
            replicate = self.replicate(Q[itersize-1], numIn, -1, name = '_replicate')   #******Q[itersize]-->Q[itersize-1]  2-->3
            pheat = tf.nn.sigmoid(tf.multiply(input, replicate))
            return pheat

    @layer
    def AttentionPartsCRF(self, input, numIn, lrnSize, itersize, usepart, name):
        with tf.variable_scope(name) as scope:
            if usepart == 0:
                return self.AttentionIter(input, numIn, lrnSize, itersize, name = '_AttIter')
            else:
                partnum = N_CLASSES
                pre = []
                for i in range(0,N_CLASSES):
                    att = self.AttentionIter(input, numIn, lrnSize, itersize, name = 'att_'+ str(i))
                    s = self.conv(att, 1, 1, 1, 1, 1, biased=True, relu=False, name = 's_' + str(i), padding='SAME')
                    pre.append(s)
                return tf.concat(pre, -1)

    def repResidual(self, input, num, nRep, name):
        with tf.variable_scope(name) as scope:
            out = []
            for i in range(0, nRep):  #***1-->0
                if i == 0:
                    tmpout = self.Residual(input, num, num, name = 'tmpout_' + str(i))
                else:
                    tmpout = self.ResidualPool(out[i-1], num, num, name = 'tmpout_' + str(i))
                out.append(tmpout)
            return out[nRep-1]

    
    def hourglasses(self, input, n, f, imsize, nModual, name):
        with tf.variable_scope(name) as scope:
            # upper branch
            pool = self.max_pool(input, 2, 2, 2, 2, name = 'pool')
            up = []
            low = []
            for i in range(0,nModual):
                if i==0:
                    if n>1:
                        tmpup = self.repResidual(input, f, n-1, name = 'tmpup_' + str(i))
                    else:
                        tmpup = self.Residual(input, f, f, name = 'tmpup_' + str(i))
                    tmplow = self.Residual(pool, f, f, name = 'tmplow_' + str(i))
                else:
                    if n>1:
                        tmpup = self.repResidual(up[i-1], f, n-1, name = 'tmpup_' + str(i))
                    else:
                        tmpup = self.Residual(up[i-1], f, f, name = 'tmpup_' + str(i))
                    tmplow = self.Residual(low[i-1], f, f, name = 'tmplow_' + str(i))
                up.append(tmpup)
                low.append(tmplow)

            # lower branch
            if n>1:
                low2 = self.hourglasses(low[nModual-1], n-1, f, imsize/2, nModual, name = 'low2')
                #print(low2.get_shape())
            else:
                low2 = self.Residual(low[nModual-1], f, f, name = 'low2')
            low3 = self.Residual(low2, f, f, name='low3')
            #print(low3.get_shape())
            up2 = tf.image.resize_nearest_neighbor(low3, size=[int(low3.get_shape()[1])*2, int(low3.get_shape()[1])*2])
            comb = tf.add_n([up[nModual-1], up2])
            return comb

    @layer
    def hourglass(self, input, n, f, imsize, nModual, name):
        return self.hourglasses(input, n, f, imsize, nModual, name)

    @layer
    def lin(self, input, numIn, numOut, name):
        with tf.variable_scope(name) as scope:
            l = self.conv(input, 1, 1, numOut, 1, 1, biased=True, relu=False, name = 'l_', padding='SAME')
            return tf.nn.relu(tf.layers.batch_normalization(l))

    @layer
    def preProcess(self, input, numOut, name):
        with tf.variable_scope(name) as scope:
            conv1_ = self.conv(input, 7, 7, 64, 1, 1, biased=True, relu=False, name = 'conv1_', padding='SAME')
            conv1 = tf.nn.relu(tf.layers.batch_normalization(conv1_))
            r1 = self.Residual(conv1, 64, 64, name = 'r1')
            pool1 = self.max_pool(r1, 2, 2, 2, 2, name = '_pool1')
            r2 = self.Residual(pool1, 64, 64, name = 'r2')
            r3 = self.Residual(r2, 64, 128, name = 'r3')

            pool2 = self.max_pool(r3, 2, 2, 2, 2, name = '_pool2')
            r4 = self.Residual(pool2, 128, 128, name = 'r4')
            r5 = self.Residual(r4, 128, 128, name = 'r5')
            r6 = self.Residual(r5, 128, numOut, name = 'r6')
            return r6





