from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100

def print_activations(t):
    '''
    定义一个用来显示每一层网络结构的函数
    接受一个tensor作为输入，并显示其名称和tensor的尺寸
    :param t:
    :return:
    '''
    print(t.op.name,' ',t.get_shape().as_list())

def inference(images):
    '''
    接受images作为输入，返回最后一层pool5及parameters(Alexnet中所有需要训练的模型参数)
    :param images:
    :return:pool5,parameters
    '''
    parameters = []

    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,stddev = 1e-1),name='weights')
        conv = tf.nn.conv2d(images,kernel,strides=[1,4,4,1],padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[64],dtype = tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias,name = scope)
        print_activations(conv1)
        parameters += [kernel,biases]

    lrn1 = tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta = 0.75,name='lrn1')
    pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides = [1,2,2,1],padding = 'VALID',name='pool1')
    print_activations(pool1)

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,64,192],dtype = tf.float32,stddev = 1e-1),name='weights')
        conv = tf.nn.conv2d(pool1,kernel,strides = [1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape = [192],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(bias,name=scope)
        print_activations(conv2)
        parameters += [kernel,biases]

    lrn2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9,beta = 0.75,name= 'lrn2')
    pool2 = tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool2')
    print_activations(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,stddev = 1e-1),name="weights")
        conv = tf.nn.conv2d(pool2,kernel,strides=[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[384],dtype = tf.float32),trainable=True,name = 'biases')
        bias = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
        print_activations(conv3)

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,384,256],dtype = tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(conv3,kernel,strides=[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape = [256]),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv4 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
        print_activations(conv4)

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,256,256],dtype = tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(conv4,kernel,strides=[1,1,1,1],padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape = [256]),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv5 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
        print_activations(conv5)

    pool5 = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides = [1,2,2,1],padding='VALID',name = 'pool5')
    print_activations(pool5)

    return pool5,parameters

def time_tensorflow_run(session,target,info_string):
    '''
    评估Ａｌｅｘｎｅｔ每轮计算时间
    :param session: tensorflow的session作为输入
    :param target: 需要测评的运算算子
    :param info_string: 测试的名称
    :return:
    '''
    num_steps_burn_in = 10   #定义预热轮数为１０，用来给程序热身，骰记录呢迭代可能有显存加载，cache命中等问题
    total_duration = 0.0   #用来记录总时间
    total_duration_squared = 0.0   #储存每次计算时间的平方，用来计算方差

    for i in range(num_batches+num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i>= num_steps_burn_in:
            if not i%10:
                print('%s:step %d,duration = %.3f'%(datetime.now(),i-num_steps_burn_in,duration))
            total_duration += duration
            total_duration_squared += duration**2

    mn = total_duration/num_batches
    vr = total_duration_squared/num_batches - mn**2
    sd = math.sqrt(vr)
    print("%s:%s across %d steps, %.3f +/- %.3f sec/batch"%(datetime.now(),info_string,num_batches,mn,sd))
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],dtype = tf.float32,stddev=1e-1))
        pool5,parameters = inference(images)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess,pool5,"Forward")

        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective,parameters)

run_benchmark()