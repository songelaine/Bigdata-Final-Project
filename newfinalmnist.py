
from flask import Flask,request
from scipy import misc
import tensorflow as tf
import logging
logging.basicConfig(level=logging.INFO)
import datetime

log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)
#from cassandra.cluster import Cluster
#from cassandra import ConsistencyLevel
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

KEYSPACE = "elainekeyspace"
session=0
def createKeySpace():
    global session
    cluster = Cluster(contact_points=['127.0.0.1'],port=9042)
    session = cluster.connect()

    log.info("Creating keyspace...")
    try:
        session.execute("""
            CREATE KEYSPACE %s
            WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
            """ % KEYSPACE)

        log.info("setting keyspace...")
        session.set_keyspace(KEYSPACE)

        log.info("creating table...")
        session.execute("""
            CREATE TABLE mnistnumber (
                id int,
                filename text,
                number text,
                uploadtime text,
                PRIMARY KEY (id)
            )
            """)

    except Exception as e:
        log.error("Unable to create keyspace")
        log.error(e)
        
createKeySpace();

app = Flask(__name__)


def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def imageprepare_stream(img):
    
    im = img
    return im


def preditc_img(img,name):
    if img is not None:
        result = imageprepare_stream(img)
   
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    saver = tf.train.Saver()
    res=-1

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "/Users/song/Documents/Bigdata/new_model/model.ckpt")  # 使用模型，参数和之前的代码保持一致
        prediction = tf.argmax(y_conv, 1)
        predint = prediction.eval(feed_dict={x: result, keep_prob: 1.0}, session=sess)

        print('识别结果:')
        print(predint[0])
    res = predint[0]

    return res

        
        #cql = """INSERT INTO mnistnumber(filename, number, uploadtime)
              #VALUES(
              #'%s',
             # '%s',
            #  '%s');"""%( name, str(res), time );
                     

idnum=1
@app.route('/upload', methods=['POST'])
def upload():
    global idnum
    f = request.files['file']
    name = f.filename
    im = misc.imread(f)
    img = im.reshape((1,784))
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    l = preditc_img(img,name)

    cql = """INSERT INTO mnistnumber(id, filename, number, uploadtime)VALUES("""+str(idnum)+""",'"""+name+"""','"""+str(l)+"""','"""+str(time)+"""');"""
    idnum=idnum+1
    print("The csql is:"+cql)
    global session
    session.set_keyspace(KEYSPACE)
    session.execute(cql)
    
    res = '''
    	<!doctype html>
    	<html>
    	<body>
    	<form action='/upload' method='post' enctype='multipart/form-data'>
      		<input type='file' name='file'>
        	<input type='submit' value='Upload'>
    	</form>
    	<p>predict: %s </p>
    	</body>
    	</html>
    	'''%(l)

    # return 'predict: %s ' % (l)
    return res
    

@app.route('/')
def index():
    return '''
    	<!doctype html>
    	<html>
    	<body>
    	<form action='/upload' method='post' enctype='multipart/form-data'>
      		<input type='file' name='file'>
        	<input type='submit' value='Upload'>
    	</form>
    	</body>
    	</html>
    	'''


if __name__ == '__main__':
    app.debug = True
    app.run()