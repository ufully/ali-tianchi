# -*- coding:utf-8 -*-
import argparse
import tensorflow as tf
from data.data_loader import *

def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        'prefix/input_x:0': x_batch,
        # 'input_y': y_batch,
        'prefix/keep_prob:0': keep_prob
    }
    return feed_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="checkpoints/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()
    # 加载已经将参数固化后的图
    graph = load_graph(args.frozen_model_filename)

    # We can list operations
    # op.values() gives you a list of tensors it produces
    # op.name gives you the name
    # 输入,输出结点也是operation,所以,我们可以得到operation的名字
    for op in graph.get_operations():
        print(op.name, op.values())
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
    # 操作有:prefix/Placeholder/inputs_placeholder
    # 操作有:prefix/Accuracy/predictions
    # 为了预测,我们需要找到我们需要feed的tensor,那么就需要该tensor的名字
    # 注意prefix/Placeholder/inputs_placeholder仅仅是操作的名字,prefix/Placeholder/inputs_placeholder:0才是tensor的名字
    # x = graph.get_tensor_by_name('prefix/Placeholder/inputs_placeholder:0')
    # y = graph.get_tensor_by_name('prefix/Accuracy/predictions:0')
    x = graph.get_tensor_by_name('prefix/input_x:0')
    y = graph.get_tensor_by_name('prefix/score/output:0')

    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab('data/vocab.txt')

    xval = process_test_file('data/test.data.csv', word_to_id, cat_to_id )
    feed_dict = feed_data(xval, y, 0.5)

    with tf.Session(graph=graph, config=tf.ConfigProto(device_count={'GPU':0})) as sess:
        y_out = sess.run(y, feed_dict=feed_dict)
        print(y_out)  # [[ 0.]] Yay!
        print(type(y_out))
        idx = 1
        for v_y in y_out:
            open('data/result.tmp.txt', 'a').write(str(idx) + "\t" + str(categories[v_y]) + "\n")
            idx += 1

    print("finish")