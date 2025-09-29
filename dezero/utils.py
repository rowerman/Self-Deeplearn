# 函数名前添加下划线表示只在本文件中使用此函数
# verbose参数表示是否显示变量的形状和数据类型
import os
import subprocess

def _dot_var(v, verbose=False):
    _dot_var = '{} [label = "{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return _dot_var.format(id(v), name)

def _dot_func(f):
    _dot_func = '{} [label = "{}", color=lightblue, style=filled, shape=box]\n'
    ret = _dot_func.format(id(f), f.__class__.__name__)

    dot_edges = '{} -> {}\n'
    for x in f.inputs:
        ret += dot_edges.format(id(x), id(f))
    for y in f.outputs:
        ret += dot_edges.format(id(f), id(y()))
    return ret

def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)
    while funcs:
        f = funcs.pop()
        txt += _dot_func(f)
        for x in f.inputs:
            txt += _dot_var(x, verbose)
            if x.creator is not None:
                add_func(x.creator)
    
    return 'digraph g {\n' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    # 将dot数据保存至文件
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    # 调用dot命令
    extension = os.path.splitext(to_file)[1][1:]  # 获取扩展名
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)  # shell=True表示通过shell来执行cmd命令