import pyperclip
from Tkinter import Tk
from anytree import Node, RenderTree


def findChildren(_headings, root_level, _start_id, _root_node, n_headings):
    nodes = []
    _id = _start_id
    while _id < n_headings:
        words = _headings[_id].split(' ')
        curr_level = words[0].count('#')

        if curr_level <= root_level:
            break

        heading_words = []
        for word in words[1:]:
            if word.startswith('@'):
                break
            if word and not word.isspace():
                heading_words.append(word)

        parent_text = ''
        if _root_node is not None and _root_node.parent is not None:
            parent_text = _root_node.name
            if curr_level > 2:
                # parent_text = str(_root_node)
                parent_text = '{}/{}'.format(_root_node.parent_text, parent_text)
        heading_text = '_'.join(heading_words)
        new_node = Node(heading_text, parent=_root_node, orig_text=_headings[_id], parent_text=parent_text,
                        marker=words[0], id=_id)
        nodes.append(new_node)

        child_nodes, ___id = findChildren(_headings, curr_level, _id + 1, new_node, n_headings)
        nodes += child_nodes
        _id = ___id


    return nodes, _id


def main():
    in_txt = Tk().clipboard_get()

    lines = in_txt.split('\n')
    lines = [line for line in lines if line.strip()]
    start_t = None
    curr_t = None

    curr_root = Node("root_node")
    headings = [k for k in lines if k.startswith('#')]
    n_headings = len(headings)
    heading_id = 0
    level = 0

    nodes, _ = findChildren(headings, 0, 0, curr_root, n_headings)

    print(RenderTree(curr_root))
    out_txt = in_txt

    for node in nodes:
        if node.is_root or node.parent.is_root:
            continue
        orig_text = node.orig_text
        new_text = '{} {} @ {}\n'.format(node.marker, node.name, node.parent_text)

        print('{}: new_text: {}'.format(node, new_text))

        out_txt = out_txt.replace(orig_text + '\n', new_text)

    # print(out_txt)

    # with open(out_fname, 'w') as out_fid:
    #     out_fid.write(out_txt)

    try:
        pyperclip.copy(out_txt)
        spam = pyperclip.paste()
    except pyperclip.PyperclipException as e:
        print('Copying to clipboard failed: {}'.format(e))


if __name__ == '__main__':
    main()
