import os
import paramparse
from xml.etree import ElementTree
from lxml import etree


class Params:
    xml_path = ""
    dark_style = 'stylers.xml'
    light_style = 'DarkModeDefault.xml'
    dark_mode_file = 'npp_dark_mode.txt'
    light_mode_file = 'npp_light_mode.txt'

def main():
    params = Params()
    paramparse.process(params)

    # params  = paramparse.process(Params) #type: Params
    xml_path = params.xml_path

    parser = etree.XMLParser(encoding='utf-8')
    xml_tree = ElementTree.parse(xml_path, parser=parser)
    xml_root = xml_tree.getroot()

    # print()
    gui_root = xml_root.find('GUIConfigs')
    # print()
    gui_iter = gui_root.findall('GUIConfig')

    dark_mode_file = os.path.abspath(params.dark_mode_file)
    light_mode_file = os.path.abspath(params.light_mode_file)

    if os.path.exists(dark_mode_file):
        os.remove(dark_mode_file)

    if os.path.exists(light_mode_file):
        os.remove(light_mode_file)

    from datetime import datetime

    time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

    for gui_obj in gui_iter:
        # print(gui_obj.tag, gui_obj.attrib)

        if gui_obj.attrib['name'] == 'stylerTheme':
            style_dir = os.path.dirname(gui_obj.attrib['path'])
            style_name = os.path.basename(gui_obj.attrib['path'])
            if style_name == params.dark_style:
                style_name = params.light_style
            else:
                style_name = params.dark_style

            style_path = os.path.join(style_dir, style_name)
            gui_obj.attrib['path'] = style_path

        if gui_obj.attrib['name'] == 'DarkMode':
            if gui_obj.attrib['enable'] == 'no':
                out_file = light_mode_file
                gui_obj.attrib['enable'] = 'yes'
            else:
                out_file = dark_mode_file
                gui_obj.attrib['enable'] = 'no'
            print(f'out_file: {out_file}')
            with open(out_file, 'w') as fid:
                fid.write(time_stamp + '\n')

    xml_tree.write(xml_path)


if __name__ == '__main__':
    main()
