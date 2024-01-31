import os
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
xml_path = "C:/Users/Tommy/AppData/Roaming/Notepad++/config.xml"
parser = etree.XMLParser(encoding='utf-8')
xml_tree = ElementTree.parse(xml_path, parser=parser)
xml_root = xml_tree.getroot()

# print()
gui_root = xml_root.find('GUIConfigs')
# print()
gui_iter = gui_root.findall('GUIConfig')

for gui_obj in gui_iter:
    # print(gui_obj.tag, gui_obj.attrib)

    if gui_obj.attrib['name'] == 'stylerTheme':
        style_dir = os.path.dirname(gui_obj.attrib['path'])
        style_name = os.path.basename(gui_obj.attrib['path'])
        if style_name == 'stylers.xml':
            style_name = 'DarkModeDefault.xml'
        else:
            style_name = 'stylers.xml'

        style_path = os.path.join(style_dir, style_name)
        gui_obj.attrib['path'] = style_path

    if gui_obj.attrib['name'] == 'DarkMode':
        if gui_obj.attrib['enable'] == 'no':
            gui_obj.attrib['enable'] = 'yes'
        else:
            gui_obj.attrib['enable'] = 'no'

xml_tree.write(xml_path)
# print()

