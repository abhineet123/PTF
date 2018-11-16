def get_interfaces_with_mac_addresses(interface_name_substring=''):
    import subprocess
    import xml.etree.ElementTree

    cmd = 'wmic.exe nic'
    if interface_name_substring:
        cmd += ' where "name like \'%%%s%%\'" ' % interface_name_substring
    cmd += ' get /format:rawxml'

    DETACHED_PROCESS = 8
    xml_text = subprocess.check_output(cmd, creationflags=DETACHED_PROCESS)

    # convert xml text to xml structure
    xml_root = xml.etree.ElementTree.fromstring(xml_text)

    xml_types = dict(
        datetime=str,
        boolean=bool,
        uint16=int,
        uint32=int,
        uint64=int,
        string=str,
    )


    def xml_to_dict(xml_node):
        """ Convert the xml returned from wmic to a dict """
        dict_ = {}
        for child in xml_node:
            name = child.attrib['NAME']
            xml_type = xml_types[child.attrib['TYPE']]

            if child.tag == 'PROPERTY':
                if len(child):
                    for value in child:
                        dict_[name] = xml_type(value.text)
            elif child.tag == 'PROPERTY.ARRAY':
                if len(child):
                    assert False, "This case is not dealt with"
            else:
                assert False, "This case is not dealt with"

        return dict_


    # convert the xml into a list of dict for each interface
    interfaces = [xml_to_dict(x)
                  for x in xml_root.findall("./RESULTS/CIM/INSTANCE")]

    # get only the interfaces which have a mac address
    interfaces_with_mac = [
        intf for intf in interfaces if intf.get('MACAddress')]

    return interfaces_with_mac

for intf in get_interfaces_with_mac_addresses('Realtek'):
    print intf['Name'], intf['MACAddress']
