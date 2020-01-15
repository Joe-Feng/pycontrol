from xml.dom.minidom import Document




def writeVOCxml(filename, elements):
    """
    elements:
        elements = {
            'filename': '1.jpg',
            'width': 239,
            'height': 300,
            'depth': 3,
            'object': [
                {
                    'name': 'safety_hat',
                    'bndbox': [60,66,910,1108]
                },
                {
                    'name': 'no_safety_hat',
                    'bndbox': [13,67,356,112]
                }
            ]
        }
    """
    tags = {
        'folder': 'Unknown',
        'filename': 'Unknown',
        'path': 'Unknown',
        'database': 'Unknown',
        'width': 'Unknown',
        'height': 'Unknown',
        'depth': 'Unknown',
        'segmented': '0',
        'object': 'Unknown'
    }

    for key in elements.keys():
        if key in tags:
            tags[key] = elements[key]
        else:
            raise KeyError("Unknown element: " + key)


    doc = Document()

    # annotation
    root = doc.createElement('annotation')
    doc.appendChild(root)

    # folder
    node_folder = doc.createElement('folder')
    node_folder.appendChild(doc.createTextNode(tags['folder']))

    # filename
    node_filename = doc.createElement('filename')
    node_filename.appendChild(doc.createTextNode(tags['filename']))

    # path
    node_path = doc.createElement('path')
    node_path.appendChild(doc.createTextNode(tags['path']))

    # source
    node_source = doc.createElement('source')
    # database
    node_database = doc.createElement('database')
    node_database.appendChild(doc.createTextNode(tags['database']))
    node_source.appendChild(node_database)

    # size
    node_size = doc.createElement('size')
    # width
    node_width = doc.createElement('width')
    node_width.appendChild(doc.createTextNode(str(tags['width'])))
    node_size.appendChild(node_width)
    # height
    node_height = doc.createElement('height')
    node_height.appendChild(doc.createTextNode(str(tags['height'])))
    node_size.appendChild(node_height)
    # depth
    node_depth = doc.createElement('depth')
    node_depth.appendChild(doc.createTextNode(str(tags['depth'])))
    node_size.appendChild(node_depth)


    root.appendChild(node_folder)
    root.appendChild(node_filename)
    root.appendChild(node_path)
    root.appendChild(node_source)
    root.appendChild(node_size)


    # object
    objects = tags['object']
    for object in objects:
        node_object = doc.createElement('object')
        # name
        node_name = doc.createElement('name')
        node_name.appendChild(doc.createTextNode(object['name']))
        node_object.appendChild(node_name)
        # pose
        node_pose = doc.createElement('pose')
        node_pose.appendChild(doc.createTextNode(object.get('pose', 'Unspecified')))
        node_object.appendChild(node_pose)
        # truncated
        node_truncated = doc.createElement('truncated')
        node_truncated.appendChild(doc.createTextNode(str(object.get('truncated', '0'))))
        node_object.appendChild(node_truncated)
        # difficult
        node_difficult = doc.createElement('difficult')
        node_difficult.appendChild(doc.createTextNode(str(object.get('difficult', '0'))))
        node_object.appendChild(node_difficult)

        # boundingbox
        node_bndbox = doc.createElement('bndbox')
        # xmin
        node_xmin = doc.createElement('xmin')
        node_xmin.appendChild(doc.createTextNode(str(int(object['bndbox'][0]))))
        node_bndbox.appendChild(node_xmin)
        # ymin
        node_ymin = doc.createElement('ymin')
        node_ymin.appendChild(doc.createTextNode(str(int(object['bndbox'][1]))))
        node_bndbox.appendChild(node_ymin)
        # xmax
        node_xmax = doc.createElement('xmax')
        node_xmax.appendChild(doc.createTextNode(str(int(object['bndbox'][2]))))
        node_bndbox.appendChild(node_xmax)
        # ymax
        node_ymax = doc.createElement('ymax')
        node_ymax.appendChild(doc.createTextNode(str(int(object['bndbox'][3]))))
        node_bndbox.appendChild(node_ymax)

        node_object.appendChild(node_bndbox)
        root.appendChild(node_object)


    with open(filename, 'w') as f:
        doc.writexml(f, indent='\t', addindent='\t', newl='\n')



        