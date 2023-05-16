import omero.gateway as gw
from omero import model
from omero.model import enums, LengthI
from omero import grid
from random import choice
from string import ascii_letters

import numpy as np

DTYPES_NP_TO_OMERO = {'int8': enums.PixelsTypeint8,
                      'int16': enums.PixelsTypeint16,
                      'uint16': enums.PixelsTypeuint16,
                      'int32': enums.PixelsTypeint32,
                      'float_': enums.PixelsTypefloat,
                      'float8': enums.PixelsTypefloat,
                      'float16': enums.PixelsTypefloat,
                      'float32': enums.PixelsTypefloat,
                      'float64': enums.PixelsTypedouble,
                      'complex_': enums.PixelsTypecomplex,
                      'complex64': enums.PixelsTypecomplex}

DTYPES_OMERO_TO_NP = {enums.PixelsTypeint8: 'int8',
                      enums.PixelsTypeuint8: 'uint8',
                      enums.PixelsTypeint16: 'int16',
                      enums.PixelsTypeuint16: 'uint16',
                      enums.PixelsTypeint32: 'int32',
                      enums.PixelsTypeuint32: 'uint32',
                      enums.PixelsTypefloat: 'float32',
                      enums.PixelsTypedouble: 'double'}


COLUMN_TYPES = {'string': grid.StringColumn,
                'long': grid.LongColumn,
                'bool': grid.BoolColumn,
                'double': grid.DoubleColumn,
                'long_array': grid.LongArrayColumn,
                'float_array': grid.FloatArrayColumn,
                'double_array': grid.DoubleArrayColumn,
                'image': grid.ImageColumn,
                'dataset': grid.DatasetColumn,
                'plate': grid.PlateColumn,
                'well': grid.WellColumn,
                'roi': grid.RoiColumn,
                'mask': grid.MaskColumn,
                'file': grid.FileColumn,
                }


def _create_column(data_type, kwargs):
    column_class = COLUMN_TYPES[data_type]

    return column_class(**kwargs)


def _create_table(column_names, columns_descriptions, values, types=None):
    # validate lengths
    if not len(column_names) == len(columns_descriptions) == len(values):
        raise IndexError('Error creating table. Names, description and values not matching or empty.')
    if types is not None and len(types) != len(values):
        raise IndexError('Error creating table. Types and values lengths are not matching.')
    # TODO: Verify implementation of empty table creation

    columns = []
    for i, (cn, cd, v) in enumerate(zip(column_names, columns_descriptions, values)):
        # Verify column names and descriptions are strings
        if not type(cn) == type(cd) == str:
            raise TypeError(f'Types of column name ({type(cn)}) or description ({type(cd)}) is not string')

        if types is not None:
            v_type = types[i]
        else:
            v_type = [type(v[0][0])] if isinstance(v[0], (list, tuple)) else type(v[0])
        # Verify that all elements in values are the same type
        # if not all(isinstance(x, v_type) for x in v):
        #     raise TypeError(f'Not all elements in column {cn} are of the same type')
        if v_type == str:
            size = len(max(v, key=len)) * 2  # We assume here that the max size is double of what we really have...
            args = {'name': cn, 'description': cd, 'size': size, 'values': v}
            columns.append(_create_column(data_type='string', kwargs=args))
        elif v_type == int:
            if cn.lower() in ["image", "imageid", "image id", "image_id"]:
                args = {'name': cn, 'description': cd, 'values': v}
                columns.append(_create_column(data_type='image', kwargs=args))
            elif cn.lower() in ["dataset", "datasetid", "dataset id", "dataset_id"]:
                args = {'name': cn, 'description': cd, 'values': v}
                columns.append(_create_column(data_type='dataset', kwargs=args))
            elif cn.lower() in ["plate", "plateid", "plate id", "plate_id"]:
                args = {'name': cn, 'description': cd, 'values': v}
                columns.append(_create_column(data_type='plate', kwargs=args))
            elif cn.lower() in ["well", "wellid", "well id", "well_id"]:
                args = {'name': cn, 'description': cd, 'values': v}
                columns.append(_create_column(data_type='well', kwargs=args))
            elif cn.lower() in ["roi", "roiid", "roi id", "roi_id"]:
                args = {'name': cn, 'description': cd, 'values': v}
                columns.append(_create_column(data_type='roi', kwargs=args))
            elif cn.lower() in ["mask", "maskid", "mask id", "mask_id"]:
                args = {'name': cn, 'description': cd, 'values': v}
                columns.append(_create_column(data_type='mask', kwargs=args))
            elif cn.lower() in ["file", "fileid", "file id", "file_id"]:
                args = {'name': cn, 'description': cd, 'values': v}
                columns.append(_create_column(data_type='file', kwargs=args))
            else:
                args = {'name': cn, 'description': cd, 'values': v}
                columns.append(_create_column(data_type='long', kwargs=args))
        elif v_type == float:
            args = {'name': cn, 'description': cd, 'values': v}
            columns.append(_create_column(data_type='double', kwargs=args))
        elif v_type == bool:
            args = {'name': cn, 'description': cd, 'values': v}
            columns.append(_create_column(data_type='string', kwargs=args))
        elif v_type in [gw.ImageWrapper, model.ImageI]:
            args = {'name': cn, 'description': cd, 'values': [img.getId() for img in v]}
            columns.append(_create_column(data_type='image', kwargs=args))
        elif v_type in [gw.RoiWrapper, model.RoiI]:
            args = {'name': cn, 'description': cd, 'values': [roi.getId() for roi in v]}
            columns.append(_create_column(data_type='roi', kwargs=args))
        elif isinstance(v_type, (list, tuple)):  # We are creating array columns

            # Verify that every element in the 'array' is the same length and type
            if any(len(x) != len(v[0]) for x in v):
                raise IndexError(f'Not all elements in column {cn} have the same length')
            if not all(all(isinstance(x, type(v[0][0])) for x in a) for a in v):
                raise TypeError(f'Not all the elements in the array column {cn} are of the same type')

            args = {'name': cn, 'description': cd, 'size': len(v[0]), 'values': v}
            if v_type[0] == int:
                columns.append(_create_column(data_type='long_array', kwargs=args))
            elif v_type[0] == float:  # We are casting all floats to doubles
                columns.append(_create_column(data_type='double_array', kwargs=args))
            else:
                raise TypeError(f'Error on column {cn}. Datatype not implemented for array columns')
        else:
            raise TypeError(f'Could not detect column datatype for column {cn}')

    return columns


def create_annotation_table(connection, table_name, column_names, column_descriptions, values, types=None, namespace=None,
                            table_description=None):
    """Creates a table annotation from a list of lists"""

    column_length = len(values[0])
    if any(len(l) != column_length for l in values):
        raise ValueError('The columns have different lengths')

    table_name = f'{table_name}_{"".join([choice(ascii_letters) for _ in range(32)])}.h5'

    columns = _create_table(column_names=column_names,
                            columns_descriptions=column_descriptions,
                            values=values,
                            types=types)
    resources = connection.c.sf.sharedResources()
    repository_id = resources.repositories().descriptions[0].getId().getValue()
    table = resources.newTable(repository_id, table_name)
    table.initialize(columns)
    table.addData(columns)

    original_file = table.getOriginalFile()
    table.close()  # when we are done, close.
    file_ann = gw.FileAnnotationWrapper(connection)
    file_ann.setNs(namespace)
    file_ann.setDescription(table_description)
    file_ann.setFile(model.OriginalFileI(original_file.id.val, False))  # TODO: try to get this with a wrapper
    file_ann.save()
    return file_ann


def link_annotation(object_wrapper, annotation_wrapper):
    object_wrapper.linkAnnotation(annotation_wrapper)
