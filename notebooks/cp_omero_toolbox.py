import tempfile

import omero.gateway as gw
from omero import model
from omero.model import enums, LengthI, MaskI, RoiI
from omero import grid
from random import choice
from string import ascii_letters

import omero_rois
from omero.gateway import ColorHolder
from omero.rtypes import rstring, rdouble, rint

import cellprofiler_core.pipeline as cp_pipeline
import cellprofiler_core.measurement as cp_measurement
from cellprofiler_core.modules.injectimage import InjectImage

import ezomero

import numpy as np
import pandas as pd

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

def run_cp_pipeline(conn: gw,
                    dataset_id: tempfile.TemporaryDirectory,
                    objects_to_image_table: str = None,
                    objects_to_mask: iter = None,
                    objects_to_point: iter = None,
                    link_to_project: bool = False,
                    output_dir: str = None,
                    input_dir: str = None):

    file_ann_ids = ezomero.get_file_annotation_ids(conn, "Dataset", dataset_id)
    for file_ann_id in file_ann_ids:
        if conn.getObject("FileAnnotation", file_ann_id).getFile().getName().endswith(".cppipe"):
            cp_pipeline_path = ezomero.get_file_annotation(conn, file_ann_id, input_dir.name)
            print(f"Downloaded {cp_pipeline_path}")
            break

    pipeline = cp_pipeline.Pipeline()
    pipeline.load(cp_pipeline_path)

    for i in range(4):
        print('Remove module: ', pipeline.modules()[0].module_name)
        pipeline.remove_module(1)

    print('Pipeline modules:')
    for module in pipeline.modules(False):
        print(module.module_num, module.module_name)

    measurement_dfs = {}
    for column in pipeline.get_measurement_columns():
        if column[0] not in measurement_dfs.keys():
            measurement_dfs[column[0]] = pd.DataFrame()

    # Get the ids of the images we want to analyze
    dataset = conn.getObject("Dataset", dataset_id)
    image_ids = ezomero.get_image_ids(conn=conn, dataset=dataset_id, across_groups=False)

    # Remove duplicates
    objects_to_mask = set(objects_to_mask)
    objects_to_point = set(objects_to_point)

    # Let's collect all images in a dataset and feed them one at a time into the pipeline.
    for image_id in image_ids:
        image, image_pixels = ezomero.get_image(conn, image_id)

        pipeline_copy = pipeline.copy()

        for c in range(image.getSizeC()):
            inject_image_module = InjectImage(f"ch{c}", image_pixels[..., c].squeeze())
            inject_image_module.set_module_num(1)
            pipeline_copy.add_module(inject_image_module)

        measurements = pipeline_copy.run()

        for object_name, _ in measurement_dfs.items():
            if object_name == "Experiment":
                continue  # TODO: put this into description or comment

            data = {feature: measurements.get_measurement(object_name, feature) for feature in
                    measurements.get_feature_names(object_name)}

            if object_name in objects_to_point:
                data["Roi"] = np.zeros(shape=(len(data["Number_Object_Number"])), dtype="int")

                for i, object_nr in enumerate(data["Number_Object_Number"]):
                    point = ezomero.rois.Point(data["Location_Center_X"][i],
                                               data["Location_Center_Y"][i],
                                               data["Location_Center_Z"][i],
                                               label=f"{object_name}_{object_nr}")
                    point_id = ezomero.post_roi(conn=conn,
                                                image_id=image_id,
                                                shapes=[point],
                                                name=None,
                                                description=None)
                    data["Roi"][i] = point_id

            if object_name in objects_to_mask:
                data["Roi"] = np.zeros(shape=(len(data["Number_Object_Number"])), dtype="int")

                labels = np.load(f"{output_dir.name}/{object_name}.npy")
                if labels.ndim == 2:
                    labels = np.expand_dims(labels, 0)

                masks = masks_from_labels_image_3d(labels,
                                                   rgba=(0, 255, 0, 100),
                                                   text=object_name)
                for i, mask in masks.items():
                    roi_id = create_roi(conn=conn,
                                        img=image,
                                        shapes=mask,
                                        name=None)
                    data["Roi"][i - 1] = roi_id

            if object_name == objects_to_image_table:
                data["Image"] = np.full(shape=(len(data["Number_Object_Number"])), fill_value=image_id)
                data["Dataset"] = np.full(shape=(len(data["Number_Object_Number"])), fill_value=dataset_id)

                objects_table = create_annotation_table(conn, f"{object_name}_table",
                                                        column_names=[k for k in data.keys()],
                                                        column_descriptions=["" for _ in data.keys()],
                                                        values=[v.tolist() for v in data.values()],
                                                        types=None,
                                                        namespace="CellProfiler_v4.2.5",
                                                        table_description=f"{object_name}_table"
                                                        )
                link_annotation(image, objects_table)

            if object_name == "Image":
                data["Image"] = np.full((1,), image_id)
                data["Dataset"] = np.full((1,), dataset_id)

            measurement_dfs[object_name] = pd.concat([measurement_dfs[object_name], pd.DataFrame.from_dict(data)],
                                                     ignore_index=True)

    images_table = create_annotation_table(conn, "images_table",
                                           column_names=measurement_dfs[objects_to_image_table].columns.tolist(),
                                           column_descriptions=measurement_dfs[objects_to_image_table].columns.tolist(),
                                           values=[measurement_dfs[objects_to_image_table][c].values.tolist() for c in
                                                   measurement_dfs[objects_to_image_table].columns],
                                           types=None,
                                           namespace="CellProfiler_v4.2.5",
                                           table_description="images_table"
                                           )
    if link_to_project:
        project = dataset.getParent()
        link_annotation(project, images_table)
    else:
        link_annotation(dataset, images_table)

    return measurement_dfs

def create_roi(conn, img, shapes, name):
    updateService = conn.getUpdateService()

    # create an ROI, link it to Image
    roi = RoiI()
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(img._obj)
    roi.setName(rstring(name))
    for shape in shapes:
        # shape.setTextValue(rstring(name))
        roi.addShape(shape)
    # Save the ROI (saves any linked shapes too)
    return updateService.saveAndReturnObject(roi).id.val


def masks_from_labels_image_3d(
        labels_3d, rgba=None, c=None, t=None, text=None,
        raise_on_no_mask=True):  # sourcery skip: low-code-quality
    """
    Create a mask shape from a binary image (background=0)

    :param numpy.array labels_3d: labels 3D array
    :param rgba int-4-tuple: Optional (red, green, blue, alpha) colour
    :param c: Optional C-index for the mask
    :param t: Optional T-index for the mask
    :param text: Optional text for the mask
    :param raise_on_no_mask: If True (default) throw an exception if no mask
           found, otherwise return an empty Mask
    :return: An OMERO mask
    :raises NoMaskFound: If no labels were found
    :raises InvalidBinaryImage: If the maximum labels is greater than 1
    """
    shapes = {}
    for i in range(1, labels_3d.max() + 1):
        if not np.any(labels_3d == i):
            continue

        masks = []
        bin_img = labels_3d == i
        # Find bounding box to minimise size of mask
        xmask = bin_img.sum(0).sum(0).nonzero()[0]
        ymask = bin_img.sum(0).sum(1).nonzero()[0]
        if any(xmask) and any(ymask):
            x0 = min(xmask)
            w = max(xmask) - x0 + 1
            y0 = min(ymask)
            h = max(ymask) - y0 + 1
            submask = bin_img[:, y0:(y0 + h), x0:(x0 + w)]
        else:
            if raise_on_no_mask:
                raise omero_rois.NoMaskFound()
            x0 = 0
            w = 0
            y0 = 0
            h = 0
            submask = []

        for z, plane in enumerate(submask):
            if np.any(plane):
                mask = MaskI()
                mask.setBytes(np.packbits(np.asarray(plane, dtype=int)))
                mask.setWidth(rdouble(w))
                mask.setHeight(rdouble(h))
                mask.setX(rdouble(x0))
                mask.setY(rdouble(y0))
                mask.setTheZ(rint(z))

                if rgba is not None:
                    ch = ColorHolder.fromRGBA(*rgba)
                    mask.setFillColor(rint(ch.getInt()))
                if c is not None:
                    mask.setTheC(rint(c))
                if t is not None:
                    mask.setTheT(rint(t))
                if text is not None:
                    mask.setTextValue(rstring(f"{text}_{i}"))

                masks.append(mask)

        shapes[i] = masks

    return shapes


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
