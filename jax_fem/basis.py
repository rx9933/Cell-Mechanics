import basix
import numpy as onp
import os
import gmsh
import meshio

from jax_fem import logger


# def get_full_integration_poly_degree(ele_type, lag_order, dim):
#     """Only works for weak forms of (grad_u, grad_v).
#     TODO: Is this correct?
#     Reference:
#     https://zhuanlan.zhihu.com/p/521630645
#     """
#     if ele_type == 'hexahedron' or ele_type == 'quadrilateral':
#         return 2 * (dim*lag_order - 1)

#     if ele_type == 'tetrahedron' or ele_type == 'triangle':
#         return 2 * (dim*(lag_order - 1) - 1)

def get_elements(ele_type):
    """Mesh node ordering is important.
    If the input mesh file is Gmsh .msh or Abaqus .inp, meshio would convert it to
    its own ordering. My experience shows that meshio ordering is the same as Abaqus.
    For example, for a 10-node tetrahedron element, the ordering of meshio is the following
    https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node33.html
    The troublesome thing is that basix has a different ordering. As shown below
    https://defelement.com/elements/lagrange.html
    The consequence is that we need to define this "re_order" variable to make sure the
    ordering is correct.
    """
    element_family = basix.ElementFamily.P
    if ele_type == 'HEX8':
        re_order = [0, 1, 3, 2, 4, 5, 7, 6]
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 3 # 2x2x2, TODO: is this full integration?
        degree = 1
    elif ele_type == 'HEX27':
        print(f"Warning: 27-node hexahedron is rarely used in practice and not recommended.")
        re_order = [0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19,
                    17, 10, 12, 15, 14, 22, 23, 21, 24, 20, 25, 26]
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 6 # 6x6x6, full integration
        degree = 2
    elif ele_type == 'HEX20':
        re_order = [0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19,
                    17, 10, 12, 15, 14]
        element_family = basix.ElementFamily.serendipity
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 2 # 6x6x6, full integration
        degree = 2
    elif ele_type == 'HEX64':
        print(f"Warning: 64-node hexahedron is currently in development and is not guaranteed to be correct.")
        #TODO: this re_order variable is not correct, we need to see the algorithm basix uses to order dofs and the algorithm meshio uses
        re_order = [0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 19, 18, 22, 23, 20, 21, 24, 25,
                    26, 27, 28, 29, 31, 30, 32, 34, 35, 33, 36, 37,
                    39, 38, 40, 42, 43, 41, 44, 45, 47, 46, 49, 48,
                    50, 51, 52, 53, 55, 54, 56, 57, 59, 58, 60, 61,
                    63, 62]
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 9
        degree = 3
    elif ele_type == 'HEX125':
        print(f"Warning: 125-node hexahedron is currently in development and is not guaranteed to be correct.")
        #TODO: this re_order variable is not correct, we need to see the algorithm basix uses to order dofs and the algorithm meshio uses
        re_order = [0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 24, 23,
                    29, 30, 31, 26, 27, 28, 32, 33, 34, 35, 36, 37,
                    38, 39, 40, 43, 42, 41, 44, 50, 52, 46, 47, 51,
                    49, 45, 48, 53, 55, 61, 59, 54, 58, 60, 56, 57,
                    62, 68, 70, 64, 65, 69, 67, 63, 66, 71, 73, 79,
                    77, 72, 76, 78, 74, 75, 82, 80, 86, 88, 81, 83,
                    87, 85, 84, 89, 91, 97, 95, 90, 94, 96, 92, 93,
                    98, 100, 106, 104, 116, 118, 124, 122, 99, 101,
                    107, 103, 109, 105, 115, 113, 117, 119, 121, 123,
                    102, 108, 110, 112, 114, 120, 111]
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 12
        degree = 4
    elif ele_type == 'HEX216':
        print(f"Warning: 216-node hexahedron is currently in development and is not guaranteed to be correct.")
        #TODO: this re_order variable is not correct, we need to see the algorithm basix uses to order dofs and the algorithm meshio uses
        re_order = [0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                    26, 27, 31, 30, 29, 28, 36, 37, 38, 39, 32, 33,
                    34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                    50, 51, 55, 54, 53, 52, 56, 68, 71, 59, 60, 64,
                    69, 70, 67, 63, 58, 57, 61, 65, 66, 62, 72, 75,
                    87, 84, 73, 74, 79, 83, 86, 85, 80, 76, 77, 78,
                    82, 81, 88, 100, 103, 91, 92, 96, 101, 102, 99,
                    95, 90, 89, 93, 97, 98, 94, 104, 107, 119, 116,
                    105, 106, 111, 115, 118, 117, 112, 108, 109, 110,
                    114, 113, 123, 120, 132, 135, 122, 121, 124, 128,
                    133, 134, 131, 127, 126, 125, 129, 130, 136, 139,
                    151, 148, 137, 138, 143, 147, 150, 149, 144, 140,
                    141, 142, 146, 145, 152, 155, 167, 164, 200, 203,
                    215, 212, 153, 154, 156, 160, 168, 184, 159, 163,
                    171, 187, 166, 165, 183, 199, 180, 196, 201, 202,
                    204, 208, 207, 211, 214, 213, 157, 161, 162, 158,
                    169, 170, 186, 185, 172, 188, 192, 176, 175, 179,
                    195, 191, 182, 181, 197, 198, 205, 206, 210, 209,
                    173, 174, 178, 177, 189, 190, 194, 193]
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 15
        degree = 5
    elif ele_type.startswith('GENHEX'):
        degree = int(ele_type[6:])
        re_order = hex_reorder(degree)
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 3*degree
    elif ele_type == 'TET4':
        re_order = [0, 1, 2, 3]
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 0 # 1, full integration
        degree = 1
    elif ele_type == 'TET10':
        re_order = [0, 1, 2, 3, 9, 6, 8, 7, 5, 4]
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 2 # 4, full integration
        degree = 2
    # TODO: Check if this is correct.
    elif ele_type == 'QUAD4':
        re_order = [0, 1, 3, 2]
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
        degree = 1
    elif ele_type == 'QUAD8':
        re_order = [0, 1, 3, 2, 4, 6, 7, 5]
        element_family = basix.ElementFamily.serendipity
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
        degree = 2
    elif ele_type == 'TRI3':
        re_order = [0, 1, 2]
        basix_ele = basix.CellType.triangle
        basix_face_ele = basix.CellType.interval
        gauss_order = 0 # 1, full integration
        degree = 1
    elif  ele_type == 'TRI6':
        re_order = [0, 1, 2, 5, 3, 4]
        basix_ele = basix.CellType.triangle
        basix_face_ele = basix.CellType.interval
        gauss_order = 2 # 3, full integration
        degree = 2
    else:
        raise NotImplementedError

    return element_family, basix_ele, basix_face_ele, gauss_order, degree, re_order

def hex_reorder(degree):
    element_family = basix.ElementFamily.P
    basix_ele = basix.CellType.hexahedron
    lagrange_variant = basix.LagrangeVariant.equispaced
    element = basix.create_element(element_family, basix_ele, degree, lagrange_variant)
    ele_type = 'HEX'+str((degree+1)**3)
    if degree ==1:
        cell_type = 'hexahedron'
    else:
        cell_type = 'hexahedron'+str((degree+1)**3)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    msh_dir = os.path.join(data_dir, 'msh')
    os.makedirs(msh_dir, exist_ok=True)
    msh_file = os.path.join(msh_dir, 'box.msh')

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # save in old MSH format
    p = gmsh.model.geo.addPoint(0.,0.,0.)
    l = gmsh.model.geo.extrude([(0, p)], 1., 0, 0, [1], [1])
    s = gmsh.model.geo.extrude([l[1]], 0, 1., 0, [1], [1], recombine=True)
    v = gmsh.model.geo.extrude([s[1]], 0, 0, 1., [1], [1], recombine=True)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(degree)
    gmsh.write(msh_file)
    gmsh.finalize()

    mesh = meshio.read(msh_file)
    os.remove(msh_file)
    points = mesh.points # (num_total_nodes, dim)
    cells =  mesh.cells_dict[cell_type] # (num_cells, num_nodes)

    a = points[cells[0]]
    b = element.points
    reorder = [int(onp.nonzero(onp.isclose(row, b).all(-1))[0][0]) for row in a]
    return reorder

def reorder_inds(inds, re_order):
    new_inds = []
    for ind in inds.reshape(-1):
        new_inds.append(onp.argwhere(re_order == ind))
    new_inds = onp.array(new_inds).reshape(inds.shape)
    return new_inds


def get_shape_vals_and_grads(ele_type, gauss_order=None):
    """TODO: Add comments

    Returns
    -------
    shape_values: ndarray
        (8, 8) = (num_quads, num_nodes)
    shape_grads_ref: ndarray
        (8, 8, 3) = (num_quads, num_nodes, dim)
    weights: ndarray
        (8,) = (num_quads,)
    """
    element_family, basix_ele, basix_face_ele, gauss_order_default, degree, re_order = get_elements(ele_type)

    if gauss_order is None:
        gauss_order = gauss_order_default

    quad_points, weights = basix.make_quadrature(basix_ele, gauss_order)
    if degree >2:
        element = basix.create_element(element_family, basix_ele, degree, lagrange_variant = basix.LagrangeVariant.equispaced)
    else:
        element = basix.create_element(element_family, basix_ele, degree)
    vals_and_grads = element.tabulate(1, quad_points)[:, :, re_order, :]
    #print(type(quad_points))
    shape_values = vals_and_grads[0, :, :, 0]
    shape_grads_ref = onp.transpose(vals_and_grads[1:, :, :, 0], axes=(1, 2, 0))
    logger.debug(f"ele_type = {ele_type}, quad_points.shape = (num_quads, dim) = {quad_points.shape}")
    return shape_values, shape_grads_ref, weights


def get_face_shape_vals_and_grads(ele_type, gauss_order=None):
    """TODO: Add comments

    Returns
    -------
    face_shape_vals: ndarray
        (6, 4, 8) = (num_faces, num_face_quads, num_nodes)
    face_shape_grads_ref: ndarray
        (6, 4, 3) = (num_faces, num_face_quads, num_nodes, dim)
    face_weights: ndarray
        (6, 4) = (num_faces, num_face_quads)
    face_normals:ndarray
        (6, 3) = (num_faces, dim)
    face_inds: ndarray
        (6, 4) = (num_faces, num_face_vertices)
    """
    element_family, basix_ele, basix_face_ele, gauss_order_default, degree, re_order = get_elements(ele_type)

    if gauss_order is None:
        gauss_order = gauss_order_default

    # TODO: Check if this is correct.
    # We should provide freedom for seperate gauss_order for volume integral and surface integral
    # Currently, they're using the same gauss_order!
    points, weights = basix.make_quadrature(basix_face_ele, gauss_order)

    map_degree = 1
    lagrange_map = basix.create_element(basix.ElementFamily.P, basix_face_ele, map_degree)
    values = lagrange_map.tabulate(0, points)[0, :, :, 0]
    vertices = basix.geometry(basix_ele)
    dim = len(vertices[0])
    facets = basix.cell.sub_entity_connectivity(basix_ele)[dim - 1]
    # Map face points
    # Reference: https://docs.fenicsproject.org/basix/main/python/demo/demo_facet_integral.py.html
    face_quad_points = []
    face_inds = []
    face_weights = []
    for f, facet in enumerate(facets):
        mapped_points = []
        for i in range(len(points)):
            vals = values[i]
            mapped_point = onp.sum(vertices[facet[0]] * vals[:, None], axis=0)
            mapped_points.append(mapped_point)
        face_quad_points.append(mapped_points)
        face_inds.append(facet[0])
        jacobian = basix.cell.facet_jacobians(basix_ele)[f]
        if dim == 2:
            size_jacobian = onp.linalg.norm(jacobian)
        else:
            size_jacobian = onp.linalg.norm(onp.cross(jacobian[:, 0], jacobian[:, 1]))
        face_weights.append(weights*size_jacobian)
    face_quad_points = onp.stack(face_quad_points)
    face_weights = onp.stack(face_weights)

    face_normals = basix.cell.facet_outward_normals(basix_ele)
    face_inds = onp.array(face_inds)
    face_inds = reorder_inds(face_inds, re_order)
    num_faces, num_face_quads, dim = face_quad_points.shape
    if degree >2:
        element = basix.create_element(element_family, basix_ele, degree, lagrange_variant = basix.LagrangeVariant.equispaced)
    else:
        element = basix.create_element(element_family, basix_ele, degree)
    vals_and_grads = element.tabulate(1, face_quad_points.reshape(-1, dim))[:, :, re_order, :]
    face_shape_vals = vals_and_grads[0, :, :, 0].reshape(num_faces, num_face_quads, -1)
    face_shape_grads_ref = vals_and_grads[1:, :, :, 0].reshape(dim, num_faces, num_face_quads, -1)
    face_shape_grads_ref = onp.transpose(face_shape_grads_ref, axes=(1, 2, 3, 0))
    logger.debug(f"face_quad_points.shape = (num_faces, num_face_quads, dim) = {face_quad_points.shape}")
    return face_shape_vals, face_shape_grads_ref, face_weights, face_normals, face_inds
