import OpenGL.GL as GL              # standard Python OpenGL wrapper

import numpy as np                  # all matrix manipulations & OpenGL args
import pyassimp                     # 3D ressource loader
import pyassimp.errors              # assimp error management + exceptions


from SceneNode import Node, SkinningControlNode
from Shader import Shader

from transform import translate, scale, identity, Trackball, sincos
from transform import (lerp, quaternion_slerp, quaternion_matrix, quaternion,
                       quaternion_from_euler, vec)

# ------------  simple color fragment shader demonstrated in Practical 1 ------
with open('../data/Shader/shader1_VS.glsl', 'r') as myfile:
    COLOR_VERT=myfile.read()

with open('../data/Shader/shader1_FS.glsl', 'r') as myfile:
    COLOR_FRAG=myfile.read()

# -------------- Linear Blend Skinning : TP7 ---------------------------------
MAX_VERTEX_BONES = 4
MAX_BONES = 128

# new shader for skinned meshes, fully compatible with previous color fragment
# TODO: complete the loop for TP7 exercise 1
SKINNING_VERT = """#version 330 core
// ---- camera geometry
uniform mat4 projection, view;

// ---- skinning globals and attributes
const int MAX_VERTEX_BONES=%d, MAX_BONES=%d;
uniform mat4 boneMatrix[MAX_BONES];

// ---- vertex attributes
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec4 bone_ids;
layout(location = 3) in vec4 bone_weights;

// ----- interpolated attribute variables to be passed to fragment shader
out vec3 fragColor;

void main() {

    // ------ creation of the skinning deformation matrix
    mat4 skinMatrix = mat4(0.);  // TODO complete shader here for exercise 1!

    for (int i = 0; i < MAX_VERTEX_BONES; i++)
    {
        skinMatrix += boneMatrix[int(bone_ids[i])] * bone_weights[i];
    }

    // ------ compute world and normalized eye coordinates of our vertex
    vec4 wPosition4 = skinMatrix * vec4(position, 1.0);
    gl_Position = projection * view * wPosition4;
    
    fragColor = color;
}
""" % (MAX_VERTEX_BONES, MAX_BONES)

class VertexArray:
    """helper class to create and self destroy vertex array objects."""
    def __init__(self, attributes, index=None, usage=GL.GL_STATIC_DRAW):
        """ Vertex array from attributes and optional index array. Vertex
            attribs should be list of arrays with dim(0) indexed by vertex. """

        # create vertex array object, bind it
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []  # we will store buffers in a list
        nb_primitives, size = 0, 0

        # load a buffer per initialized vertex attribute (=dictionary)
        for loc, data in enumerate(attributes):
            if data is None:
                continue

            # bind a new vbo, upload its data to GPU, declare its size and type
            self.buffers += [GL.glGenBuffers(1)]
            data = np.array(data, np.float32, copy=False)
            nb_primitives, size = data.shape
            GL.glEnableVertexAttribArray(loc)  # activates for current vao only
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data, usage)
            GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, 0, None)

        # optionally create and upload an index buffer for this object
        self.draw_command = GL.glDrawArrays
        self.arguments = (0, nb_primitives)
        if index is not None:
            self.buffers += [GL.glGenBuffers(1)]
            index_buffer = np.array(index, np.int32, copy=False)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer, usage)
            self.draw_command = GL.glDrawElements
            self.arguments = (index_buffer.size, GL.GL_UNSIGNED_INT, None)

        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def draw(self, primitive):
        """draw a vertex array, either as direct array or indexed array"""
        GL.glBindVertexArray(self.glid)
        self.draw_command(primitive, *self.arguments)
        GL.glBindVertexArray(0)

    def __del__(self):  # object dies => kill GL array and buffers from GPU
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(len(self.buffers), self.buffers)

class SkinnedMesh:
    """class of skinned mesh nodes in scene graph """
    def __init__(self, attributes, bone_nodes, bone_offsets, index=None):

        # setup shader attributes for linear blend skinning shader
        self.vertex_array = VertexArray(attributes, index)

        # feel free to move this up in Viewer as shown in previous practicals
        self.skinning_shader = Shader(SKINNING_VERT, COLOR_FRAG)

        # store skinning data
        self.bone_nodes = bone_nodes
        self.bone_offsets = bone_offsets

    def draw(self, projection, view, _model, **_kwargs):
        """ skinning object draw method """

        shid = self.skinning_shader.glid
        GL.glUseProgram(shid)

        # setup camera geometry parameters
        loc = GL.glGetUniformLocation(shid, 'projection')
        GL.glUniformMatrix4fv(loc, 1, True, projection)
        loc = GL.glGetUniformLocation(shid, 'view')
        GL.glUniformMatrix4fv(loc, 1, True, view)

        # bone world transform matrices need to be passed for skinning
        for bone_id, node in enumerate(self.bone_nodes):
            bone_matrix = node.world_transform @ self.bone_offsets[bone_id]

            bone_loc = GL.glGetUniformLocation(shid, 'boneMatrix[%d]' % bone_id)
            GL.glUniformMatrix4fv(bone_loc, 1, True, bone_matrix)

        # draw mesh vertex array
        self.vertex_array.draw(GL.GL_TRIANGLES)

        # leave with clean OpenGL state, to make it easier to detect problems
        GL.glUseProgram(0)

class ColorMesh:

    def __init__(self, attributes, index=None):
        self.vertex_array = VertexArray(attributes, index)

    # def draw(self, projection, view, model, color_shader):
    def draw(self, projection, view, model, **param):

        color_shader=None
        color=(1, 1, 1, 1)

        names = ['view', 'projection', 'model']
        loc = {n: GL.glGetUniformLocation(color_shader.glid, n) for n in names}
        GL.glUseProgram(color_shader.glid)

        GL.glUniformMatrix4fv(loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(loc['model'], 1, True, model)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        self.vertex_array.draw(GL.GL_TRIANGLES)
        
class Cylinder(Node):
    """ Very simple cylinder based on practical 2 load function """
    def __init__(self):
        super().__init__()
        self.add(*load('cylinder.obj'))  # just load the cylinder from file


# -------------- Deformable Cylinder Mesh  ------------------------------------
class SkinnedCylinder(SkinningControlNode):
    """ Deformable cylinder """
    def __init__(self, sections=11, quarters=20, **params):

        # this "arm" node and its transform serves as control node for bone 0
        # we give it the default identity keyframe transform, doesn't move
        super().__init__({0: (0, 0, 0)}, {0: quaternion()}, {0: 1}, **params)

        # we add a son "forearm" node with animated rotation for the second
        # part of the cylinder
        self.add(SkinningControlNode(
            {0: (0, 0, 0)},
            {0: quaternion(), 2: quaternion_from_euler(90), 4: quaternion()},
            {0: 1}))

        # there are two bones in this animation corresponding to above noes
        bone_nodes = [self, self.children[0]]

        # these bones have no particular offset transform
        bone_offsets = [identity(), identity()]

        # vertices, per vertex bone_ids and weights
        vertices, faces, bone_id, bone_weights = [], [], [], []
        for x_c in range(sections+1):
            for angle in range(quarters):
                # compute vertex coordinates sampled on a cylinder
                z_c, y_c = sincos(360 * angle / quarters)
                vertices.append((x_c - sections/2, y_c, z_c))

                # the index of the 4 prominent bones influencing this vertex.
                # since in this example there are only 2 bones, every vertex
                # is influenced by the two only bones 0 and 1
                bone_id.append((0, 1, 0, 0))

                # per-vertex weights for the 4 most influential bones given in
                # a vec4 vertex attribute. Not using indices 2 & 3 => 0 weight
                # vertex weight is currently a hard transition in the middle
                # of the cylinder
                # TODO: modify weights here for TP7 exercise 2
                weight = 1 if x_c <= sections/2 else 0
                bone_weights.append((weight, 1 - weight, 0, 0))

        # face indices
        faces = []
        for x_c in range(sections):
            for angle in range(quarters):

                # indices of the 4 vertices of the current quad, % helps
                # wrapping to finish the circle sections
                ir0c0 = x_c * quarters + angle
                ir1c0 = (x_c + 1) * quarters + angle
                ir0c1 = x_c * quarters + (angle + 1) % quarters
                ir1c1 = (x_c + 1) * quarters + (angle + 1) % quarters

                # add the 2 corresponding triangles per quad on the cylinder
                faces.extend([(ir0c0, ir0c1, ir1c1), (ir0c0, ir1c1, ir1c0)])

        # the skinned mesh itself. it doesn't matter where in the hierarchy
        # this is added as long as it has the proper bone_node table
        self.add(SkinnedMesh([vertices, bone_weights, bone_id, bone_weights],
                             bone_nodes, bone_offsets, faces))


# -------------- 3D ressource loader -----------------------------------------
def load(file):
    """ load resources from file using pyassimp, return list of ColorMesh """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []  # error reading => return empty list

    meshes = [ColorMesh([m.vertices, m.normals], m.faces) for m in scene.meshes]
    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes



# -------------- 3D resource loader -------------------------------------------
def load_skinned(file):
    """load resources from file using pyassimp, return node hierarchy """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []

    # ----- load animations
    def conv(assimp_keys, ticks_per_second):
        """ Conversion from assimp key struct to our dict representation """
        return {key.time / ticks_per_second: key.value for key in assimp_keys}

    # load first animation in scene file (could be a loop over all animations)
    transform_keyframes = {}
    if scene.animations:
        anim = scene.animations[0]
        for channel in anim.channels:
            # for each animation bone, store trs dict with {times: transforms}
            # (pyassimp name storage bug, bytes instead of str => convert it)
            transform_keyframes[channel.nodename.data.decode('utf-8')] = (
                conv(channel.positionkeys, anim.tickspersecond),
                conv(channel.rotationkeys, anim.tickspersecond),
                conv(channel.scalingkeys, anim.tickspersecond)
            )

    # ---- prepare scene graph nodes
    # create SkinningControlNode for each assimp node.
    # node creation needs to happen first as SkinnedMeshes store an array of
    # these nodes that represent their bone transforms
    nodes = {}  # nodes: string name -> node dictionary

    def make_nodes(pyassimp_node):
        """ Recursively builds nodes for our graph, matching pyassimp nodes """
        trs_keyframes = transform_keyframes.get(pyassimp_node.name, (None,))

        node = SkinningControlNode(*trs_keyframes, name=pyassimp_node.name,
                                   transform=pyassimp_node.transformation)
        nodes[pyassimp_node.name] = node, pyassimp_node
        node.add(*(make_nodes(child) for child in pyassimp_node.children))
        return node

    root_node = make_nodes(scene.rootnode)

    # ---- create SkinnedMesh objects
    for mesh in scene.meshes:
        # -- skinned mesh: weights given per bone => convert per vertex for GPU
        # first, populate an array with MAX_BONES entries per vertex
        v_bone = np.array([[(0, 0)]*MAX_BONES] * mesh.vertices.shape[0],
                          dtype=[('weight', 'f4'), ('id', 'u4')])
        for bone_id, bone in enumerate(mesh.bones[:MAX_BONES]):
            for entry in bone.weights:  # weight,id pairs necessary for sorting
                v_bone[entry.vertexid][bone_id] = (entry.weight, bone_id)

        v_bone.sort(order='weight')             # sort rows, high weights last
        v_bone = v_bone[:, -MAX_VERTEX_BONES:]  # limit bone size, keep highest

        # prepare bone lookup array & offset matrix, indexed by bone index (id)
        bone_nodes = [nodes[bone.name][0] for bone in mesh.bones]
        bone_offsets = [bone.offsetmatrix for bone in mesh.bones]

        # initialize skinned mesh and store in pyassimp_mesh for node addition
        mesh.skinned_mesh = SkinnedMesh(
                [mesh.vertices, mesh.normals, v_bone['id'], v_bone['weight']],
                bone_nodes, bone_offsets, mesh.faces
        )

    # ------ add each mesh to its intended nodes as indicated by assimp
    for final_node, assimp_node in nodes.values():
        final_node.add(*(_mesh.skinned_mesh for _mesh in assimp_node.meshes))

    nb_triangles = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded', file, '\t(%d meshes, %d faces, %d nodes, %d animations)' %
          (len(scene.meshes), nb_triangles, len(nodes), len(scene.animations)))
    pyassimp.release(scene)
    return [root_node]
