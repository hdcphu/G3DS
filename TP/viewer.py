#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args

import pyassimp                     # 3D ressource loader
import pyassimp.errors              # assimp error management + exceptions

from transform import translate, rotate, scale, vec, frustum, perspective
from transform import Trackball, identity

# ------------ low level OpenGL object wrappers ----------------------------
class Shader:
    """ Helper class to create and automatically destroy shader program """
    @staticmethod
    def _compile_shader(src, shader_type):
        src = open(src, 'r').read() if os.path.exists(src) else src
        src = src.decode('ascii') if isinstance(src, bytes) else src
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        src = ('%3d: %s' % (i+1, l) for i, l in enumerate(src.splitlines()))
        if not status:
            log = GL.glGetShaderInfoLog(shader).decode('ascii')
            GL.glDeleteShader(shader)
            src = '\n'.join(src)
            print('Compile failed for %s\n%s\n%s' % (shader_type, log, src))
            return None
        return shader

    def __init__(self, vertex_source, fragment_source):
        """ Shader can be initialized with raw strings or source file names """
        self.glid = None
        vert = self._compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        if vert and frag:
            self.glid = GL.glCreateProgram()  # pylint: disable=E1111
            GL.glAttachShader(self.glid, vert)
            GL.glAttachShader(self.glid, frag)
            GL.glLinkProgram(self.glid)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            status = GL.glGetProgramiv(self.glid, GL.GL_LINK_STATUS)
            if not status:
                print(GL.glGetProgramInfoLog(self.glid).decode('ascii'))
                GL.glDeleteProgram(self.glid)
                self.glid = None

    def __del__(self):
        GL.glUseProgram(0)
        if self.glid:                      # if this is a valid shader object
            GL.glDeleteProgram(self.glid)  # object dies => destroy GL object


# ------------  Simple color shaders ---------------a @ b a---------------------------
COLOR_VERT = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
out vec3 c;
uniform mat4 matrix;

void main() {
    c = color;
    gl_Position = matrix * vec4(position, 1);

}"""

COLOR_FRAG = """#version 330 core
out vec4 outColor;
in vec3 c;
void main() {
    //outColor = vec4(1, 0, 0, 1);
    outColor = vec4(c, 1);
}"""


# ------------  Scene object classes ------------------------------------------
class SimplePyramid:

    def __init__(self):

        # triangle position buffer
        #use_vertex_array
        #    position = np.array(((0, .5, 0), (.5, -.5, 0), (-.5, -.5, 0)), 'f')

        #use index array
        position = np.array(((0, 1.4, 0), (1, 0, 1), (-1, 0, 1), (-1, 0, -1), (1, 0, -1)), np.float32)

        self.index = np.array((0, 2, 1, 0, 1, 4, 0, 4, 3, 0, 2, 3, 3, 4, 1, 3, 1, 2), np.uint32)
        color = np.array(((1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 1, 0)), 'f')

        self.glid = GL.glGenVertexArrays(1)  # create OpenGL vertex array id
        GL.glBindVertexArray(self.glid)      # activate to receive state below
        self.buffers = [GL.glGenBuffers(1)]   # create buffer for position attrib

        # bind the vbo, upload position data to GPU, declare its size and type
        GL.glEnableVertexAttribArray(0)      # assign to layout = 0 attribute
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[0])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, position, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)

        # bind the vbo, upload position data to GPU, declare its size and type
        self.buffers += [GL.glGenBuffers(1)]
        GL.glEnableVertexAttribArray(1)      # assign to layout = 0 attribute
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[1])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, color, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 0, None)

        self.buffers += [GL.glGenBuffers(1)]                                           # create GPU index buffer
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])                  # make it active to receive
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.index, GL.GL_STATIC_DRAW)     # our index array here


        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self.fovy = 1

    def draw(self, projection, view, model, color_shader):
        GL.glUseProgram(color_shader.glid)

        matrix_location = GL.glGetUniformLocation(color_shader.glid, 'matrix')
        # matrix = frustum(-20, 20, -20, 20, -20, 20)
        # print(matrix)
        matrix = perspective(45, 480/640, -20, 100)


        # print(matrix)
        # matrix_S = scale(0.5)
        # matrix[0][0] = matrix_S[0][0]
        # matrix[1][1] = matrix_S[1][1]
        # matrix[2][1] = matrix_S[2][2]
        # matrix[3][3] = matrix_S[3][3]
        # matrix = matrix @ matrix_S

        matrix = matrix @ projection #rotate(vec(0, 0, 1), 45)

        GL.glUniformMatrix4fv(matrix_location, 1, True, matrix)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        GL.glBindVertexArray(self.glid)
        #GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        #use index buffer
        GL.glDrawElements(GL.GL_TRIANGLES, self.index.size, GL.GL_UNSIGNED_INT, None)  # 9 indexed verts = 3 triangles
        GL.glBindVertexArray(0)

    def __del__(self):
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(1, self.buffers)

class SimpleTriangle:
    """Hello triangle object"""

    def __init__(self):

        # triangle position buffer
        #use_vertex_array
        #    position = np.array(((0, .5, 0), (.5, -.5, 0), (-.5, -.5, 0)), 'f')

        #use index array
        position = np.array(((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)), np.float32)
        self.index = np.array((0, 2, 1, 2, 3, 1, 3, 2, 4), np.uint32)
        color = np.array(((1, 1, 0), (1, 1, 0), (1, 0, 0)), 'f')

        self.glid = GL.glGenVertexArrays(1)  # create OpenGL vertex array id
        GL.glBindVertexArray(self.glid)      # activate to receive state below
        self.buffers = [GL.glGenBuffers(1)]   # create buffer for position attrib

        # bind the vbo, upload position data to GPU, declare its size and type
        GL.glEnableVertexAttribArray(0)      # assign to layout = 0 attribute
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[0])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, position, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)

        # bind the vbo, upload position data to GPU, declare its size and type
        self.buffers += [GL.glGenBuffers(1)]
        GL.glEnableVertexAttribArray(1)      # assign to layout = 0 attribute
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[1])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, color, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 0, None)

        self.buffers += [GL.glGenBuffers(1)]                                           # create GPU index buffer
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])                  # make it active to receive
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.index, GL.GL_STATIC_DRAW)     # our index array here


        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self.fovy = 1

    def draw(self, projection, view, model, color_shader):
        GL.glUseProgram(color_shader.glid)

        matrix_location = GL.glGetUniformLocation(color_shader.glid, 'matrix')
        # matrix = frustum(-20, 20, -20, 20, -20, 20)
        # print(matrix)
        matrix = perspective(45, 480/640, -20, 100)


        # print(matrix)
        # matrix_S = scale(0.5)
        # matrix[0][0] = matrix_S[0][0]
        # matrix[1][1] = matrix_S[1][1]
        # matrix[2][1] = matrix_S[2][2]
        # matrix[3][3] = matrix_S[3][3]
        # matrix = matrix @ matrix_S

        matrix = matrix @ projection #rotate(vec(0, 0, 1), 45)

        GL.glUniformMatrix4fv(matrix_location, 1, True, matrix)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        GL.glBindVertexArray(self.glid)
        #GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        #use index buffer
        GL.glDrawElements(GL.GL_TRIANGLES, self.index.size, GL.GL_UNSIGNED_INT, None)  # 9 indexed verts = 3 triangles
        GL.glBindVertexArray(0)

    def __del__(self):
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(1, self.buffers)

class VertexArray:

    def __init__(self, attributes, index=None):

        # triangle position buffer
        #use_vertex_array

        self.glid = GL.glGenVertexArrays(1)  # create OpenGL vertex array id
        GL.glBindVertexArray(self.glid)      # activate to receive state below

        bufferCount = 0;        
        self.buffers = GL.glGenBuffers(len(attributes) + 1)   # create buffers for all attributes
        for layout_index, buffer_data in enumerate(attributes):            
            # bind the vbo, upload position data to GPU, declare its size and type
            GL.glEnableVertexAttribArray(layout_index)      # assign to layout = index
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[layout_index])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, buffer_data, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(layout_index, 3, GL.GL_FLOAT, False, 0, None)

        self.index = index
        # self.buffers += [GL.glGenBuffers(1)]                                           # create GPU index buffer
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])                  # make it active to receive
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.index, GL.GL_STATIC_DRAW)     # our index array here

        # cleanup and unbind so no accidental subsequent statglBufferDatae update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self.fovy = 1

    def draw(self, projection, view, model, color_shader):
        GL.glUseProgram(color_shader.glid)

        matrix_location = GL.glGetUniformLocation(color_shader.glid, 'matrix')


        matrix = perspective(45, 480/640, -20, 100)


        matrix = matrix @ projection #rotate(vec(0, 0, 1), 45)

        GL.glUniformMatrix4fv(matrix_location, 1, True, matrix)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        GL.glBindVertexArray(self.glid)
        #use index buffer
        GL.glDrawElements(GL.GL_TRIANGLES, self.index.size, GL.GL_UNSIGNED_INT, None)  # 9 indexed verts = 3 triangles
        GL.glBindVertexArray(0)

    def __del__(self):
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(1, self.buffers)

class ColorMesh:

    def __init__(self, attributes, index=None):

        # triangle position buffer
        #use_vertex_array

        self.glid = GL.glGenVertexArrays(1)  # create OpenGL vertex array id
        GL.glBindVertexArray(self.glid)      # activate to receive state below

        bufferCount = 0;        
        self.buffers = GL.glGenBuffers(len(attributes) + 1)   # create buffers for all attributes
        for layout_index, buffer_data in enumerate(attributes):            
            # bind the vbo, upload position data to GPU, declare its size and type
            GL.glEnableVertexAttribArray(layout_index)      # assign to layout = index
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[layout_index])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, buffer_data, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(layout_index, 3, GL.GL_FLOAT, False, 0, None)

        self.index = index
        # self.buffers += [GL.glGenBuffers(1)]                                           # create GPU index buffer
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])                  # make it active to receive
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.index, GL.GL_STATIC_DRAW)     # our index array here

        # cleanup and unbind so no accidental subsequent statglBufferDatae update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self.fovy = 1

    def draw(self, projection, view, model, color_shader):
        GL.glUseProgram(color_shader.glid)

        matrix_location = GL.glGetUniformLocation(color_shader.glid, 'matrix')


        matrix = perspective(45, 480/640, -20, 100)


        matrix = matrix @ projection #rotate(vec(0, 0, 1), 45)

        GL.glUniformMatrix4fv(matrix_location, 1, True, matrix)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        GL.glBindVertexArray(self.glid)
        #use index buffer
        GL.glDrawElements(GL.GL_TRIANGLES, self.index.size, GL.GL_UNSIGNED_INT, None)  # 9 indexed verts = 3 triangles
        GL.glBindVertexArray(0)

    def __del__(self):
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(1, self.buffers)

# ------------  Viewer class & window management ------------------------------
class Viewer:
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=640, height=480):

        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)
        self.trackball = GLFWTrackball(self.win)
        self.trackball.distance = max(self.trackball.distance, 120.0)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.1, 0.1, 0.1, 0.1)

        # compile and initialize shader programs once globally
        self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)

        # initially empty list of object to draw
        self.drawables = []

    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            # draw our scene objects
            winsize = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(winsize)
            for obj in self.drawables:
                for mesh in obj:
                    mesh.draw(projection, view, identity(), self.color_shader)


            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables):
        """ add objects to draw in this window """
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)

class GLFWTrackball(Trackball):
    """ Use in Viewer for interactive viewpoint control """

    def __init__(self, win):
        """ Init needs a GLFW window handler 'win' to register callbacks """
        super().__init__()
        self.mouse = (0, 0)
        glfw.set_cursor_pos_callback(win, self.on_mouse_move)
        glfw.set_scroll_callback(win, self.on_scroll)

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.drag(old, self.mouse, glfw.get_window_size(win))
            print("mouse left")
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.pan(old, self.mouse)
            print("mouse right")

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.zoom(deltay, glfw.get_window_size(win)[1])
        print("mouse scroll", self.distance)

# -------------- 3D ressource loader -----------------------------------------
def load(file):
    """ load resources from file using pyassimp, return list of ColorMesh """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []     # error reading => return empty list

    meshes = [ColorMesh([m.vertices, m.normals], m.faces) for m in scene.meshes]
    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes

# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    # place instances of our basic objects
    # viewer.add(SimpleTriangle())

    # position = np.array(((0, 1.4, 0), (1, 0, 1), (-1, 0, 1), (-1, 0, -1), (1, 0, -1)), np.float32)
    # index = np.array((0, 2, 1, 0, 1, 4, 0, 4, 3, 0, 2, 3, 3, 4, 1, 3, 1, 2), np.uint32)
    # color = np.array(((1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 1, 0)), 'f')

    # simplePyramid = VertexArray([position, color], index)
    # viewer.add(simplePyramid)

    obj1 = load('suzanne.obj')
    viewer.add(obj1)

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
