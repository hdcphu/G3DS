import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import pyassimp                     # 3D ressource loader
import pyassimp.errors              # assimp error management + exceptions


from transform import translate, scale, identity, Trackball, sincos
from transform import (lerp, quaternion_slerp, quaternion_matrix, quaternion,
                       quaternion_from_euler, vec)



# -------------- Keyframing Utilities TP6 ------------------------------------
class KeyFrames:
    """ Stores keyframe pairs for any value type with interpolation_function"""
    def __init__(self, time_value_pairs, interpolation_function=lerp):
        if isinstance(time_value_pairs, dict):  # convert to list of pairs
            time_value_pairs = time_value_pairs.items()
        keyframes = sorted(((key[0], key[1]) for key in time_value_pairs))
        self.times, self.values = zip(*keyframes)  # pairs list -> 2 lists
        self.interpolate = interpolation_function
        # if (len(self.times) == 1):
        #     raise Exception("du lieu loi con meo` no roi")


    def value(self, time):
        """ Computes interpolated value from keyframes, for a given time """

        # 1. ensure time is within bounds else return boundary keyframe
        before = after = self.times[0]
        if (time < self.times[0]):
            time = (time - self.times[0]) + self.times[-1]
            
        if (time > self.times[-1]):
            if (self.times[-1] > 0):
                time = time % self.times[-1]
            else: 
                time = 0
        # 2. search for closest index entry in self.times, using bisect_left            
        # 3. using the retrieved index, interpolate between the two neighboring
        # values in self.values, using the stored self.interpolate function
        ret = self.values[0]
        for i in range(0, len(self.times)):
            if (self.times[i] == time):
                ret = self.values[i]
                break
            if (self.times[i] > time):                
                after = i
                before = i - 1
                fraction = (time - self.times[before]) / (self.times[after] - self.times[before])
                ret = self.interpolate(self.values[before], self.values[after], fraction)
                break
        return ret
        


class TransformKeyFrames:
    """ KeyFrames-like object dedicated to 3D transforms """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        """ stores 3 keyframe sets for translation, rotation, scale """
        self.translate_keys = KeyFrames(translate_keys)
        self.rotate_keys = KeyFrames(rotate_keys, quaternion_slerp)
        self.scale_keys = KeyFrames(scale_keys)


    def value(self, time):
        """ Compute each component's interpolation and compose TRS matrix """
        t = self.translate_keys.value(time)
        r = quaternion_matrix(self.rotate_keys.value(time))
        s = self.scale_keys.value(time)
        scale = {1., 1., 1.}
        if (isinstance(s, float) or isinstance(s, int)):
            scale = [s, s, s]
        else:
            scale = s
            
        m = identity()
        for i in range(0, 3, 1):
            for j in range(0, 3, 1):
                m[i][j] = r[i][j] * scale[j]
        for i in range (0, 3, 1):
            m[i][3] = t[i]
        float_formatter = lambda x: "%.2f" % x
        np.set_printoptions(formatter={'float_kind':float_formatter})  
        return m
