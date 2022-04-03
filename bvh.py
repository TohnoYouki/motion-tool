import re
import numpy as np
from functools import reduce
from skeleton import Skeleton

def vector_to_string(vector):
    result = ''
    for value in vector: result += str(value) + ' '
    return result[:-1] 

class BVH:
    def __init__(self, data):
        names, channels, frames, fps, parents, \
        offsets, end_offsets, data_block = data
        self.names = names
        self.channels = channels
        self.frames = frames
        self.fps = fps
        self.parents = parents
        self.offsets = offsets
        self.end_offsets = end_offsets
        self.data_block = data_block

    @staticmethod
    def generate(skeleton):
        assert(isinstance(skeleton, Skeleton))
        channel = []
        return BVH((skeleton.joints, '', skeleton.frame, '',
                    skeleton.parent, skeleton.offset, skeleton.end_site, ''))
        '''
        offset, orientation = [], []
        position = [self.root_pos[i].copy() for i in range(self.number)]
        for joint in self.joints: offset.append(self.offset[joint].copy())
        for i in range(self.number):
            orientation.append([])
            for joint in self.joints:
                euler = self.orientation[i][joint].as_euler('yxz', degrees = True)
                orientation[-1].append([euler[2], euler[1], euler[0]])
        end_site = [value for value in self.end_site]
        return offset, position, orientation, end_site
        '''
    @staticmethod
    def load(filename): 
        with open(filename, 'r') as file:
            content = list(file)

        offsets, end_offsets = [], []
        names, parents, channels = [], [], []

        patterns = [r'ROOT (\w+)', '\s*JOINT\s+(\w+)', '\s*End Site', 
                    'HIERARCHY','\s*{', '\s*}', 
                r'\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)',
                r'\s*CHANNELS\s+(\d+)', 
                'MOTION', '\s*Frames:\s+(\d+)', '\s*Frame Time:\s+([\d\.]+)']
        active_joint, end_site, frames, fps, number = -1, False, 0, 0, 0

        for index, line in enumerate(content):
            mresults = [re.match(pattern, line) for pattern in patterns]
            mresults = list(zip(range(len(patterns)), mresults))
            success_patterns = [x for x in mresults if x[1] is not None]
            assert(len(success_patterns) == 1)
        
            pattern_index, match_obj = success_patterns[0]
            if pattern_index < 2:
                names.append(match_obj.group(1))
                offsets.append(np.zeros(3))
                parents.append(active_joint)
                end_offsets.append(None)
                active_joint = len(parents) - 1
            if pattern_index == 2:
                end_site = True
            elif pattern_index == 5:
                if end_site: end_site = False
                else: active_joint = parents[active_joint]
            elif pattern_index == 6:
                offset = np.array(list(map(float, match_obj.groups())))
                if end_site: end_offsets[active_joint] = offset
                else: offsets[active_joint] = offset
            elif pattern_index == 7:
                number += int(match_obj.group(1))
                channels.append(line.split()[2:])
                assert(len(channels[-1]) == int(match_obj.group(1)))
            elif pattern_index == 9:
                frames = int(match_obj.group(1))
            elif pattern_index == 10:
                fps = int(1 / float(match_obj.group(1)))
                break
    
        data_block = []
        assert(len(content) > index + frames)
        for i in range(index + 1, index + 1 + frames):
            data = content[i].split(' ')[:number]
            data_block.append([float(x) for x in data])
        return BVH((names, channels, frames, fps, np.array(parents),
               np.array(offsets), end_offsets, np.array(data_block)))

    def joint_buffer(self, index):
        buffer = []
        buffer.append('JOINT ' + self.names[index])
        buffer.append('{')
        offset = self.offsets[index]
        buffer.append('\tOFFSET ' + vector_to_string(offset))
        buffer.append('\tCHANNELS ' + str(len(self.channels[index])))
        for channel in self.channels[index]: 
            buffer[-1] += ' ' + channel
        children = []
        for i, parent in enumerate(self.parents):
            if parent == index: children.append(i)
        if self.end_offsets[index] is not None:
            assert(len(children) == 0)
            buffer.append('\tEnd Site')
            buffer.append('\t{')
            offset = self.end_offsets[index]
            buffer.append('\t\tOFFSET ' + vector_to_string(offset))
            buffer.append('\t}')
        else:
            for child in children:
                child_buffer = self.joint_buffer(child)
                child_buffer = ['\t' + x for x in child_buffer]
                buffer.extend(child_buffer)
        buffer.append('}')
        return buffer

    def bvh_save(self, filename):
        buffer = ['HIERARCHY']
        buffer.extend(self.joint_buffer(0))
        buffer[1] = 'ROOT ' + self.names[0]
        buffer.append('MOTION')
        buffer.append('Frames: ' + str(self.frames))
        buffer.append('Frame Time: ' + str(1 / self.fps))
        for data in self.data_block:
            buffer.append(vector_to_string(data))
        buffer = reduce(lambda x, y: x + y, [x + '\n' for x in buffer])
        with open(filename, 'w') as file:
            file.write(buffer)