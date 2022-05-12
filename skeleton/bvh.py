import re
import numpy as np
from utils.convert import Convert
from functools import reduce
from utils.utils import vector_to_string

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
    def generate(motion, fps):
        skeleton = motion.skeleton
        channels = [[] for _ in range(len(skeleton.joints))]
        channels[0] = ['Xposition', 'Yposition', 'Zposition']
        for axis in skeleton.order:
            for channel in channels:
                channel.append(axis.upper() + 'rotation')
        position = np.array(motion.root_pos)[:, np.newaxis, :]
        rotation = Convert.quaternion_to_euler(motion.rotations, skeleton.order)
        data = np.concatenate((position, rotation), 1)
        data = data.reshape(motion.frame, -1)
        return BVH((skeleton.joints, channels, motion.frame, fps,
            skeleton.parents, skeleton.offsets, skeleton.end_sites, data))

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
                end_offsets.append([])
                active_joint = len(parents) - 1
            if pattern_index == 2:
                end_site = True
            elif pattern_index == 5:
                if end_site: end_site = False
                else: active_joint = parents[active_joint]
            elif pattern_index == 6:
                offset = np.array(list(map(float, match_obj.groups())))
                if end_site: end_offsets[active_joint].append(offset)
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
        
        end_offsets = [np.array(x) for x in end_offsets]
        data_block = []
        assert(len(content) > index + frames)
        for i in range(index + 1, index + 1 + frames):
            data = content[i].split(' ')[:number]
            data_block.append([float(x) for x in data])
        return BVH((names, channels, frames, fps, np.array(parents),
               np.array(offsets), end_offsets, np.array(data_block)))

    def __joint_buffer__(self, index):
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
        for endsite in self.end_offsets[index]:
            buffer.append('\tEnd Site')
            buffer.append('\t{')
            buffer.append('\t\tOFFSET ' + vector_to_string(endsite))
            buffer.append('\t}')
        else:
            for child in children:
                child_buffer = self.__joint_buffer__(child)
                child_buffer = ['\t' + x for x in child_buffer]
                buffer.extend(child_buffer)
        buffer.append('}')
        return buffer

    def bvh_save(self, filename):
        buffer = ['HIERARCHY']
        buffer.extend(self.__joint_buffer__(0))
        buffer[1] = 'ROOT ' + self.names[0]
        buffer.append('MOTION')
        buffer.append('Frames: ' + str(self.frames))
        buffer.append('Frame Time: ' + str(1 / self.fps))
        for data in self.data_block:
            buffer.append(vector_to_string(data))
        buffer = reduce(lambda x, y: x + y, [x + '\n' for x in buffer])
        with open(filename, 'w') as file:
            file.write(buffer)