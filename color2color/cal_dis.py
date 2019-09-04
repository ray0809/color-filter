
import sys
import time
import json
import imghdr
import numpy as np
from PIL import Image
from colorthief import ColorThief


from CIEDE2000 import *





class CalColorDis():
    '''根据输入的图像判断跟各个颜色的距离
    输入：
    @rgbs：各个基色的rgb值
    @hexs：各个基色的16进制
    @fake_hexs：各个基色在web上的对应16进制

    '''
    def __init__(self, json_path):
        self.hexs, self. rgbs = self._parsingJson(json_path)

    def rgb_dis(self, img):
        '''计算rgb距离
        输入：
        @img：rgba通道的Image图像
        输出：
        @MAP：跟各个基色的距离，用字典返回，key是颜色16进制，value是距离值
        '''
        try:
            color_thief = ColorThief(img)
            palette, white_ratio = color_thief.get_palette(color_count=8, quality=5)
            palette = [(i[-1] / 100.0, i[:3]) for i in  palette]
            palette = sorted(palette, key=lambda x:x[0], reverse=True)


            MAP = {}
            for key in set(self.hexs):
                MAP[key] = 1

            if len(palette) == 1:
                dis = []
                for rgb in self.rgbs:
                    dis.append(round(dist_rgb(palette[0][1], rgb) / 255.0, 5))

                index = np.argmin(dis)
                key = self.hexs[index]
                MAP[key] = dis[index]



            else:
                ratios1, rgb1 = palette[0]
                ratios2, rgb2 = palette[1]
                #如果两种颜色占比相差不大，那么主色同级,或者是前两个颜色基本构成了整张图的颜色分布
                if ((ratios1 - ratios2) <= 0.3) or ((ratios1 + ratios2) >= 0.8 and (ratios2 >= 0.2)):
                    dis1 = []
                    dis2 = []
                    for rgb in self.rgbs:
                        dis1.append(round(dist_rgb(rgb1, rgb) / 255.0, 5))
                        dis2.append(round(dist_rgb(rgb2, rgb) / 255.0, 5))

                    index1 = np.argmin(dis1)
                    key1 = self.hexs[index1]

                    index2 = np.argmin(dis2)
                    key2 = self.hexs[index2]

                    if key1 == key2:
                        MAP[key1] = dis1[index1] if (dis1[index1] < dis2[index2]) else dis2[index2]
                    else:
                        MAP[key1] = dis1[index1]
                        MAP[key2] = dis2[index2]
                else:
                    dis = []
                    for rgb in self.rgbs:
                        dis.append(round(dist_rgb(palette[0][1], rgb) / 255.0, 5))

                    index = np.argmin(dis)
                    key = self.hexs[index]
                    MAP[key] = dis[index]

            if white_ratio >= 0.35:
                MAP['#FFFFFF'] = 0.0

            return MAP

        except Exception as e:
            print('error:{}'.format(e))
            return {i:1 for i in set(self.hexs)}

    def _parsingJson(self, json_path):
        '''
        解析json，将每个颜色下的rgb都罗列出来
        '''
        with open(json_path, 'r') as f:
            colors = json.load(f)

        hexs = []
        rgbs = []
        for key, value in colors.items():
            for v in value:
                rgbs.append(tuple(v))
                hexs.append(key)

        return hexs, rgbs


if __name__ == '__main__':
    #自己测试，输入图像路径
    path = sys.argv[1]

    if imghdr.what(path):
        #初始化
        calcolordis = CalColorDis('./colors.json')
        img = Image.open(path).convert('RGBA')


        #返回结果
        begin = time.time()
        MAP = calcolordis.rgb_dis(img)
        print('耗时:{:.3f}ms'.format((time.time() - begin) * 1000))
        # img.show()



        #打印结果
        result = sorted(MAP.items(), key=lambda x:x[1])[:2]
        print(result)