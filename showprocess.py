# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:53:54 2017

@author: admin
"""

import sys, time
import threading

class ProgressBar:
    """
    显示处理进度的类
    调用该类show_process函数即可实现处理进度的显示
    """
    # 初始化函数，需要知道总共的处理次数
    def __init__(self, total=0, width=50):        
        self.total = total   # how many iter counts totally? 
        self.width = width
        sys.stdout.write('\r')
        sys.stdout.flush()  
        
    def show_process(self, count):
        count = count + 1
#        time.sleep(0.01)        
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        progress = self.width * count // self.total
        sys.stdout.write('{0:3}/{1:3}: '.format(count, self.total))
        sys.stdout.write('[' + '>' * progress + '=' * (self.width - progress) + ']' + '\r')        
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()

    def update(self, count):
        t = threading.Thread(target=self.show_process, args=(count,))
        t.start()
        t.join()

if __name__ == '__main__':
    example_bar = ProgressBar(50000)
    for i in range(50000):
#        example_bar.show_process(i)
        example_bar.update(i)

