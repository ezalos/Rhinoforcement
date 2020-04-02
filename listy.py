import sys
from time import sleep
import time
from color import *


def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def get_time(time):
    if time // 10 < 1:
        str_time = ' '
    else:
        str_time = ''
    str_time += str(truncate(time, 2))
    str_time += "s "
    return str_time

def get_percent(i, total):
    if i < total // 10:
        percent = "[ "
    else:
        percent = "["
    percent += str((((i + 1) * 100) // total))
    percent += '%'
    percent += ']' 
    return percent

def get_arrow(i, total):
    unit = int(total / 20)
    if unit == 0:
        unit = 1
    arrow =  '[' 
    arrow += (i // unit) * '=' 
    arrow += '>' 
    arrow += ((total - i - 1) // unit) * ' ' 
    arrow += '] ' 
    return arrow

def get_count(i, total):
    count = str(i + 1) + '/' + str(total) 
    return count

def ft_progress(listy):
    total = len(listy)
    unit = int(total / 20)
    for i in listy:
        if (i == 0):
            start_time = time.time()
        elapsed_time = (time.time()) - (start_time)
        current_time = get_time(elapsed_time)
        eta = get_time((elapsed_time) * (total - i) / (i + 1))
        percent = get_percent(i, total)
        arrow = get_arrow(i, total)
        count = get_count(i, total)
        print("ETA: ", eta, percent, arrow, count,
        " | elapsed time ", current_time, end="\r")
        yield i
    print("")

