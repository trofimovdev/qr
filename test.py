# -*- coding: utf-8 -*-
import cv2
import numpy as np
from math import atan, pi

def ac(img):
    y, u, v = (img * [0.2126, 0.7152, 0.0722]).sum(axis = 2), (img * [-0.0999, -0.3360, 0.4360]).sum(axis = 2),(img * [0.6150, -0.5586, -0.0563]).sum(axis = 2)
    img1 = np.sort(y, axis = None)
    xmin, xmax = img1[round(y.size * 0.05)], img1[-round(y.size * 0.05) - 1]
    y1 = ((y - xmin) * (255 / (xmax - xmin))).clip(0, 255)
    r, g, b = y1 + 1.2803 * v, y1 - 0.2148 * u - 0.3805 * v, y1 + 2.1279 * u
    return np.clip(np.dstack((r, g, b)), 0, 255).round().astype('uint8')

def h(img):
    cdf = np.cumsum(np.histogram(img, range(257))[0])
    return (((cdf[img] - cdf[cdf != 0].min()) / (img.size - 1) * 255).round()).astype('uint8')

def code128(img):
    alphabet = {
                '11010010000': '',
                '10010000110': 'b',
                '10000110100': 'i',
                '11110010010': 'x',
                '11000010100': 'n',
                '10110010000': 'e',
                '11001010000': 'l',
                '10011001110': '.',
                '10010011110': 'r',
                '10011110010': 'u',
                '11000111010': '',
                '11': ''}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a = np.where(gray <= 80)
    if a[1][0] < gray.shape[1] // 2:
        deg = atan((a[0][0] - a[0][np.where(a[1] == a[1].max())[0][0]]) / (a[1][0] - a[1].max())) * (180 / pi)
    else:
        deg = atan((a[0][0] - a[0][np.where(a[1] == a[1].min())[0][0]]) / (a[1][0] - a[1].min())) * (180 / pi)
    rows, cols = gray.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, 1)
    img_rotated = cv2.warpAffine(img, M, (cols, rows), borderValue = (255, 255, 255))
    gray_rotated = cv2.warpAffine(gray, M, (cols, rows), borderValue = (255, 255, 255))
    a = np.where(gray_rotated <= 80)
    c = np.array(['1' if i < 128 else '0' for i in gray_rotated[6]])
    module = np.where(c == '0')[0][0] // 2
    c = ''.join([c[s] for s in range(0, len(c), module)])
    p = [alphabet[c[s : s + 11]] for s in range(11, len(c) - 3 * 11, 11)]
    return (''.join(p), module)


def qr(img):
    masks = {'000': lambda j, i: (i + j) % 2 == 0,
             '001': lambda j, i: i % 2 == 0,
             '010': lambda j, i: j % 3 == 0,
             '011': lambda j, i: (i + j) % 3 == 0,
             '100': lambda j, i: ((i // 2) + (j // 3)) % 2 == 0,
             '101': lambda j, i: (i * j) % 2 + (i * j) % 3 == 0,
             '110': lambda j, i: ((i * j) % 2 + (i * j) % 3) % 2 == 0,
             '111': lambda j, i: ((i + j) % 2 + (i * j) % 3) % 2 == 0}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a = np.where(gray <= 80)
    if a[1][0] < gray.shape[1] // 2:
        deg = atan((a[0][0] - a[0][np.where(a[1] == a[1].max())[0][0]]) / (a[1][0] - a[1].max())) * (180 / pi)
    else:
        deg = atan((a[0][0] - a[0][np.where(a[1] == a[1].min())[0][0]]) / (a[1][0] - a[1].min())) * (180 / pi)
    rows, cols = gray.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, 1)
    img_rotated = cv2.warpAffine(img, M, (cols, rows), borderValue = (255, 255, 255))
    gray_rotated = cv2.warpAffine(gray, M, (cols, rows), borderValue = (255, 255, 255))
    a = np.where(gray_rotated <= 128)
    gray_cropped = gray_rotated[a[0][0] : a[0][-1], a[0][0] : a[0][-1]]


    #~ code = [['1' if gray_cropped[y][i] < 128 else '0' for i in range(0, len(gray_cropped[y]), 8)] for y in range(0, len(gray_cropped), 8)]
    code = np.array([['1' if gray_cropped[y][i] < 128 else '0' for i in range(0, len(gray_cropped[y]), 8)] for y in range(0, len(gray_cropped), 8)])
    print('\n'.join(' '.join(i) for i in code))

    sys_info = bin(int(''.join(code[8][0 : 5]), 2) ^ int('10101', 2))[2:].rjust(5, '0')
    print(code[8][0 : 5])
    print(''.join(code[8][0 : 5]), sys_info)
    #~ sys_info = bin(int(''.join(code[8][0 : 6]), 2) ^ int('10101', 2))[2:].rjust(5, '0')
    cor_level, mask = sys_info[0 : 2], sys_info[2 : 5]
    print('Correction level: %s'%cor_level)
    print('Sys mask: %s'%mask, end = '\n\n')
    temp = code[9:21, 18:16:-1]
    print(temp)
    print(temp.flatten())
    print(''.join(temp.flatten()))
    #~ print(''.join(code[9:21, 19 - 2:19].flatten()))
    #~ data_type = [[code[y][x] for x in range(20, 18, -1)] for y in range(20, 15, -1)]
    cols = ''
    for x, i in enumerate(range(21, 14, -2)):
        cols += ''.join(code[8:21, i - 2:i].flatten()) if x % 2 != 0 else ''.join(code[8:21, i - 2:i].flatten())[::-1]
    #~ for i in range(13, 8, -2):
        #~ cols += ''.join(code[0:21, i - 2:i].flatten())[::-1]
    #~ for i in range(0, len(cols), 24):
        #~ print(cols[i:i + 24], end = ' ')
    print('\n')
    print(cols[:24], cols[24:])
    print('\n' * 4)
    
    #~ data_type = cols[-1][0] + cols[-2][0] + cols[-1][1] + cols[-2][1]
    data_type = str(cols[:4])    

    m = "%i%i%i%i"%(int(masks[mask](20, 20)), int(masks[mask](19, 20)), int(masks[mask](20, 19)), int(masks[mask](19, 19)))
    print('Data type: %s'%(str(int(data_type, 2) ^ int(m, 2))), m)
    data_type = str(int(data_type, 2) ^ int(m, 2)).rjust(4, '0')
    
    print('Data type: %s'%data_type)
    #~ info = bin(int(cols[4:96], 2) ^ int(cor_level + ''.join([m + cor_level for i in range(15)]), 2))[2:].rjust(20, '0')
    info = bin(int(cols[4:96], 2) ^ int(((str(cor_level) + str(m)) * 16)[:92], 2))[2:].rjust(92, '0')
    
    #~ print(((str(cor_level)+str(m))*16)[:92])
    #~ print(cor_level + ''.join([m + cor_level for i in range(15)]))
    read_mask = (cor_level + m) * 34 + cor_level
    read_cols = str(bin(int(cols[4:], 2) ^ int(read_mask[:len(cols[4:])], 2)))[2:].rjust(len(cols[4:]), '0')
    p = int(read_cols[:10], 2)
    print('Packages: %s'%p)
    #~ print(cor_level, m)
    #~ done = [read_cols[19 + i : 9 + i : -1] for i in range(0, p * 10, 10)]
    done = [read_cols[10 + i : 14 + i] for i in range(0, p * 4, 4)]
    
    print(read_cols[:10], read_cols[10:])
    print(done)
    print('\n')
    #~ print(str(bin(int(cols[4:], 2) ^ int(read_mask[:len(cols[4:])], 2)))[2:])
    #~ print(read_cols)
    print(cols[:50])
    print('\n')
    print('\n' * 4)
    # брать по 10 бит и делать инверсию 
    if data_type == '0001':
        for i in done:
            print(int(i, 2))
    
    #~ print(cols[:8], cols[4 : 12], cols[12 : 20], cols[20 : 28], cols[28 : 36], cols[36 : 44])
    #~ print(list(map(lambda x: int(x, 2), [cols[:8], cols[4 : 12], cols[12 : 20], cols[20 : 28], cols[28 : 36], cols[36 : 44]])))
    #~ a = [cols[i : i + 8] for i in range(p, p ** 2, 8)]
    #~ print(a)
    cv2.imshow('cropped', gray_cropped)
    cv2.imshow('default', gray)
    cv2.waitKey(1000000)
    
print(qr(cv2.imread('qrcode12.png')))




