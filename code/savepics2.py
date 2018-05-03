#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
在网页上爬取图片
...
# Created by David Teng on 18-5-2
import requests

from scrapy import Selector
import os

def listfiles(dirpath):
    return len(os.listdir(dirpath))
    pass

def savepics():
    """

    :return:
    """

    headers = {
        "Host": "sdimgs.pandengit.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.9 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "http://mail.qq.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control":"max-age=0"
    }
    url1 = "http://sdimgs.pandengit.com/"
    dirctorylist = ["0ffffea6a501/", "0ffffea6a502/"]
    path = '/home/wlw/project/0503indata/'
    for diritem in dirctorylist:
        res = requests.get(url1+diritem, headers=headers)
        sel = Selector(text=res.text)
        linklist = sel.xpath("//li/a/@href").extract()
        linklist = [p for p in linklist if '.jpg' in p]
        print len(linklist)
        if os.path.exists(path + diritem):
            pass
        else:
            os.mkdir(path + diritem)
        savedpicslist = os.listdir(path + diritem)
        for link in linklist:
            if link not in savedpicslist:
                img = requests.get(url1 + diritem +link)
                with open(path + diritem+link, "wb")as wr:
                    # print img.content
                    wr.write(img.content)

                print u"图片:%s 下载完成！..." % link


if __name__ == '__main__':
    savepics()
    # print listfiles("./0ffffea6a502/")
    pass
